import math
import os

import numpy as np
import matplotlib.pyplot as plt
import nengo_spa as spa
import pickle as pkl
import torch
import plotly.graph_objects as go
from scipy.ndimage import maximum_filter, label
from datetime import datetime
from matplotlib.patches import Rectangle, Polygon, Circle
from sspspace import HexagonalSSPSpace, SSP, ssp

SHAPES = [
    # walls
    'square',
    # objects
    'heart', 'clove_tree', 'moon', 'wine', 'double_dia',
    'flag', 'capsule', 'vase', 'curved_triangle', 'spoon',
    'medal',  # inaccessible goal
    # home
    'home_shape',
    # agent
    'pentagon', 'clove', 'kite',
    # key
    'triangle0', 'triangle180', 'triangle90', 'triangle270',
    # lock
    'triangle_slot0', 'triangle_slot180', 'triangle_slot90', 'triangle_slot270'
]

ENTITIES = [
    'agent', 'walls', 'fuse_walls', 'key',
    'blocking', 'home_entity', 'objects', 'pin', 'lock'
]


# helper class
class Element:
    def __init__(self):
        self.correct = 0
        self.false = 0
        self.average_distance = 0
        self.average_distance_non_correct = 0
        self.median_distance = []
        self.median_distance_non_correct = []

    def to_dict(self):
        return {
            "correct": self.correct,
            "false": self.false,
            "average_distance": self.average_distance,
            "average_distance_non_correct": self.average_distance_non_correct,
            "median_distance": self.median_distance,
            "median_distance_non_correct": self.median_distance_non_correct
        }


class Result:
    def __init__(self, n_rotations: int = 13, n_scale: int = 13, length_scale: int = 1):
        self.img_amount = 0
        self.dimensions = 0
        self.length_scale = length_scale
        self.n_rotations = n_rotations
        self.n_scale = n_scale
        self.correct_overall = 0
        self.false_overall = 0
        self.average_distance_overall = 0
        self.average_distance_overall_non_correct = 0
        self.median_distance_overall = []
        self.median_distance_overall_non_correct = []
        self.Elements = {
            "agent": Element(),
            "walls": Element(),
            "key": Element(),
            "blocking": Element(),
            "fuse_walls": Element(),
            "home_entity": Element(),
            "objects": Element(),
            "pin": Element(),
            "lock": Element()
        }

    def to_dict(self):
        return {
            "img_amount": self.img_amount,
            "dimensions": self.dimensions,
            "length_scale": self.length_scale,
            "n_rotations": self.n_rotations,
            "n_scale": self.n_scale,
            "correct_overall": self.correct_overall,
            "false_overall": self.false_overall,
            "average_distance_overall": self.average_distance_overall,
            "average_distance_overall_non_correct": self.average_distance_overall_non_correct,
            "median_distance_overall": self.median_distance_overall,
            "median_distance_overall_non_correct": self.median_distance_overall_non_correct,
            "Elements": {key: element.to_dict() for key, element in self.Elements.items()}
        }


def plot_3d_heatmap(sims_map):
    # 3d plot
    x = np.arange(sims_map.shape[1])  # x-Werte: 0 bis 199
    y = np.arange(sims_map.shape[0])  # y-Werte: 0 bis 199
    X, Y = np.meshgrid(x, y)
    # Erstelle den interaktiven 3D-Plot
    fig = go.Figure(data=[go.Surface(z=sims_map, x=X[0], y=Y[:, 0], colorscale='Viridis')])
    # Layout und Labels
    fig.update_layout(
        title="Interactive 3D Plot of Sims Map",
        scene=dict(
            xaxis_title="X values",
            yaxis_title="Y values",
            zaxis_title="Map Values",
        ),
        margin=dict(l=0, r=0, t=50, b=0)  # Reduziert leere Ränder
    )
    # Zeige den Plot
    fig.show()
    fig.write_html("interactive_3d_plot_25-25-5.html")


def compute_new_array(ssp_grid, env_ssp, vocab_combined):
    num_samples = ssp_grid.shape[0]
    new_array = np.empty(num_samples, dtype=object)

    # Iteriere über alle SSPs
    for idx in range(num_samples):
        ssp = ssp_grid[idx].reshape(1, -1)

        inverted_ssp = ~ssp
        bound_ssp = inverted_ssp * env_ssp

        best_key = None
        best_similarity = -np.inf

        for key, vocab_ssp in vocab_combined.items():
            similarity = np.dot(bound_ssp, vocab_ssp.T)
            if similarity > best_similarity and similarity > 0.7:
                best_similarity = similarity
                best_key = key

        new_array[idx] = best_key

    # Keys in eine numerische Darstellung umwandeln
    keys = list(set(new_array))
    key_to_int = {key: i for i, key in enumerate(keys)}
    int_array = np.array([key_to_int[key] for key in new_array]).reshape(200, 200)
    int_array = np.rot90(int_array, k=1)
    # Erstelle die Heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(int_array, cmap='tab10', origin='lower')  # Verwende eine qualitative Farbcodierung
    cbar = plt.colorbar()
    cbar.set_ticks(range(len(keys)))
    cbar.set_ticklabels(keys)
    plt.title('Heatmap of Keys', fontsize=16)
    plt.xlabel('X-axis', fontsize=12)
    plt.ylabel('Y-axis', fontsize=12)
    plt.show()
    pass


def plot_line_at_y(similarity_map, y_positions):
    plt.figure(figsize=(10, 6))

    # Iteriere über alle y-Positionen
    for y in y_positions:
        values_at_y = similarity_map[y, :]  # Extrahiere die Werte bei der aktuellen y-Position
        plt.plot(range(similarity_map.shape[1]), values_at_y, label=f'y = {y}', linewidth=2)

    # Beschriftung und Titel
    plt.title('Line Chart for Multiple y Positions on Similarity Map', fontsize=16)
    plt.xlabel('x Values', fontsize=12)
    plt.ylabel('Similarity Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Plot anzeigen
    plt.show()


def find_top_k_points(sims_map, k=10):
    flat_map = sims_map.ravel()
    indices = np.argsort(flat_map)[::-1]
    top_k_indices = indices[:k]
    top_k_values = flat_map[top_k_indices]

    top_k_coords = np.array(
        np.unravel_index(top_k_indices, sims_map.shape)).T

    return top_k_coords, top_k_values


class SSPConstructor:

    def __init__(self, n_rotations: int = 13, n_scale: int = 13, length_scale: int = 1):
        self.n_scale = n_scale
        self.n_rotations = n_rotations
        self.RES_X, self.RES_Y = 200, 200
        self.length_scale = length_scale
        self.counter = 1
        self.results = Result(n_rotations=n_rotations, n_scale=n_scale, length_scale=length_scale)
        self.ssp_grid, self.ssp_space, self.vocab_combined = self.init_ssp_constructor()
        self.results.dimensions = self.ssp_space.domain_dim

    def generate_env_ssp(self, grid_objects_param, mode):
        # init empty environment
        global_env_ssp = SSP(np.zeros(self.ssp_space.ssp_dim).reshape((1, -1)))
        self.results.img_amount = self.results.img_amount + 1
        # bind and add alls object ssps
        for obj in grid_objects_param:
            obj_type = get_value_from_string(obj.name['type'])
            obj_shape = get_value(obj.name['shape'], "shape")
            obj_x = int(float(obj.name['x']))
            obj_y = int(float(obj.name['y']))

            object_ssp = self.vocab_combined[f"{obj_type.upper()}_{obj_shape.upper()}"]
            position_ssp = self.ssp_space.encode([[obj_x, obj_y]])

            object_pos_ssp = object_ssp * position_ssp
            global_env_ssp = global_env_ssp + object_pos_ssp
        # example visualization of image
        # self._print_image(grid_objects_param)

        # create results
        self.update_results(global_env_ssp, grid_objects_param)
        # example extracted visualization
        # self._print_extracted_image(global_env_ssp, grid_objects_param)
        return global_env_ssp
        # return vector = "idk yet"

    def init_ssp_constructor(self):
        filename = 'data/external/bib_train/vocab_and_ssp_grid.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pkl.load(f)
        else:
            # Create SSP spaces for xy coordinates and labels
            xs = np.linspace(0, self.RES_X, self.RES_X)
            ys = np.linspace(0, self.RES_Y, self.RES_Y)
            x, y = np.meshgrid(xs, ys)
            xys = np.vstack((x.flatten(), y.flatten())).T

            ssp_space = HexagonalSSPSpace(domain_dim=2, n_rotates=self.n_rotations, n_scales=self.n_scale)
            ssp_space.update_lengthscale(self.length_scale)

            # Generate xy coordinates
            ssp_grid = ssp_space.encode(xys)

            # Generate object SSPs
            vocab = spa.Vocabulary(dimensions=ssp_space.ssp_dim)
            vocab_simple = {}
            vocab_shape_helper = {}
            vocab_type_helper = {}
            vocab_combined = {}

            # Add semantic pointers for each shape to the vocabulary
            for i, shape_name in enumerate(SHAPES):
                shape_vector = vocab.algebra.create_vector(ssp_space.ssp_dim, properties={"positive", "unitary"})
                shape_ssp = SSP(shape_vector)
                vocab_simple[f"{shape_name.upper().replace(' ', '')}"] = shape_ssp
                vocab_shape_helper[f"{shape_name.upper().replace(' ', '')}"] = shape_ssp

            # Add semantic pointers for each entity to the vocabulary
            for i, entity_name in enumerate(ENTITIES):
                entity_vector = vocab.algebra.create_vector(ssp_space.ssp_dim, properties={"positive", "unitary"})
                entity_ssp = SSP(entity_vector)
                vocab_simple[f"{entity_name.upper().replace(' ', '')}"] = entity_ssp
                vocab_type_helper[f"{entity_name.upper().replace(' ', '')}"] = entity_ssp

            # combine ssps
            for type_key, type_ssp in vocab_type_helper.items():
                for shape_key, shape_ssp in vocab_shape_helper.items():
                    object_ssp = type_ssp * shape_ssp
                    vocab_combined[f"{type_key.upper()}_{shape_key.upper()}"] = object_ssp
            with open(filename, 'wb') as f:
                pkl.dump([ssp_grid, ssp_space, vocab_combined], f)
            return ssp_grid, ssp_space, vocab_combined

    def _print_extracted_image(self, global_env_ssp_param, grid_objects):
        # compute_new_array(self.ssp_grid, global_env_ssp_param, self.vocab_combined)
        walls_count = 0
        for obj in grid_objects:
            if get_value_from_string(obj.name['type']) == 'walls':
                walls_count += 1
        for obj in grid_objects:
            obj_type = get_value_from_string(obj.name['type'])
            if obj_type == 'walls':
                obj_type = get_value_from_string(obj.name['type'])
                obj_shape = get_value(obj.name['shape'], "shape")
                object_ssp = self.vocab_combined[f"{obj_type.upper()}_{obj_shape.upper()}"]

                inv_ssp = ~object_ssp

                # get similarity map of label with all locations by binding with inverse ssp
                out = global_env_ssp_param * inv_ssp
                sims = self.ssp_grid | out
                sims_map = sims.reshape((self.RES_X, self.RES_Y))
                sims_map = np.rot90(sims_map, k=1)
                # plot_line_at_y(sims_map, [10, 90])

                top_k_coords, top_k_values = find_top_k_points(sims_map=sims_map, k=walls_count)
                print("Koordinaten der größten Warscheinlichkeiten:", top_k_coords)
                # plot_3d_heatmap(sims_map)

                pred_loc = np.array(np.unravel_index(np.argmax(sims_map), sims_map.shape))
                x_correct = obj.y
                y_correct = 200 - obj.x
                distance = abs(math.sqrt((pred_loc[1] - x_correct) ** 2 + (pred_loc[0] - y_correct) ** 2))

                print(
                    f'{obj_type.upper()} predicted location: {pred_loc[1], pred_loc[0]}, correct location: {x_correct, y_correct}, distance between: {distance}')
                # img
                plt.imshow(sims_map, cmap='plasma', origin='lower', extent=(0, self.RES_X, 0, self.RES_Y),
                           interpolation='none')
                plt.colorbar()
                plt.xticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
                plt.yticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
                plt.show()
                pass

    def _print_image(self, grid_objects):
        fig, ax = plt.subplots()
        global_grid = np.zeros((self.RES_X, self.RES_Y))
        plt.imshow(global_grid, extent=[0, 200, 0, 200], cmap='gray')  # Setze extent auf [0, 200, 0, 200]
        plt.xticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, self.RES_X])  # x-Achse bei 0, 100 und 200
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, self.RES_Y])  # y-Achse bei 0, 100 und 200
        # Gitterlinien
        grid_color = (.3, .3, .3, .5)
        grid_spacing = 20  # Abstand der Gitterlinien in Pixeln
        for i in range(0, self.RES_X, grid_spacing):  # Setze Gitterlinien im Abstand von 20
            plt.hlines(i + grid_spacing, 0, self.RES_Y, color=grid_color, linewidth=1.0)
            plt.vlines(i + grid_spacing, 0, self.RES_X, color=grid_color, linewidth=1.0)
        # add elements
        for object_element in grid_objects:
            x_pos_nulled = object_element.x - 10 + 20
            y_pos_nulled = object_element.y - 10
            # walls
            if get_value_from_string(object_element.name['type']) == "walls":
                ax.add_patch(
                    Rectangle((y_pos_nulled, self.RES_X - x_pos_nulled), width=20, height=20, color='gray',
                              zorder=3))
            # agent
            if get_value_from_string(object_element.name['type']) == "agent":
                ax.add_patch(Polygon(
                    [[y_pos_nulled, self.RES_X - x_pos_nulled], [y_pos_nulled, self.RES_X - x_pos_nulled + 20],
                     [y_pos_nulled + 20, self.RES_X - x_pos_nulled + 10]], closed=True, color='r', zorder=5))
            # objects
            if get_value_from_string(object_element.name['type']) == "objects":
                ax.add_patch(Circle((y_pos_nulled, self.RES_X - x_pos_nulled), radius=10, color='g', zorder=5))
            if get_value_from_string(object_element.name['type']) == "home_entity":
                ax.add_patch(
                    Rectangle((y_pos_nulled, self.RES_X - x_pos_nulled), width=15, height=15, color='m', zorder=5))
        plt.title('Global Environment')
        plt.show()

    def update_results(self, global_env_ssp_param, grid_objects_param):
        walls_count = 0
        for obj in grid_objects_param:
            if get_value_from_string(obj.name['type']) == 'walls':
                walls_count += 1
        for obj in grid_objects_param:
            obj_type = get_value_from_string(obj.name['type'])
            obj_shape = get_value(obj.name['shape'], "shape")
            object_ssp = self.vocab_combined[f"{obj_type.upper()}_{obj_shape.upper()}"]

            inv_ssp = ~object_ssp

            # get similarity map of label with all locations by binding with inverse ssp
            out = global_env_ssp_param * inv_ssp
            sims = self.ssp_grid | out
            sims_map = sims.reshape((self.RES_X, self.RES_Y))
            sims_map = np.rot90(sims_map, k=1)

            top_k_coords, top_k_values = find_top_k_points(sims_map, k=walls_count)
            pred_loc = np.array(np.unravel_index(np.argmax(sims_map), sims_map.shape))
            x_correct = obj.y
            y_correct = 200 - obj.x

            correct = 0
            false = 0
            distance = 0

            if obj_type != 'walls':
                distance = abs(math.sqrt((pred_loc[1] - x_correct) ** 2 + (pred_loc[0] - y_correct) ** 2))
                if distance < 5:
                    correct = 1
                else:
                    false = 1
            else:
                for top_coord in top_k_coords:
                    distance = abs(math.sqrt((top_coord[1] - x_correct) ** 2 + (top_coord[0] - y_correct) ** 2))
                    if distance < 5:
                        correct = 1
                        false = 0
                        break
                    else:
                        false = 1

            self.results.correct_overall = self.results.correct_overall + correct
            self.results.false_overall = self.results.false_overall + false
            self.results.average_distance_overall = (self.results.average_distance_overall + distance) / 2
            self.results.median_distance_overall.append(distance)
            if false:
                self.results.average_distance_overall_non_correct = (
                                                                            self.results.average_distance_overall_non_correct + distance) / 2
                self.results.median_distance_overall_non_correct.append(distance)

            self.results.Elements[obj_type].correct = self.results.Elements[obj_type].correct + correct
            self.results.Elements[obj_type].false = self.results.Elements[obj_type].false + false
            self.results.Elements[obj_type].average_distance = (self.results.Elements[
                                                                    obj_type].average_distance + distance) / 2
            self.results.Elements[obj_type].median_distance.append(
                distance)
            if false:
                self.results.Elements[obj_type].average_distance_non_correct = (self.results.Elements[
                                                                                    obj_type].average_distance_non_correct + distance) / 2
                self.results.Elements[
                    obj_type].median_distance_non_correct.append(distance)


def get_value_from_string(indexes_as_string):
    indexes = np.fromstring(indexes_as_string.replace('[', '').replace(']', ''), sep=' ')
    return get_value(indexes)


def get_value(indexes_param, attribute="type"):
    index_of_one = np.argmax(indexes_param - 1)
    if attribute == "type":
        return ENTITIES[index_of_one]
    elif attribute == "shape":
        return SHAPES[index_of_one]
