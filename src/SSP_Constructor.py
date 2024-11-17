import math
import os

import numpy as np
import matplotlib.pyplot as plt
import nengo_spa as spa
import torch

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


class SSP_Constructor:

    def __init__(self, n_rotations: int = 13, n_scale: int = 13, length_scale: int = 1):
        self.n_scale = n_scale
        self.n_rotations = n_rotations
        self.RES_X, self.RES_Y = 200, 200
        self.length_scale = length_scale
        self.counter = 1
        self.ssp_grid, self.ssp_space, self.vocab_combined = self.init_ssp_constructor()

    def generate_env_ssp(self, grid_objects_param):
        # init empty environment
        global_env_ssp = SSP(np.zeros(self.ssp_space.ssp_dim).reshape((1, -1)))

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

        # example extracted visualization
        # self._print_extracted_image(global_env_ssp, grid_objects_param)
        # return vector = "idk yet"

    def init_ssp_constructor(self):
        # Create SSP spaces for xy coordinates and labels
        xs = np.linspace(0, self.RES_X, self.RES_X)
        ys = np.linspace(0, self.RES_Y, self.RES_Y)
        x, y = np.meshgrid(xs, ys)
        xys = np.vstack((x.flatten(), y.flatten())).T

        ssp_space = HexagonalSSPSpace(domain_dim=2, n_rotates=self.n_rotations, n_scales=self.n_scale)
        ssp_space.update_lengthscale(self.length_scale)
        print(f"dim: {ssp_space.ssp_dim}")
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

        return ssp_grid, ssp_space, vocab_combined

    def _print_extracted_image(self, global_env_ssp_param, grid_objects):
        for obj in grid_objects:
            obj_type = get_value_from_string(obj.name['type'])
            if obj_type == 'agent':
                obj_type = get_value_from_string(obj.name['type'])
                obj_shape = get_value(obj.name['shape'], "shape")
                object_ssp = self.vocab_combined[f"{obj_type.upper()}_{obj_shape.upper()}"]

                inv_ssp = ~object_ssp

                # get similarity map of label with all locations by binding with inverse ssp
                out = global_env_ssp_param * inv_ssp
                sims = self.ssp_grid | out
                sims_map = sims.reshape((self.RES_X, self.RES_Y))
                sims_map = np.rot90(sims_map, k=1)

                pred_loc = np.array(np.unravel_index(np.argmax(sims_map), sims_map.shape))
                x_correct = obj.y
                y_correct = 200 - obj.x
                distance = abs(math.sqrt((pred_loc[1] - x_correct) ** 2 + (pred_loc[0] - y_correct) ** 2))

                RED = '\033[31m'  # Rot
                GREEN = '\033[32m'  # Grün
                YELLOW = '\033[33m'  # Gelb
                RESET = '\033[0m'  # Zurücksetzen auf Standardfarbe
                color = RED
                if distance < 5:
                    color = GREEN
                elif distance < 20:
                    color = YELLOW

                print(
                    f'{obj_type.upper()} predicted location: {pred_loc[1], pred_loc[0]}, correct location: {x_correct, y_correct}, {color} distance between: {distance} {RESET}')
                # img
                plt.imshow(sims_map, cmap='plasma', origin='lower', extent=(0, self.RES_X, 0, self.RES_Y),
                           interpolation='none')
                plt.colorbar()
                plt.xticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
                plt.yticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])

                size = 1
                plt.plot([pred_loc[1] - size, pred_loc[1] + size], [pred_loc[0] - size, pred_loc[0] + size],
                         color="red", linewidth=1.5)
                plt.plot([pred_loc[1] - size, pred_loc[1] + size], [pred_loc[0] + size, pred_loc[0] - size],
                         color="red", linewidth=1.5)
                plt.plot([x_correct - size, x_correct + size], [y_correct - size, y_correct + size],
                         color="blue", linewidth=1.5)
                plt.plot([x_correct - size, x_correct + size], [y_correct + size, y_correct - size],
                         color="blue", linewidth=1.5)

                # local_max = maximum_filter(sims_map, size=3)
                # maxima = (sims_map == local_max)
                # plt.scatter(np.where(maxima)[1], np.where(maxima)[0], color='white', s=20, label='Local Maxima')

                # plt.scatter(pred_loc[1], pred_loc[0], color='green', s=20, label='Local Maxima')

                # folder_path = f"output_images/DIM{self.ssp_space.ssp_dim}_ls5"
                # os.makedirs(folder_path, exist_ok=True)
                # image_path = os.path.join(folder_path, f"plot_{self.counter}.png")
                # self.counter = self.counter + 1
                # plt.title(
                #     f"n_scale: {self.n_scale}, n_rotations: {self.n_rotations}, length_scale: {self.length_scale}, dimensions: {self.ssp_space.ssp_dim}")
                # plt.savefig(image_path)
                # lt.show()
                # plt.clf()

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


def get_value_from_string(indexes_as_string):
    indexes = np.fromstring(indexes_as_string.replace('[', '').replace(']', ''), sep=' ')
    return get_value(indexes)


def get_value(indexes_param, attribute="type"):
    index_of_one = np.argmax(indexes_param - 1)
    if attribute == "type":
        return ENTITIES[index_of_one]
    elif attribute == "shape":
        return SHAPES[index_of_one]
