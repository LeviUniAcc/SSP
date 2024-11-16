import math

import numpy as np
import torch
import matplotlib.pyplot as plt
import nengo_spa as spa

from matplotlib.patches import Rectangle, Polygon, Circle
from webcolors import rgb_to_name

from sspspace import HexagonalSSPSpace

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


class SSP:

    def __init__(self,
                 grid_objects_param,
                 ):
        self.grid_objects = grid_objects_param
        self.SSP_DIM = 1015
        self.RES_X, self.RES_Y = 200, 200

        rng = np.random.RandomState(42)
        vocab = spa.Vocabulary(dimensions=self.SSP_DIM, pointer_gen=rng)

        # Add semantic pointers for each shape to the vocabulary
        for i, shape_name in enumerate(SHAPES):
            vector = vocab.algebra.create_vector(self.SSP_DIM, properties={"positive", "unitary"})
            vocab.add(f"{shape_name.upper().replace(' ', '')}", vector)

        # Add semantic pointers for each entity to the vocabulary
        for i, entity_name in enumerate(ENTITIES):
            vector = vocab.algebra.create_vector(self.SSP_DIM, properties={"positive", "unitary"})
            vocab.add(f"{entity_name.upper().replace(' ', '')}", vector)

        # Add semantic pointers for each color to the vocabulary
        for obj in self.grid_objects:
            vector = vocab.algebra.create_vector(self.SSP_DIM, properties={"positive", "unitary"})
            color_name = rgb_to_name(obj.name['color']).upper().replace(' ', '')
            if vocab.get(color_name) is None:
                vocab.add(f"{color_name.upper()}", vector)

        # Create SSP spaces for xy coordinates and labels
        ssp_space = HexagonalSSPSpace(domain_dim=2, ssp_dim=self.SSP_DIM, length_scale=5,
                                      domain_bounds=np.array([[0, self.RES_X], [0, self.RES_Y]]))

        # Generate xy coordinates
        x_coords, y_coords = torch.meshgrid(torch.arange(0, self.RES_X), torch.arange(0, self.RES_Y),
                                            indexing='ij')
        coords = np.stack((x_coords.flatten(), y_coords.flatten()), axis=-1)
        ssp_grid = ssp_space.encode(coords)
        ssp_grid = ssp_grid.reshape((self.RES_X, self.RES_Y, -1))

        print(f'Generated SSP grid: {ssp_grid.shape}')

        # combine all object ssps
        ## init empty environment
        global_env_ssp = np.zeros(self.SSP_DIM)

        ## bind and add alls object ssps
        for obj in self.grid_objects:
            obj_type = get_value_from_string(obj.name['type'])
            obj_shape = get_value(obj.name['shape'], "shape")
            obj_color = rgb_to_name(obj.name['color']).upper().replace(' ', '')
            obj_x = int(float(obj.name['x']))
            obj_y = int(float(obj.name['y']))

            type_ssp = vocab[obj_type.upper()].v
            shape_ssp = vocab[obj_shape.upper()].v
            color_ssp = vocab[obj_color].v
            position_ssp = ssp_grid[obj_x, obj_y]

            object_ssp = ssp_space.bind(ssp_space.bind(ssp_space.bind(type_ssp, shape_ssp), color_ssp), position_ssp)
            global_env_ssp += object_ssp.squeeze()

        # example visualization of image
        self._print_image()
        # example extracted visualization
        self._print_extracted_image(global_env_ssp, ssp_grid, ssp_space, vocab)

        self.vector = "idk yet"

    def _print_extracted_image(self, ssp_dim_param, global_env_ssp_param, ssp_grid_param,
                               ssp_space_param, vocab_param):
        for obj in self.grid_objects:
            obj_type = get_value_from_string(obj.name['type'])
            if obj_type == 'agent':
                obj_type = get_value_from_string(obj.name['type'])
                obj_shape = get_value(obj.name['shape'], "shape")
                obj_color = rgb_to_name(obj.name['color']).upper().replace(' ', '')

                type_ssp = vocab_param[obj_type.upper()].v
                shape_ssp = vocab_param[obj_shape.upper()].v
                color_ssp = vocab_param[obj_color].v

                object_without_position_ssp = ssp_space_param.bind(ssp_space_param.bind(type_ssp, shape_ssp), color_ssp)

                inv_ssp = ssp_space_param.invert(object_without_position_ssp)

                # get similarity map of label with all locations by binding with inverse ssp
                out = ssp_space_param.bind(global_env_ssp_param, inv_ssp)
                sims = out @ ssp_grid_param.reshape((-1, ssp_dim_param)).T

                # decode location = point with maximum similarity to label
                sims_map = sims.reshape((self.RES_X, self.RES_Y))

                pred_loc = np.array(np.unravel_index(np.argmax(sims_map), sims_map.shape))
                x_predicted = pred_loc[1] - 10
                y_predicted = 200 - (pred_loc[0] - 10 + 20)
                x_correct = obj.y - 10
                y_correct = 200 - (obj.x - 10 + 20)
                distance = abs(math.sqrt((x_predicted - x_correct) ** 2 + (y_predicted - y_correct) ** 2))

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
                    f'{obj_type.upper()} predicted location: {x_predicted, y_predicted}, correct location: {x_correct, y_correct}, {color} distance between: {distance} {RESET}')

                plt.imshow(sims_map, extent=(0, self.RES_X, 0, self.RES_Y))
                plt.xticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
                plt.yticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])

                size = 5
                plt.plot([x_predicted - size, x_predicted + size], [y_predicted - size, y_predicted + size],
                         color="red", linewidth=1.5)
                plt.plot([x_predicted - size, x_predicted + size], [y_predicted + size, y_predicted - size],
                         color="red", linewidth=1.5)
                plt.plot([x_correct - size, x_correct + size], [y_correct - size, y_correct + size],
                         color="blue", linewidth=1.5)
                plt.plot([x_correct - size, x_correct + size], [y_correct + size, y_correct - size],
                         color="blue", linewidth=1.5)

                plt.show()

    def _print_image(self):
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
        for object_element in self.grid_objects:
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
