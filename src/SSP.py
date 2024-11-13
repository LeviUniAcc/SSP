import numpy as np
import torch
import matplotlib.pyplot as plt
import nengo_spa as spa
import grid_objects

from matplotlib.patches import Rectangle, Polygon, Circle
from sspspace import SPSpace, SSPSpace, HexagonalSSPSpace

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


# CLASS_LST = ['GREEN', 'PURPLE', 'YELLOW', 'CIRCLE', 'SQUARE', 'TRIANGLE']

class SSP:

    def __init__(self,
                 grid_objects_param,
                 ):
        self.grid_objects = grid_objects_param
        SSP_DIM = 1015
        RES_X, RES_Y = 200, 200
        RNG = np.random.RandomState(42)

        vocab = spa.Vocabulary(dimensions=SSP_DIM, pointer_gen=RNG)

        # Add semantic pointers for each shape to the vocabulary
        for i, shape_name in enumerate(SHAPES):
            vector = vocab.algebra.create_vector(SSP_DIM, properties={"positive", "unitary"})
            vocab.add(f"{shape_name.upper().replace(' ', '')}", vector)

        # Add semantic pointers for each entity to the
        for i, entity_name in enumerate(ENTITIES):
            vector = vocab.algebra.create_vector(SSP_DIM, properties={"positive", "unitary"})
            vocab.add(f"{entity_name.upper().replace(' ', '')}", vector)

        # Create SSP spaces for xy coordinates and labels
        ssp_space = HexagonalSSPSpace(domain_dim=2, ssp_dim=SSP_DIM, length_scale=5,
                                      domain_bounds=np.array([[-RES_X, RES_X + 1], [-RES_Y, RES_Y + 1]]))

        # Generate xy coordinates
        x_coords, y_coords = torch.meshgrid(torch.arange(-RES_X, RES_X + 1), torch.arange(-RES_Y, RES_Y + 1),
                                            indexing='ij')
        coords = np.stack((x_coords.flatten(), y_coords.flatten()), axis=-1)
        ssp_grid = ssp_space.encode(coords)
        ssp_grid = ssp_grid.reshape((RES_X * 2 + 1, RES_Y * 2 + 1, -1))
        print(f'Generated SSP grid: {ssp_grid.shape}')

        # example visualisation--------------------------------------- remove
        global_grid = np.zeros((RES_X * 2 + 1, RES_Y * 2 + 1))

        fig, ax = plt.subplots()

        plt.imshow(global_grid, extent=[-10, 10, -10, 10], cmap='gray')
        plt.xticks([-10, 0, 10])
        plt.yticks([-10, 0, 10])
        plt.hlines(0, -10, 10, color=(.8, .8, .8, .5))
        plt.vlines(0, -10, 10, color=(.8, .8, .8, .5))

        grid_color = (.3, .3, .3, .5)

        for i in range(10):
            plt.hlines(i + 0.5, -10, 10, color=grid_color, linewidth=1.)
            plt.hlines(-i + 0.5, -10, 10, color=grid_color, linewidth=1.)
            plt.vlines(i + 0.5, -10, 10, color=grid_color, linewidth=1.)
            plt.vlines(-i + 0.5, -10, 10, color=grid_color, linewidth=1.)

        # add walls
        for object_element in self.grid_objects:
            if get_value_as_string(object_element.name['type']) == "walls":
                ax.add_patch(
                    Rectangle((object_element.x, object_element.y), width=20, height=20, color='gray', zorder=3))

        # add agent
        for object_element in self.grid_objects:
            if get_value_as_string(object_element.name['type']) == "agent":
                ax.add_patch(Polygon([[object_element.x, object_element.y], [object_element.x, object_element.y+20], [object_element.x+20, object_element.y+10]], closed=True, color='r', zorder=5))

        # add objects
        for object_element in self.grid_objects:
            if get_value_as_string(object_element.name['type']) == "vase":
                ax.add_patch(Circle((object_element.x, object_element.y), radius=.35, color='g', zorder=5))
        for object_element in self.grid_objects:
            if get_value_as_string(object_element.name['type']) == "home_entity":
                ax.add_patch(Rectangle((object_element.x, object_element.y), width=.6, height=.6, color='m', zorder=5))

        plt.title('Global Environment')
        plt.show()

        # combine all object ssps

        self.vector = "idk yet"


def get_value_as_string(indexes):
    value_list = np.fromstring(indexes.replace('[', '').replace(']', ''), sep=' ')
    index_of_one = np.argmax(value_list)
    entity_name = ENTITIES[index_of_one]
    return entity_name
