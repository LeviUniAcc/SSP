import numpy as np
from sklearn.preprocessing import OneHotEncoder
import itertools

SHAPES = {
    # walls
    'square': 0,
    # objects
    'heart': 1, 'clove_tree': 2, 'moon': 3, 'wine': 4, 'double_dia': 5,
    'flag': 6, 'capsule': 7, 'vase': 8, 'curved_triangle': 9, 'spoon': 10,
    'medal': 11,  # inaccessible goal
    # home
    'home': 12,
    # agent
    'pentagon': 13, 'clove': 14, 'kite': 15,
    # key 
    'triangle0': 16, 'triangle180': 16, 'triangle90': 16, 'triangle270': 16,
    # lock 
    'triangle_slot0': 17, 'triangle_slot180': 17, 'triangle_slot90': 17, 'triangle_slot270': 17
}

ENTITIES = {
    'agent': 0, 'walls': 1, 'fuse_walls': 2, 'key': 3,
    'blocking': 4, 'home': 5, 'objects': 6, 'pin': 7, 'lock': 8
}


# =============================== GridPbject class ===============================

class GridObject():
    "object is specified by its location"

    def __init__(self, x, y, object_type, attributes=[]):
        self.x = x
        self.y = y
        self.type = object_type
        self.attributes = attributes

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def name(self):
        return {'type': str(self.type),
                'x': str(self.x),
                'y': str(self.y),
                'color': self.attributes[0],
                'shape': self.attributes[1]}


# =============================== Helper functions ===============================

def type2index(key):
    for name, idx in ENTITIES.items():
        if name == key:
            return idx


def find_shape(shape_string, print_shape=False):
    try:
        shape = shape_string.split('/')[-1].split('.')[0]
    except:
        shape = shape_string.split('.')[0]
    if print_shape: print(shape)
    for name, idx in SHAPES.items():
        if name == shape:
            return idx


def parse_objects(frame):
    """
    x and y are computed differently from walls and objects
    for walls x, y = obj[0][0] + obj[1][0]/2, obj[0][1] + obj[1][1]/2
    for objects x, y = obj[0][0] + obj[1], obj[0][1] + obj[1]
    :param obj:
    :return: GridObject
    """
    shape_onehot_encoder = OneHotEncoder(sparse_output=False)
    shape_onehot_encoder.fit([[i] for i in range(len(SHAPES) - 6)])
    type_onehot_encoder = OneHotEncoder(sparse_output=False)
    type_onehot_encoder.fit([[i] for i in range(len(ENTITIES))])
    # remove duplicate walls
    frame['walls'].sort()
    frame['walls'] = list(k for k, _ in itertools.groupby(frame['walls']))
    # remove boundary walls
    frame['walls'] = [w for w in frame['walls'] if
                      (w[0][0] != 0 and w[0][0] != 180 and w[0][1] != 0 and w[0][1] != 180)]
    # remove duplicate fuse_walls
    frame['fuse_walls'].sort()
    frame['fuse_walls'] = list(k for k, _ in itertools.groupby(frame['fuse_walls']))
    grid_objs = []
    assert 'agent' in frame.keys()
    for key in frame.keys():
        print(key)
        if key == 'size':
            continue
        obj = frame[key]
        if obj == []:
            # print(key, 'skipped')
            continue
        obj_type = type2index(key)
        obj_type = type_onehot_encoder.transform([[obj_type]])
        if key == 'walls':
            for wall in obj:
                x, y = wall[0][0] + wall[1][0] / 2, wall[0][1] + wall[1][1] / 2
                # x, y = (wall[0][0] + wall[1][0]/2)/200, (wall[0][1] + wall[1][1]/2)/200 if u use this you need to change relations.py!!!
                color = [0, 0, 0] if key == 'walls' else [80, 146, 56]
                # color = [c / 255 for c in color]
                shape = 0
                assert shape in SHAPES.values(), 'Shape not found'
                shape = shape_onehot_encoder.transform([[shape]])[0]
                grid_obj = GridObject(x=x, y=y, object_type=obj_type, attributes=[color, shape])
                grid_objs.append(grid_obj)
        elif key == 'fuse_walls':
            # resample green barriers
            obj = [obj[i] for i in range(len(obj)) if (obj[i][0][0] % 20 == 0 and obj[i][0][1] % 20 == 0)]
            for wall in obj:
                x, y = wall[0][0] + wall[1][0] / 2, wall[0][1] + wall[1][1] / 2
                # x, y = (wall[0][0] + wall[1][0]/2)/200, (wall[0][1] + wall[1][1]/2)/200 if u use this you need to change relations.py!!!
                color = [80, 146, 56]
                # color = [c / 255 for c in color]
                shape = 0
                assert shape in SHAPES.values(), 'Shape not found'
                shape = shape_onehot_encoder.transform([[shape]])[0]
                grid_obj = GridObject(x=x, y=y, object_type=obj_type, attributes=[color, shape])
                grid_objs.append(grid_obj)
        elif key == 'objects':
            for ob in obj:
                x, y = ob[0][0] + ob[1], ob[0][1] + ob[1]
                color = ob[-1]
                # color = [c / 255 for c in color]
                shape = find_shape(ob[2], print_shape=False)
                assert shape in SHAPES.values(), 'Shape not found'
                shape = shape_onehot_encoder.transform([[shape]])[0]
                grid_obj = GridObject(x=x, y=y, object_type=obj_type, attributes=[color, shape])
                grid_objs.append(grid_obj)
        elif key == 'key':
            obj = obj[0]
            x, y = obj[0][0] + obj[1], obj[0][1] + obj[1]
            color = obj[-1]
            # color = [c / 255 for c in color]
            shape = find_shape(obj[2], print_shape=False)
            assert shape in SHAPES.values(), 'Shape not found'
            shape = shape_onehot_encoder.transform([[shape]])[0]
            grid_obj = GridObject(x=x, y=y, object_type=obj_type, attributes=[color, shape])
            grid_objs.append(grid_obj)
        elif key == 'lock':
            obj = obj[0]
            x, y = obj[0][0] + obj[1], obj[0][1] + obj[1]
            color = obj[-1]
            # color = [c / 255 for c in color]
            shape = find_shape(obj[2], print_shape=False)
            assert shape in SHAPES.values(), 'Shape not found'
            shape = shape_onehot_encoder.transform([[shape]])[0]
            grid_obj = GridObject(x=x, y=y, object_type=obj_type, attributes=[color, shape])
            grid_objs.append(grid_obj)
        else:
            try:
                x, y = obj[0][0] + obj[1], obj[0][1] + obj[1]
                color = obj[-1]
                # color = [c / 255 for c in color]
                shape = find_shape(obj[2], print_shape=False)
                assert shape in SHAPES.values(), 'Shape not found'
                shape = shape_onehot_encoder.transform([[shape]])[0]
            except:
                # [[[x, y], extension, shape, color]] in some cases in instrumental_no_barrier (bib_evaluation_1_1)
                x, y = obj[0][0][0] + obj[0][1], obj[0][0][1] + obj[0][1]
                color = obj[0][-1]
                # color = [c / 255 for c in color]
                assert len(color) == 3
                shape = find_shape(obj[0][2], print_shape=False)
                assert shape in SHAPES.values(), 'Shape not found'
                shape = shape_onehot_encoder.transform([[shape]])[0]
            grid_obj = GridObject(x=x, y=y, object_type=obj_type, attributes=[color, shape])
            grid_objs.append(grid_obj)
    return grid_objs
