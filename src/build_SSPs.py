import os

from dataset import TransitionDataset
from src.SSP_Constructor import SSP_Constructor
from datetime import datetime

def generate_files(idx, initialized_ssp):
    print('Generating idx', idx)
    print(os.path.abspath(PATH + str(idx) + '.pkl'))
    if os.path.exists(PATH + str(idx) + '.pkl'):
        print('Index', idx, 'skipped.')
        return
    states, actions, lens, n_nodes = dataset.__getitem_ssp__(idx, initialized_ssp)
    return
    with open(PATH + str(idx) + '.pkl', 'wb') as f:
        pkl.dump([states, actions, lens, n_nodes], f)
        pass

    print(PATH + str(idx) + '.pkl saved.')


if __name__ == "__main__":
    PATH = '../data/SSP/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        print(PATH, 'directory created.')
    dataset = TransitionDataset(
        path='../data/',  # not original
        types=['demo_data'],
        mode="train",
        max_len=30,
        num_test=1,
        num_trials=9,
        action_range=15,
        process_data=1
    )
    before = datetime.now()
    initialized_ssp = SSP_Constructor(n_rotations=17, n_scale=17, length_scale=5)
    generate_files(5, initialized_ssp)
    after = datetime.now()
    print(f"duration: {after-before}")
