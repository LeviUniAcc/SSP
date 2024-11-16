import os
import pickle as pkl
from dataset import TransitionDataset
from src.SSP import SSP


def generate_files(idx, initialized_ssp):
    print('Generating idx', idx)
    print(os.path.abspath(PATH + str(idx) + '.pkl'))
    if os.path.exists(PATH + str(idx) + '.pkl'):
        print('Index', idx, 'skipped.')
        return
    states, actions, lens, n_nodes = dataset.__getitem_ssp__(idx, initialized_ssp)
    with open(PATH + str(idx) + '.pkl', 'wb') as f:
        pkl.dump([states, actions, lens, n_nodes], f)

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
    initialized_ssp = SSP()
    generate_files(5, initialized_ssp)
