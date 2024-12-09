import json
import random
import numpy as np
import torch
import multiprocessing as mp
import argparse
import pickle as pkl
import os
from dataset import SSPDataset
from SSPConstructor import SSPConstructor

# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--cpus', type=int,
                    help='Number of processes')
parser.add_argument('--mode', type=str,
                    help='Train (train) or validation (val)')
args = parser.parse_args()

NUM_PROCESSES = args.cpus
MODE = args.mode


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_files(arguments):
    idx, dataset, lock = arguments
    print('Generating idx', idx)
    if MODE == 'demo':
        ssp_list_for_index = dataset.__getitem__(idx)
        file = f'data/external/bib_train/results_demo/results_idx_{idx}.pkl'
        file2 = f'data/external/bib_train/ssp_dataset_demo/environment_ssps_idx_{idx}.pkl'
        with open(file, 'wb') as f:
            pkl.dump(dataset.initialized_ssp.results, f)
        with open(file2, 'wb') as f:
            pkl.dump(ssp_list_for_index, f)
    elif MODE == 'train' or MODE == 'val':
        ssp_list_for_index = dataset.__getitem__(idx)
        file = f'data/external/bib_train/results/results_idx_{idx}.pkl'
        file2 = f'data/external/bib_train/ssp_dataset/environment_ssps_idx_{idx}.pkl'
        with lock:
            with open(file, 'wb') as f:
                pkl.dump(dataset.initialized_ssp.results, f)
            with open(file2, 'wb') as f:
                pkl.dump(ssp_list_for_index, f)
    else:
        raise ValueError(f'MODE ({MODE}) can be only demo, train, val or test.')


if __name__ == "__main__":
    with mp.Manager() as manager:
        lock = manager.Lock()
        if MODE == 'demo':
            print('DEMO MODE')
            initialized_ssp = SSPConstructor(10, 10, 5, 1)
            dataset = SSPDataset(
                path='data/external/bib_train/',
                initialized_ssp=initialized_ssp,
                types=['single_object'],
                mode="train",
                max_len=30,
                num_test=1,
                num_trials=9,
                action_range=10,
                process_data=1
            )
            generate_files([0, dataset, None])
        elif MODE == 'train':
            print('TRAIN MODE')
            initialized_ssp = SSPConstructor(10, 10, 5, 0)
            dataset = SSPDataset(
                path='data/external/bib_train/',
                initialized_ssp=initialized_ssp,
                types=['single_object'],
                mode="train",
                max_len=30,
                num_test=1,
                num_trials=9,
                action_range=10,
                process_data=0
            )
            pool = mp.Pool(processes=NUM_PROCESSES)
            print('Starting SSP generation with', NUM_PROCESSES, 'processes...')
            pool.map(generate_files, [(i, dataset, lock) for i in range(dataset.__len__())])
            pool.close()
        elif MODE == 'val':
            print('VALIDATION MODE')
            types = ['multi_agent', 'instrumental_action', 'preference', 'single_object']
            for t in range(len(types)):
                PATH = '/datasets/external/bib_train/graphs/all_tasks/val_dgl_hetero_nobound_4feats/' + types[t] + '/'
                if not os.path.exists(PATH):
                    os.makedirs(PATH)
                    print(PATH, 'directory created.')
                dataset = SSPDataset(
                    path='/datasets/external/bib_train/',
                    types=[types[t]],
                    mode="val",
                    max_len=30,
                    num_test=1,
                    num_trials=9,
                    action_range=10,
                    process_data=0
                )
                pool = mp.Pool(processes=NUM_PROCESSES)
                print('Starting', types[t], 'graph generation with', NUM_PROCESSES, 'processes...')
                pool.map(generate_files, [i for i in range(dataset.__len__())])
                pool.close()
        else:
            raise ValueError
