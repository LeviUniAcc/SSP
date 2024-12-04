import json
import random
import numpy as np
import torch
import multiprocessing as mp
import argparse
import pickle as pkl
import os
from dataset import TransitionDataset, TestTransitionDatasetSequence
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


def generate_files(args):
    idx, initialized_ssp, PATH, dataset, lock = args
    print('Generating idx', idx)
    if os.path.exists(PATH + str(idx) + '.pkl'):
        print('Index', idx, 'skipped.')
        return
    if MODE == 'demo':
        ssp_list_for_index = dataset.__getitem_ssp__(idx, initialized_ssp)
        file = f'data/external/bib_train/results_demo/results_idx_{idx}.pkl'
        file2 = f'data/external/bib_train/ssp_dataset_demo/environment_ssps_idx_{idx}.pkl'
        with open(file, 'wb') as f:
            pkl.dump(initialized_ssp.results, f)
        with open(file2, 'wb') as f:
            pkl.dump(ssp_list_for_index, f)
    elif MODE == 'train' or MODE == 'val':
        ssp_list_for_index = dataset.__getitem_ssp__(idx, initialized_ssp)
        # with open(PATH + str(idx) + '.pkl', 'wb') as f:
        # pkl.dump([states, actions, lens, n_nodes], f)
        file = f'data/external/bib_train/results/results_idx_{idx}.json'
        file2 = f'data/external/bib_train/ssp_dataset/environment_ssps_idx_{idx}.json'
        with lock:
            with open(file, 'wb') as f:
                pkl.dump(initialized_ssp.results, f)
            with open(file2, 'wb') as f:
                pkl.dump(ssp_list_for_index, f)
    elif MODE == 'test':
        dem_expected_states, dem_expected_actions, dem_expected_lens, dem_expected_nodes, \
            dem_unexpected_states, dem_unexpected_actions, dem_unexpected_lens, dem_unexpected_nodes, \
            query_expected_frames, target_expected_actions, \
            query_unexpected_frames, target_unexpected_actions = dataset.__getitem__(idx)
        with open(PATH + str(idx) + '.pkl', 'wb') as f:
            pkl.dump([
                dem_expected_states, dem_expected_actions, dem_expected_lens, dem_expected_nodes, \
                dem_unexpected_states, dem_unexpected_actions, dem_unexpected_lens, dem_unexpected_nodes, \
                query_expected_frames, target_expected_actions, \
                query_unexpected_frames, target_unexpected_actions], f
            )
    else:
        raise ValueError(f'MODE ({MODE}) can be only demo, train, val or test.')
    print(PATH + str(idx) + '.pkl saved.')


if __name__ == "__main__":
    with mp.Manager() as manager:
        lock = manager.Lock()
        if MODE == 'demo':
            print('DEMO MODE')
            PATH = 'data/external/bib_train/graphs/all_tasks/demo_dgl_hetero_nobound_4feats/'
            dataset = TransitionDataset(
                path='data/external/bib_train/',
                types=['single_object'],
                mode="train",
                max_len=30,
                num_test=1,
                num_trials=9,
                action_range=10,
                process_data=1
            )
            initialized_ssp = SSPConstructor(10, 10, 5, 1)
            generate_files([0, initialized_ssp, PATH, dataset, None])
        elif MODE == 'train':
            print('TRAIN MODE')
            PATH = 'data/external/bib_train/graphs/all_tasks/train_dgl_hetero_nobound_4feats/'
            if not os.path.exists(PATH):
                os.makedirs(PATH)
                print(PATH, 'directory created.')
            dataset = TransitionDataset(
                path='data/external/bib_train/',
                types=['single_object'],
                mode="train",
                max_len=30,
                num_test=1,
                num_trials=9,
                action_range=10,
                process_data=1
            )
            pool = mp.Pool(processes=NUM_PROCESSES)
            print('Starting graph generation with', NUM_PROCESSES, 'processes...')
            initialized_ssp = SSPConstructor(10, 10, 5, 1)
            pool.map(generate_files, [(i, initialized_ssp, PATH, dataset, lock) for i in range(dataset.__len__())])
            pool.close()
        elif MODE == 'val':
            print('VALIDATION MODE')
            types = ['multi_agent', 'instrumental_action', 'preference', 'single_object']
            for t in range(len(types)):
                PATH = '/datasets/external/bib_train/graphs/all_tasks/val_dgl_hetero_nobound_4feats/' + types[t] + '/'
                if not os.path.exists(PATH):
                    os.makedirs(PATH)
                    print(PATH, 'directory created.')
                dataset = TransitionDataset(
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
        elif MODE == 'test':
            print('TEST MODE')
            types = [
                'preference', 'multi_agent', 'inaccessible_goal',
                'efficiency_irrational', 'efficiency_time', 'efficiency_path',
                'instrumental_no_barrier', 'instrumental_blocking_barrier', 'instrumental_inconsequential_barrier'
            ]
            for t in range(len(types)):
                PATH = '/datasets/external/bib_evaluation_1_1/graphs/all_tasks_dgl_hetero_nobound_4feats/' + types[
                    t] + '/'
                if not os.path.exists(PATH):
                    os.makedirs(PATH)
                    print(PATH, 'directory created.')
                dataset = TestTransitionDatasetSequence(
                    path='/datasets/external/bib_evaluation_1_1/',
                    task_type=types[t],
                    mode="test",
                    num_test=1,
                    num_trials=9,
                    action_range=10,
                    process_data=0,
                    max_len=30
                )
                pool = mp.Pool(processes=NUM_PROCESSES)
                print('Starting', types[t], 'graph generation with', NUM_PROCESSES, 'processes...')
                pool.map(generate_files, [i for i in range(dataset.__len__())])
                pool.close()
        else:
            raise ValueError
