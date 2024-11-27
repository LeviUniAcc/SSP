import json
import os
import random

import numpy as np
import torch

from dataset import TransitionDataset
from src.SSPConstructor import SSPConstructor
from datetime import datetime


def append_to_json_file(file_name, new_data):
    if os.path.exists(file_name):
        with open(file_name, "r") as json_file:
            try:
                data = json.load(json_file)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(new_data)

    with open(file_name, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Daten wurden erfolgreich zu '{file_name}' hinzugefügt.")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    set_random_seed(42)
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
    combinations = [
        {"n_rotations": 17, "n_scale": 17, "length_scale": 5}
    ]
    for combination in combinations:
        print(combination)
        initialized_ssp = SSPConstructor(
            n_rotations=combination["n_rotations"],
            n_scale=combination["n_scale"],
            length_scale=combination["length_scale"]
        )
        generate_files(5, initialized_ssp)

        file = "output_images/results.json"
        append_to_json_file(file, initialized_ssp.results.to_dict())
