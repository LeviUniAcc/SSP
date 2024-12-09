import torch
import torch.utils.data
import os
import pickle as pkl
import json
import numpy as np
from tqdm import tqdm

import sys

from SSPConstructor import SSPConstructor
from grid_objects import parse_objects


# ========================== Helper functions ==========================

def index_data(json_list, path_list):
    print(f'processing files {len(json_list)}')
    data_tuples = []
    for j, v in tqdm(zip(json_list, path_list)):
        with open(j, 'r') as f:
            state = json.load(f)
        ep_lens = [len(x) for x in state]
        past_len = 0
        for e, l in enumerate(ep_lens):
            data_tuples.append([])
            # skip first 30 frames and last 83 frames
            for f in range(30, l - 83):
                # find action taken;
                f0x, f0y = state[e][f]['agent'][0]
                f1x, f1y = state[e][f + 1]['agent'][0]
                dx = (f1x - f0x) / 2.
                dy = (f1y - f0y) / 2.
                action = [dx, dy]
                # data_tuples[-1].append((v, past_len + f, action))
                data_tuples[-1].append((j, past_len + f, action))
                # data_tuples = [[json file, frame number, action] for each episode in each video]
            assert len(data_tuples[-1]) > 0
            past_len += l
    return data_tuples


def _get_frame_ssp(jsonfile, frame_idx, initialized_ssp: SSPConstructor):
    with open(jsonfile, 'rb') as f:
        frame_data = json.load(f)
    flat_list = [x for xs in frame_data for x in xs]
    # extract entities
    grid_objs = parse_objects(flat_list[frame_idx])
    frame_ssp = initialized_ssp.generate_env_ssp(grid_objs)
    return frame_ssp


# ========================== Dataset class ==========================

class SSPDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            path,
            initialized_ssp,
            types=None,
            mode="train",
            num_test=1,
            num_trials=9,
            action_range=10,
            process_data=0,
            max_len=30
    ):
        self.initialized_ssp = initialized_ssp
        self.path = path
        self.types = types
        self.mode = mode
        self.num_trials = num_trials
        self.num_test = num_test
        self.action_range = action_range
        self.max_len = max_len
        self.ep_combs = self.num_trials * (self.num_trials - 2)  # 9p2 - 9
        self.eps = [[x, y] for x in range(self.num_trials) for y in range(self.num_trials) if x != y]
        types_str = '_'.join(self.types)
        self.path_list = []
        self.json_list = []
        # get video paths and json file paths
        for t in types:
            print(f'reading files of type {t} in {mode}')
            paths = [os.path.join(self.path, t, x) for x in os.listdir(os.path.join(self.path, t)) if
                     x.endswith(f'.mp4')]
            jsons = [os.path.join(self.path, t, x) for x in os.listdir(os.path.join(self.path, t)) if
                     x.endswith(f'.json') and 'index' not in x]
            paths = sorted(paths)
            jsons = sorted(jsons)
            if mode == 'train':
                self.path_list += paths[:int(0.8 * len(jsons))]
                self.json_list += jsons[:int(0.8 * len(jsons))]
            elif mode == 'val':
                self.path_list += paths[int(0.8 * len(jsons)):]
                self.json_list += jsons[int(0.8 * len(jsons)):]
            else:
                self.path_list += paths
                self.json_list += jsons
        self.data_tuples = []
        if process_data:
            # index the videos in the dataset directory. This is done to speed up the retrieval of videos.
            # frame index, action tuples are stored
            self.data_tuples = index_data(self.json_list, self.path_list)
            # tuples of frame index and action (displacement of agent)
            index_dict = {'data_tuples': self.data_tuples}
            with open(os.path.join(self.path, f'jindex_bib_{mode}_{types_str}.json'), 'w') as fp:
                json.dump(index_dict, fp)
        else:
            # read pre-indexed data
            with open(os.path.join(self.path, f'jindex_bib_{mode}_{types_str}.json'), 'r') as fp:
                index_dict = json.load(fp)
            self.data_tuples = index_dict['data_tuples']
        self.tot_trials = len(self.path_list) * 9

    def get_trial_ssp(self, trials, step, initialized_ssp):
        # retrieve state embeddings and actions from cached file
        trial_len = []
        ssps = []
        # 8 trials
        for t in trials:
            tl = [(t, n) for n in range(0, len(self.data_tuples[t]), step)]
            if len(tl) > self.max_len:  # 30
                tl = tl[:self.max_len]
            trial_len.append(tl)
        for tl in trial_len:
            for t, n in tl:
                video = self.data_tuples[t][n][0]
                ssps.append((f'{t}-{n}', _get_frame_ssp(video, self.data_tuples[t][n][1], initialized_ssp)))
        return ssps

    def __getitem__(self, idx):
        ep_trials = [idx * self.num_trials + t for t in range(self.num_trials)]  # [idx, ..., idx+8]
        return self.get_trial_ssp(ep_trials, step=self.action_range, initialized_ssp=self.initialized_ssp)

    def __len__(self):
        return self.tot_trials // self.num_trials