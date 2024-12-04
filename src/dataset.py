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

class TransitionDataset(torch.utils.data.Dataset):
    """
    Training dataset class for the behavior cloning mlp model.
    Args:
        path: path to the dataset
        types: list of video types to include
        mode: train, val
        num_test: number of test state-action pairs
        num_trials: number of trials in an episode
        action_range: number of frames to skip; actions are combined over these number of frames (displcement) of the agent
        process_data: whether to the videos or not (skip if already processed)
        max_len: max number of context state-action pairs
    __getitem__:
        returns: (states, actions, lens, n_nodes)
        dem_frames: batched DGLGraph.heterograph
        dem_actions: (max_len, 2)
        query_frames: DGLGraph.heterograph
        target_actions: (num_test, 2)
    """

    def __init__(
            self,
            path,
            types=None,
            mode="train",
            num_test=1,
            num_trials=9,
            action_range=10,
            process_data=0,
            max_len=30
    ):
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

    def _add_node_features(self, objs, graph):
        for obj_idx, obj in enumerate(objs):
            graph.nodes[obj_idx].data['type'] = torch.tensor(obj.type)
            graph.nodes[obj_idx].data['pos'] = torch.tensor([[obj.x, obj.y]], dtype=torch.float32)
            assert len(obj.attributes) == 2 and None not in obj.attributes[0] and None not in obj.attributes[1]
            graph.nodes[obj_idx].data['color'] = torch.tensor([obj.attributes[0]])
            graph.nodes[obj_idx].data['shape'] = torch.tensor([obj.attributes[1]])
        return graph

    def _get_spatial_rel(self, objs):
        spatial_tensors = [np.zeros([len(objs), len(objs)]) for _ in range(len(self.rel_deter_func))]
        for obj_idx1, obj1 in enumerate(objs):
            for obj_idx2, obj2 in enumerate(objs):
                direction_vec = np.array((0, -1))  # Up
                for rel_idx, func in enumerate(self.rel_deter_func):
                    if func(obj1, obj2, direction_vec):
                        spatial_tensors[rel_idx][obj_idx1, obj_idx2] = 1.0
        return spatial_tensors

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

    def __getitem_ssp__(self, idx, initialized_ssp):
        ep_trials = [idx * self.num_trials + t for t in range(self.num_trials)]  # [idx, ..., idx+8]
        return self.get_trial_ssp(ep_trials, step=self.action_range, initialized_ssp=initialized_ssp)

    def __len__(self):
        return self.tot_trials // self.num_trials


class TestTransitionDatasetSequence(torch.utils.data.Dataset):
    """
    Test dataset class for the behavior cloning rnn model. This dataset is used to test the model on the eval data.
    This class is used to compare plausible and implausible episodes.
    Args:
        path: path to the dataset
        types: list of video types to include
        size: size of the frames to be returned
        mode: test
        num_context: number of context state-action pairs
        num_test: number of test state-action pairs
        num_trials: number of trials in an episode
        action_range: number of frames to skip; actions are combined over these number of frames (displcement) of the agent
        process_data: whether to the videos or not (skip if already processed)
    __getitem__:
        returns:  (expected_dem_frames, expected_dem_actions, expected_dem_lens expected_query_frames, expected_target_actions,
        unexpected_dem_frames, unexpected_dem_actions, unexpected_dem_lens, unexpected_query_frames, unexpected_target_actions)
        dem_frames: (num_context, max_len, 3, size, size)
        dem_actions: (num_context, max_len, 2)
        dem_lens: (num_context)
        query_frames: (num_test, 3, size, size)
        target_actions: (num_test, 2)
    """

    def __init__(
            self,
            path,
            task_type=None,
            mode="test",
            num_test=1,
            num_trials=9,
            action_range=10,
            process_data=0,
            max_len=30
    ):
        self.path = path
        self.task_type = task_type
        self.mode = mode
        self.num_trials = num_trials
        self.num_test = num_test
        self.action_range = action_range
        self.max_len = max_len
        self.ep_combs = self.num_trials * (self.num_trials - 2)  # 9p2 - 9
        self.eps = [[x, y] for x in range(self.num_trials) for y in range(self.num_trials) if x != y]
        self.path_list_exp = []
        self.json_list_exp = []
        self.path_list_un = []
        self.json_list_un = []

        print(f'reading files of type {task_type} in {mode}')

        paths_expected = sorted(
            [os.path.join(self.path, task_type, x) for x in os.listdir(os.path.join(self.path, task_type)) if
             x.endswith('e.mp4')])
        jsons_expected = sorted(
            [os.path.join(self.path, task_type, x) for x in os.listdir(os.path.join(self.path, task_type)) if
             x.endswith('e.json') and 'index' not in x])
        paths_unexpected = sorted(
            [os.path.join(self.path, task_type, x) for x in os.listdir(os.path.join(self.path, task_type)) if
             x.endswith('u.mp4')])
        jsons_unexpected = sorted(
            [os.path.join(self.path, task_type, x) for x in os.listdir(os.path.join(self.path, task_type)) if
             x.endswith('u.json') and 'index' not in x])
        self.path_list_exp += paths_expected
        self.json_list_exp += jsons_expected
        self.path_list_un += paths_unexpected
        self.json_list_un += jsons_unexpected
        self.data_expected = []
        self.data_unexpected = []

        if process_data:
            # index data. This is done to speed up video retrieval.
            # frame index, action tuples are stored
            self.data_expected = index_data(self.json_list_exp, self.path_list_exp)
            index_dict = {'data_tuples': self.data_expected}
            with open(os.path.join(self.path, f'jindex_bib_test_{task_type}e.json'), 'w') as fp:
                json.dump(index_dict, fp)

            self.data_unexpected = index_data(self.json_list_un, self.path_list_un)
            index_dict = {'data_tuples': self.data_unexpected}
            with open(os.path.join(self.path, f'jindex_bib_test_{task_type}u.json'), 'w') as fp:
                json.dump(index_dict, fp)
        else:
            with open(os.path.join(self.path, f'jindex_bib_{mode}_{task_type}e.json'), 'r') as fp:
                index_dict = json.load(fp)
            self.data_expected = index_dict['data_tuples']
            with open(os.path.join(self.path, f'jindex_bib_{mode}_{task_type}u.json'), 'r') as fp:
                index_dict = json.load(fp)
            self.data_unexpected = index_dict['data_tuples']

        self.rel_deter_func = [
            is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
            is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj,
            is_left, is_right, is_front, is_back, is_aligned, is_close
        ]

        print('Done.')

    def _get_frame_graph(self, jsonfile, frame_idx):
        # load json
        with open(jsonfile, 'rb') as f:
            frame_data = json.load(f)
        flat_list = [x for xs in frame_data for x in xs]
        # extract entities
        grid_objs = parse_objects(flat_list[frame_idx])
        # --- build the graph
        adj = self._get_spatial_rel(grid_objs)
        # define edges
        is_top_adj_src, is_top_adj_dst = np.nonzero(adj[0])
        is_left_adj_src, is_left_adj_dst = np.nonzero(adj[1])
        is_top_right_adj_src, is_top_right_adj_dst = np.nonzero(adj[2])
        is_top_left_adj_src, is_top_left_adj_dst = np.nonzero(adj[3])
        is_down_adj_src, is_down_adj_dst = np.nonzero(adj[4])
        is_right_adj_src, is_right_adj_dst = np.nonzero(adj[5])
        is_down_left_adj_src, is_down_left_adj_dst = np.nonzero(adj[6])
        is_down_right_adj_src, is_down_right_adj_dst = np.nonzero(adj[7])
        is_left_src, is_left_dst = np.nonzero(adj[8])
        is_right_src, is_right_dst = np.nonzero(adj[9])
        is_front_src, is_front_dst = np.nonzero(adj[10])
        is_back_src, is_back_dst = np.nonzero(adj[11])
        is_aligned_src, is_aligned_dst = np.nonzero(adj[12])
        is_close_src, is_close_dst = np.nonzero(adj[13])
        g = dgl.heterograph({
            ('obj', 'is_top_adj', 'obj'): (torch.tensor(is_top_adj_src), torch.tensor(is_top_adj_dst)),
            ('obj', 'is_left_adj', 'obj'): (torch.tensor(is_left_adj_src), torch.tensor(is_left_adj_dst)),
            ('obj', 'is_top_right_adj', 'obj'): (
                torch.tensor(is_top_right_adj_src), torch.tensor(is_top_right_adj_dst)),
            ('obj', 'is_top_left_adj', 'obj'): (torch.tensor(is_top_left_adj_src), torch.tensor(is_top_left_adj_dst)),
            ('obj', 'is_down_adj', 'obj'): (torch.tensor(is_down_adj_src), torch.tensor(is_down_adj_dst)),
            ('obj', 'is_right_adj', 'obj'): (torch.tensor(is_right_adj_src), torch.tensor(is_right_adj_dst)),
            ('obj', 'is_down_left_adj', 'obj'): (
                torch.tensor(is_down_left_adj_src), torch.tensor(is_down_left_adj_dst)),
            ('obj', 'is_down_right_adj', 'obj'): (
                torch.tensor(is_down_right_adj_src), torch.tensor(is_down_right_adj_dst)),
            ('obj', 'is_left', 'obj'): (torch.tensor(is_left_src), torch.tensor(is_left_dst)),
            ('obj', 'is_right', 'obj'): (torch.tensor(is_right_src), torch.tensor(is_right_dst)),
            ('obj', 'is_front', 'obj'): (torch.tensor(is_front_src), torch.tensor(is_front_dst)),
            ('obj', 'is_back', 'obj'): (torch.tensor(is_back_src), torch.tensor(is_back_dst)),
            ('obj', 'is_aligned', 'obj'): (torch.tensor(is_aligned_src), torch.tensor(is_aligned_dst)),
            ('obj', 'is_close', 'obj'): (torch.tensor(is_close_src), torch.tensor(is_close_dst))
        }, num_nodes_dict={'obj': len(grid_objs)})
        g = self._add_node_features(grid_objs, g)
        return g

    def _add_node_features(self, objs, graph):
        for obj_idx, obj in enumerate(objs):
            graph.nodes[obj_idx].data['type'] = torch.tensor(obj.type)
            graph.nodes[obj_idx].data['pos'] = torch.tensor([[obj.x, obj.y]], dtype=torch.float32)
            assert len(obj.attributes) == 2 and None not in obj.attributes[0] and None not in obj.attributes[1]
            graph.nodes[obj_idx].data['color'] = torch.tensor([obj.attributes[0]])
            graph.nodes[obj_idx].data['shape'] = torch.tensor([obj.attributes[1]])
        return graph

    def _get_spatial_rel(self, objs):
        spatial_tensors = [np.zeros([len(objs), len(objs)]) for _ in range(len(self.rel_deter_func))]
        for obj_idx1, obj1 in enumerate(objs):
            for obj_idx2, obj2 in enumerate(objs):
                direction_vec = np.array((0, -1))  # Up why??????????????
                for rel_idx, func in enumerate(self.rel_deter_func):
                    if func(obj1, obj2, direction_vec):
                        spatial_tensors[rel_idx][obj_idx1, obj_idx2] = 1.0
        return spatial_tensors

    def get_trial(self, trials, data, step=10):
        # retrieve state embeddings and actions from cached file
        states = []
        actions = []
        trial_len = []
        lens = []
        n_nodes = []
        for t in trials:
            tl = [(t, n) for n in range(0, len(data[t]), step)]
            if len(tl) > self.max_len:
                tl = tl[:self.max_len]
            trial_len.append(tl)
        for tl in trial_len:
            states.append([])
            actions.append([])
            lens.append(len(tl))
            for t, n in tl:
                video = data[t][n][0]
                states[-1].append(self._get_frame_graph(video, data[t][n][1]))
                n_nodes.append(states[-1][-1].number_of_nodes())
                if len(data[t]) > n + self.action_range:
                    actions_xy = [d[2] for d in data[t][n:n + self.action_range]]
                else:
                    actions_xy = [d[2] for d in data[t][n:]]
                actions_xy = np.array(actions_xy)
                actions_xy = np.mean(actions_xy, axis=0)
                actions[-1].append(actions_xy)
            states[-1] = dgl.batch(states[-1])
            actions[-1] = torch.tensor(np.array(actions[-1]))
            trial_actions_padded = torch.zeros(self.max_len, actions[-1].size(1))
            trial_actions_padded[:actions[-1].size(0), :] = actions[-1]
            actions[-1] = trial_actions_padded
        return states, actions, lens, n_nodes

    def get_test(self, trial, data, step=10):
        # retrieve state embeddings and actions from cached file
        states = []
        actions = []
        trial_len = []
        trial_len += [(trial, n) for n in range(0, len(data[trial]), step)]
        for t, n in trial_len:
            video = data[t][n][0]
            state = self._get_frame_graph(video, data[t][n][1])
            if len(data[t]) > n + self.action_range:
                actions_xy = [d[2] for d in data[t][n:n + self.action_range]]
            else:
                actions_xy = [d[2] for d in data[t][n:]]
            actions_xy = np.array(actions_xy)
            actions_xy = np.mean(actions_xy, axis=0)
            actions.append(actions_xy)
            states.append(state)
        # states = torch.stack(states)
        states = dgl.batch(states)
        actions = torch.tensor(np.array(actions))
        return states, actions

    def __getitem__(self, idx):
        ep_trials = [idx * self.num_trials + t for t in range(self.num_trials)]
        dem_expected_states, dem_expected_actions, dem_expected_lens, dem_expected_nodes = self.get_trial(
            ep_trials[:-1], self.data_expected, step=self.action_range
        )
        dem_unexpected_states, dem_unexpected_actions, dem_unexpected_lens, dem_unexpected_nodes = self.get_trial(
            ep_trials[:-1], self.data_unexpected, step=self.action_range
        )
        query_expected_frames, target_expected_actions = self.get_test(
            ep_trials[-1], self.data_expected, step=self.action_range
        )
        query_unexpected_frames, target_unexpected_actions = self.get_test(
            ep_trials[-1], self.data_unexpected, step=self.action_range
        )
        return dem_expected_states, dem_expected_actions, dem_expected_lens, dem_expected_nodes, \
            dem_unexpected_states, dem_unexpected_actions, dem_unexpected_lens, dem_unexpected_nodes, \
            query_expected_frames, target_expected_actions, \
            query_unexpected_frames, target_unexpected_actions

    def __len__(self):
        return len(self.path_list_exp)


if __name__ == '__main__':
    types = ['preference', 'multi_agent', 'inaccessible_goal',
             'efficiency_irrational', 'efficiency_time', 'efficiency_path',
             'instrumental_no_barrier', 'instrumental_blocking_barrier', 'instrumental_inconsequential_barrier']
    for t in types:
        ttd = TestTransitionDatasetSequence(path='/datasets/external/bib_evaluation_1_1/', task_type=t, process_data=0,
                                            mode='test')
        for i in range(ttd.__len__()):
            print(i, end='\r')
            dem_expected_states, dem_expected_actions, dem_expected_lens, dem_expected_nodes, \
                dem_unexpected_states, dem_unexpected_actions, dem_unexpected_lens, dem_unexpected_nodes, \
                query_expected_frames, target_expected_actions, \
                query_unexpected_frames, target_unexpected_actions = ttd.__getitem__(i)
            for j in range(8):
                if not torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0.]) in dem_expected_states[j].ndata['type']:
                    print(i)
                if not torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0.]) in dem_unexpected_states[j].ndata['type']:
                    print(i)
            if not torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0.]) in query_expected_frames.ndata['type']:
                print(i)
            if not torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0.]) in query_unexpected_frames.ndata['type']:
                print(i)
