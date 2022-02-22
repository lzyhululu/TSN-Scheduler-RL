from .random import RandomActor
import math
import numpy as np
import logging
import os
import copy


def init_logger(filename):
    """ initialize logger config

    Parameters
    ----------
    filename : str
        filename of the log
    """
    logging.basicConfig(level=logging.INFO, filename=filename + ".log")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)


class ReplayBuffer:
    """a circular queue based on numpy array, supporting batch put and batch get"""
    def __init__(self, shape, dtype=np.float32):
        self.buffer = np.empty(shape=shape, dtype=dtype)
        self.head = 0
        self.capacity = len(self.buffer)

    def put(self, data):
        """put data to

        Parameters
        ----------
        data: numpy array
            data to add
        """
        head = self.head
        n = len(data)
        if head + n <= self.capacity:
            self.buffer[head:head+n] = data
            self.head = (self.head + n) % self.capacity
        else:
            split = self.capacity - head
            self.buffer[head:] = data[:split]
            self.buffer[:n - split] = data[split:]
            self.head = split
        return n

    def get(self, index):
        """get items

        Parameters
        ----------
        index: int or numpy array
            it can be any numpy supported index
        """
        return self.buffer[index]

    def clear(self):
        """clear replay buffer"""
        self.head = 0


class EpisodesBufferEntry:
    """Entry for episode buffer"""
    def __init__(self):
        self.features = []
        self.actions = []
        self.next_features = []
        self.rewards = []

    def append(self, feature, action, next_feature, reward):
        self.features.append(feature.copy())
        self.actions.append(action)
        self.next_features.append(next_feature.copy())
        self.rewards.append(reward)

    def full_reset(self, feature, action, next_feature, reward, idx):
        self.features[idx] = feature.copy()
        self.actions[idx] = action
        self.next_features[idx] = next_feature.copy()
        self.rewards[idx] = reward


class EpisodesBuffer:
    """Initialize buffer to store a whole episode for all agents
       one entry for one agent

       this for DDPG
    """
    def __init__(self, capacity):
        self.total_view = []
        self.total_features = []
        self.total_actions = []
        self.total_rewards = []
        self.total_next_view = []
        self.total_next_features = []
        self.buffer = {}
        self.capacity = capacity
        self.buff_counter = 0
        self.is_full = False

    def record_step(self, ids, obs, acts, next_obs, rewards, num_actions):
        """record transitions (s, a, r, terminal) in a step"""
        buffer = self.buffer
        index = np.random.permutation(len(ids))

        if self.is_full:  # extract loop invariant in else part
            idx = self.buff_counter % self.capacity
            self.total_view[idx] = obs[0].copy()
            self.total_features[idx] = obs[1].copy()
            self.total_actions[idx] = acts
            self.total_rewards[idx] = np.sum(rewards)
            self.total_next_view[idx] = next_obs[0].copy()
            self.total_next_features[idx] = next_obs[1].copy()
            for i in range(len(ids)):
                entry = buffer.get(ids[i])
                if entry is None:
                    continue
                entry.full_reset(obs[1][i], acts[num_actions*i: num_actions * i + num_actions],
                                 next_obs[1][i], rewards[i], idx)
        else:
            self.total_view.append(obs[0].copy())
            self.total_features.append(obs[1].copy())
            self.total_actions.append(acts)
            self.total_rewards.append(np.sum(rewards))
            self.total_next_view.append(next_obs[0].copy())
            self.total_next_features.append(next_obs[1].copy())
            for i in range(len(ids)):
                i = index[i]
                entry = buffer.get(ids[i])
                if entry is None:
                    entry = EpisodesBufferEntry()
                    buffer[ids[i]] = entry
                    if self.buff_counter >= self.capacity:
                        self.is_full = True
                entry.append(obs[1][i], acts[num_actions*i: num_actions*i + num_actions], next_obs[1][i], rewards[i])

        self.buff_counter += 1

    def reset(self):
        """ clear replay buffer """
        self.buffer = {}
        self.is_full = False

    def episodes(self):
        """ get episodes """
        return self.buffer.values()

    def counter(self):
        """ count num of episodes """
        return self.buff_counter


# decay schedulers
def exponential_decay(now_step, total_step, final_value, rate):
    """exponential decay scheduler"""
    decay = math.exp(math.log(final_value)/total_step ** rate)
    return max(final_value, 1 * decay ** (now_step ** rate))


def linear_decay(now_step, total_step, final_value):
    """linear decay scheduler"""
    decay = (1 - final_value) / total_step
    return max(final_value, 1 - decay * now_step)


def piecewise_decay(now_step, anchor, anchor_value):
    """piecewise linear decay scheduler

    Parameters
    ---------
    now_step : int
        current step
    anchor : list of integer
        step anchor
    anchor_value: list of float
        value at corresponding anchor
    """
    i = 0
    while i < len(anchor) and now_step >= anchor[i]:
        i += 1

    if i == len(anchor):
        return anchor_value[-1]
    else:
        return anchor_value[i-1] + (now_step - anchor[i-1]) * \
                                   ((anchor_value[i] - anchor_value[i-1]) / (anchor[i] - anchor[i-1]))


# eval observation set generator
def sample_observation(env, handles, n_obs=-1, step=-1):
    """Sample observations by random actors.
    These samples can be used for evaluation

    Parameters
    ----------
    env : environment
    handles: list of handle
    n_obs : int
        number of observation
    step : int
        maximum step

    Returns
    -------
    ret : list of raw observation
        raw observation for every group
        the format of raw observation is tuple(view, feature)
    """
    models = [RandomActor(env, handle) for handle in handles]

    n = len(handles)
    views = [[] for _ in range(n)]
    features = [[] for _ in range(n)]

    done = False
    step_ct = 0
    while not done:
        obs = [env.get_observation(handle) for handle in handles]
        ids = [env.get_agent_id(handle) for handle in handles]

        for i in range(n):
            act = models[i].infer_action(obs[i], ids[i])
            env.set_action(handles[i], act)

        done = env.step()

        # record steps
        if step_ct == 0:
            step_ct += 1
            continue
        for i in range(n):
            views[i].append(copy.deepcopy(obs[i][0]))
            features[i].append(copy.deepcopy(obs[i][1]))

        if step != -1 and step_ct > step:
            break

        if step_ct % 100 == 0:
            print("sample step %d" % step_ct)

        step_ct += 1

    for i in range(n):
        views[i] = np.array(views[i], dtype=np.float32).reshape((-1,) + env.get_view_space(handles[i]))
        features[i] = np.array(features[i], dtype=np.float32).reshape((step_ct, -1,) + env.get_feature_space(handles[i]))

    if n_obs != -1:
        for i in range(n):
            choice_list = np.random.choice(step_ct, n_obs)
            views[i] = views[i][choice_list]
            features[i] = features[i][choice_list]

    ret = [(v, f) for v, f in zip(views, features)]
    return ret


def has_gpu():
    """ check where has a nvidia gpu """
    ret = os.popen("nvidia-smi -L 2>/dev/null").read()
    return ret.find("GPU") != -1
