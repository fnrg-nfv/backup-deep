import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import collections
from ast import literal_eval
from sfcbased.model import *


class Action:
    pass


class Environment:

    @abstractmethod
    def get_reward(self, model: Model, sfc_index: int, decision: Decision, test_env: TestEnv):
        return 0

    @abstractmethod
    def get_state(self, model: Model, sfc_index: int):
        return []


class NormalEnvironment(Environment):
    def get_reward(self, model: Model, sfc_index: int, decision: Decision, test_env: TestEnv):
        return 0

    def get_state(self, model: Model, sfc_index: int):
        return []


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'new_state'])


class ExperienceBuffer:
    """
    Experience buffer class
    """

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        """
        Append an experience item
        :param experience: experience item
        :return: nothing
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """
        Sample a batch from this buffer
        :param batch_size: sample size
        :return: batch: List
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states = zip(*[self.buffer[idx] for idx in indices])
        return states, actions, rewards, next_states


def fanin_init(size, fanin=None):
    """
    Init weights
    :param size: tensor size
    :param fanin:
    :return:
    """
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


def printAction(action, window):
    sum_list = []
    i = 0
    while i < len(action):
        sum_list.append(sum(action[i: i + window: 1]) / window)
        i = i + window
    plt.plot(sum_list)
    plt.show()


def readDataset(path):
    data = []
    dataset = csv.reader(open(path, encoding='utf_8_sig'), delimiter=',')
    for rol in dataset:
        data.append(rol)
    data = data[1:len(data):1]
    for i in range(len(data)):
        data[i][0] = literal_eval(data[i][0])
        data[i][1] = literal_eval(data[i][1])
        data[i][3] = literal_eval(data[i][3])
        data[i][2] = float(data[i][2])
    return data


def formatnum(x, pos):
    return '$%.1f$x$10^{4}$' % (x / 10000)


def plotActionTrace(action_trace):
    for key in action_trace.keys():
        plt.plot(action_trace[key], label=str(int(key)))
    plt.xlabel("Iterations")
    plt.ylabel("Action")
    plt.title("Agent's Output with Time")
    plt.ylim((0, 100000))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    pass


if __name__ == '__main__':
    main()
