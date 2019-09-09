from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from ast import literal_eval

__author__ = "tristone"
__copyright__ = "Copyright (c) 2019"
__email__ = "tristone13th@outlook.com"

USE_CUDA = False
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def printAction(action, window):
    sum_list = []
    i = 0
    while i < len(action):
        sum_list.append(sum(action[i: i + window: 1]) / window)
        i = i + window
    plt.plot(sum_list)
    plt.show()


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


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