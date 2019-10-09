import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.autograd import Variable
from ast import literal_eval

USE_CUDA = False
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

class Batch(object):
    """
    This class is designed for implementing batch
    """

    def __init__(self, dataset=None, capacity=100):
        super(Batch, self).__init__()
        self.__capacity = capacity
        if dataset is None:
            self.__dataset = []
        else:
            assert len(dataset) <= capacity
            self.__dataset = dataset

    def sample(self, sample_size):
        assert len(self.__dataset) >= sample_size
        dataset = random.sample(self.__dataset, sample_size)
        prestate_batch = []
        action_batch = []
        reward_batch = []
        state_batch = []

        for item in dataset:
            prestate_batch.append(item[0])
            action_batch.append(item[1])
            reward_batch.append(item[2])
            state_batch.append(item[3])

        prestate_batch = np.array(prestate_batch).reshape(sample_size, -1)
        action_batch = np.array(action_batch).reshape(sample_size, -1)
        reward_batch = np.array(reward_batch).reshape(sample_size, -1)
        state_batch = np.array(state_batch).reshape(sample_size, -1)
        return prestate_batch, action_batch, reward_batch, state_batch

    def add(self, element):
        if len(self.__dataset) < self.__capacity:
            self.__dataset.append(element)
        else:
            self.__dataset.pop(0)
            self.__dataset.append(element)

    def __len__(self):
        return len(self.__dataset)


class Actor(object):
    """
    Actor base class
    """

    def __init__(self, len_state, len_action, gamma=0.5, tau=0.5, with_BN=False):
        super(Actor, self).__init__()
        self.net = ActorNet(in_dim=len_state, out_dim=len_action, with_BN=with_BN)
        self.action_trace = {}
        self.in_dim = len_state
        self.out_dim = len_action
        self._gamma = gamma
        self._tau = tau


class SampleActor(Actor):
    """
    Sample actor base class
    """

    def __init__(self, len_state, len_action, scale, gamma=0.5, tau=0.5, lr=0.2, with_BN=False):
        super(SampleActor, self).__init__(
            len_state, len_action, gamma, tau, with_BN=with_BN)

        self.scale = scale
        self._lr = lr
        self._optimizer = torch.optim.Adam(self.net.parameters(), lr=self._lr)

    # return net to make sure that TargetCritic can copy
    def getNet(self):
        net = self.net
        return net

    def getAction(self, state):
        """
        decision maker with specified noise
        """
        output = self.net(state)
        rand = torch.rand(self.out_dim)
        output = output + rand
        return output

    def update(self, prestate_batch, sample_critic):
        """
        update parameters once a batch
        :param prestate_batch: the batch of previous states
        :param sample_critic: the sample critic
        """
        self._optimizer.zero_grad()

        prestate_batch = to_tensor(prestate_batch)

        action_batch = self.net(prestate_batch)
        policy_loss = -sample_critic.getQValue([
            prestate_batch,
            action_batch
        ])

        # print(policy_loss)
        policy_loss = policy_loss.mean()
        # print("mean: ", policy_loss)
        policy_loss.backward()

        # print("policy loss is", policy_loss)
        # for para in self.net.parameters():
        #     print(para.grad)

        print("=============================================================================")
        for name, para in self.net.named_parameters():
            print("Name: ", name)
            print("Grid: ", para.grad)
            print()
        print("actor loss: ", policy_loss)
        print("=============================================================================")

        self._optimizer.step()


class TargetActor(Actor):
    """
    target actor class
    """

    def __init__(self, len_state, len_action, gamma=0.5, tau=0.5, with_BN=False):
        super(TargetActor, self).__init__(
            len_state, len_action, gamma, tau, with_BN=with_BN)

    def getAction(self, state):
        output = self.net(state)
        # state = to_numpy(state).tolist()[0]
        # if state[col_socket_id] not in self.action_trace.keys():
        #     self.action_trace[state[col_socket_id]] = []
        # self.action_trace[state[col_socket_id]].append(float(output[0][0]) * 100000)
        return output

    def getNet(self):
        net = self.net
        return net

    def update(self, sample_actor):
        for target_param, param in zip(self.net.parameters(), sample_actor.getNet().parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self._tau) + param.data * self._tau
            )


class ActorNet(nn.Module):
    """
    Actor neural network structure
    """

    def __init__(self, in_dim=5, n_hidden_1=10, n_hidden_2=10, out_dim=5, init_w=3e-2, with_BN=False):
        super(ActorNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.ELU = nn.ELU()
        self.Tanh = nn.Tanh()
        self.LeakyReLU = nn.LeakyReLU()
        self.with_BN = with_BN
        if self.with_BN:
            self.bn_input = nn.BatchNorm1d(in_dim)
            self.bn_hidden_1 = nn.BatchNorm1d(n_hidden_1)
            self.bn_hidden_2 = nn.BatchNorm1d(n_hidden_2)
            self.bn_output = nn.BatchNorm1d(out_dim)
        self.init_weights(init_w)

    def forward(self, x):
        if self.with_BN:
            x = self.bn_input(x)
        x = self.layer1(x)

        if self.with_BN:
            x = self.bn_hidden_1(x)
        x = self.Tanh(x)
        x = self.layer2(x)

        if self.with_BN:
            x = self.bn_hidden_2(x)
        x = self.Tanh(x)
        x = self.layer3(x)

        if self.with_BN:
            x = self.bn_output(x)
        x = self.sigmoid(x)
        return x

    def init_weights(self, init_w):
        self.layer1.weight.data = fanin_init(self.layer1.weight.data.size())
        self.layer2.weight.data = fanin_init(self.layer2.weight.data.size())
        self.layer3.weight.data.uniform_(-init_w, init_w)


class Critic(object):
    """
    Critic basic class
    """

    def __init__(self, len_state, len_action, gamma, tau, with_BN):
        super(Critic, self).__init__()
        self.net = CriticNet(in_dim=len_state + len_action, out_dim=1, with_BN=with_BN)

        self._in_dim = len_state + len_action
        self._gamma = gamma
        self._tau = tau

    def getQValue(self, input_data):
        state, action = input_data
        input_data = torch.cat([state, action], 1)
        return self.net(input_data)


class SampleCritic(Critic):
    """
    Sample Critic
    """

    def __init__(self, len_state, len_action, gamma=0.5, tau=0.5, lr=0.2, with_BN=False):
        super(SampleCritic, self).__init__(
            len_state, len_action, gamma, tau, with_BN=with_BN)
        self._lr = lr
        self._criterion = nn.MSELoss()
        self._optimizer = torch.optim.Adam(self.net.parameters(), lr=self._lr, weight_decay=1e-2)

    def getNet(self):
        """ return net to make sure that TargetCritic can copy """
        net = self.net
        return net

    def update(self, prestate_batch, action_batch, reward_batch, state_batch, target_critic, target_actor):
        next_q_values = target_critic.getQValue([
            to_tensor(state_batch, volatile=True),
            target_actor.getAction(to_tensor(state_batch, volatile=True))
        ])

        # 表示不需要对目标评论员求导
        next_q_values.volatile = False

        target_q_batch = to_tensor(reward_batch) + self._gamma * next_q_values

        self.net.zero_grad()

        q_batch = self.getQValue([
            to_tensor(prestate_batch),
            to_tensor(action_batch)
        ])

        value_loss = self._criterion(q_batch, target_q_batch)

        value_loss.backward()
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        for name, para in self.net.named_parameters():
            print("Name: ", name)
            print("Grid: ", para.grad)
            print()
        print("Critic loss: ", value_loss)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        self._optimizer.step()


class TargetCritic(Critic):
    """
    Target Critic
    """

    def __init__(self, len_state, len_action, gamma=0.5, tau=0.5, with_BN=False):
        super(TargetCritic, self).__init__(
            len_state, len_action, gamma, tau, with_BN=with_BN)

    def getNet(self):
        """ return net to make sure that TargetCritic can copy """
        net = self.net
        return net

    def update(self, sample_critic):
        for target_param, param in zip(self.net.parameters(), sample_critic.getNet().parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self._tau) + param.data * self._tau
            )


class CriticNet(nn.Module):
    """
    Critic's neural network structure
    """

    def __init__(self, in_dim=5, n_hidden_1=10, n_hidden_2=10, out_dim=5, init_w=3e-3, with_BN=False):
        super(CriticNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.LeakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.ELU = nn.ELU()
        self.with_BN = with_BN
        if self.with_BN:
            self.bn_input = nn.BatchNorm1d(in_dim)
            self.bn_hidden_1 = nn.BatchNorm1d(n_hidden_1)
            self.bn_hidden_2 = nn.BatchNorm1d(n_hidden_2)
            self.bn_output = nn.BatchNorm1d(out_dim)
        self.init_weights(init_w)

    def forward(self, x):
        if self.with_BN:
            x = self.bn_input(x)
        x = self.layer1(x)

        if self.with_BN:
            x = self.bn_hidden_1(x)
        x = self.Tanh(x)
        x = self.layer2(x)

        if self.with_BN:
            x = self.bn_hidden_2(x)
        x = self.Tanh(x)
        x = self.layer3(x)
        return x

    def init_weights(self, init_w):
        self.layer1.weight.data = fanin_init(self.layer1.weight.data.size())
        self.layer2.weight.data = fanin_init(self.layer2.weight.data.size())
        self.layer3.weight.data.uniform_(-init_w, init_w)

