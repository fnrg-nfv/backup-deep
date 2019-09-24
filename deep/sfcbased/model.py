from typing import List
import networkx as nx
from enum import Enum, unique
import torch.nn as nn
from abc import ABC, abstractmethod
import torch.nn.functional as F
import random
from utils import *


class VirtualException(BaseException):
    def __init__(self, _type, _func):
        BaseException(self)


class BaseObject(object):
    def __repr__(self):
        '''
        When function print() is called, this function will determine what to display
        :return: the __str__() result of current instance
        '''
        return self.__str__()


class Path(BaseObject):
    '''
    This class is denoted as the path from one server to another
    '''

    def __init__(self, start: int, destination: int, path: List, latency: float):
        '''
        This class is denoted as a path
        :param start: start server
        :param destination: destination server
        :param path: path list
        :param latency: latency requirement
        '''
        self.start = start
        self.destination = destination
        self.path = path
        self.latency = latency
        self.path_length = len(path)

    def __str__(self):
        return self.path.__str__()


class Monitor(BaseObject):
    '''
    Designed for Monitoring the actions of whole system
    '''
    action_list = []

    @classmethod
    def log(cls, content: str):
        cls.action_list.append(content)

    @classmethod
    def change_deploying(cls, sfc_index: int):
        cls.log("")
        cls.log("The state of SFC {} changes to Deploying".format(sfc_index))

    @classmethod
    def change_failed(cls, sfc_index: int):
        cls.log("The state of SFC {} changes to Failed".format(sfc_index))

    @classmethod
    def change_expired(cls, sfc_index: int):
        cls.log("")
        cls.log("The state of SFC {} changes to Expired".format(sfc_index))

    @classmethod
    def change_running(cls, sfc_index: int):
        cls.log("The state of SFC {} changes to Running".format(sfc_index))

    @classmethod
    def deploy_server(cls, sfc_index: int, vnf_index: int, server_id: int):
        cls.log("SFC {} VNF {} deploy on server {}".format(sfc_index, vnf_index, server_id))

    @classmethod
    def path_occupied(cls, path: Path, sfc_index: int):
        cls.log("Path {} is occupied by SFC {}".format(path, sfc_index))

    @classmethod
    def latency_occupied_change(cls, sfc_index: int, before: float, after: float):
        cls.log("Latency occupied by SFC {} changes from {} to {}".format(sfc_index, before, after))

    @classmethod
    def computing_resource_change(cls, server_id: int, before: int, after: int):
        cls.log("The computing resource of server {} from {} changes to {}".format(server_id, before, after))

    @classmethod
    def bandwidth_change(cls, start: int, destination: int, before: int, after: int):
        cls.log("The bandwidth of link from {} to {} changes from {} to {}".format(start, destination, before, after))

    @classmethod
    def print_log(cls):
        for item in cls.action_list:
            print(item)


class Batch(object):
    '''
    This class is designed for implementing batch
    '''

    def __init__(self, dataset=None, capacity=100):
        super(Batch, self).__init__()
        self.__capacity = capacity
        if dataset == None:
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
        if (len(self.__dataset) < self.__capacity):
            self.__dataset.append(element)
        else:
            self.__dataset.pop(0)
            self.__dataset.append(element)

    def __len__(self):
        return len(self.__dataset)


class Actor(object):
    '''
    Actor base class
    '''

    def __init__(self, len_state, len_action, gamma=0.5, tau=0.5, with_BN=False):
        super(Actor, self).__init__()
        self.net = ActorNet(in_dim=len_state, out_dim=len_action, with_BN=with_BN)
        self.action_trace = {}
        self.in_dim = len_state
        self.out_dim = len_action
        self._gamma = gamma
        self._tau = tau


class SampleActor(Actor):
    '''
    Sample actor base class
    '''

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
        '''
        decision maker with specified noise
        '''
        output = self.net(state)
        rand = torch.rand(self.out_dim)
        output = output + rand
        return output

    def update(self, prestate_batch, sample_critic):
        '''
        update parameters once a batch
        :param prestate_batch: the batch of previous states
        :param sample_critic: the sample critic
        '''
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
    '''
    target actor class
    '''

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
    '''
    Actor neural network structure
    '''

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
    '''
    Critic basic class
    '''

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
    '''
    Sample Critic
    '''

    def __init__(self, len_state, len_action, gamma=0.5, tau=0.5, lr=0.2, with_BN=False):
        super(SampleCritic, self).__init__(
            len_state, len_action, gamma, tau, with_BN=with_BN)
        self._lr = lr
        self._criterion = nn.MSELoss()
        self._optimizer = torch.optim.Adam(self.net.parameters(), lr=self._lr, weight_decay=1e-2)

    def getNet(self):
        ''' return net to make sure that TargetCritic can copy '''
        net = self.net
        return net

    def update(self, prestate_batch, action_batch, reward_batch, state_batch, target_critic, target_actor):
        '''
        '''
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
    '''
    Target Critic
    '''

    def __init__(self, len_state, len_action, gamma=0.5, tau=0.5, with_BN=False):
        super(TargetCritic, self).__init__(
            len_state, len_action, gamma, tau, with_BN=with_BN)

    def getNet(self):
        ''' return net to make sure that TargetCritic can copy '''
        net = self.net
        return net

    def update(self, sample_critic):
        for target_param, param in zip(self.net.parameters(), sample_critic.getNet().parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self._tau) + param.data * self._tau
            )


class CriticNet(nn.Module):
    '''
    Critic's neural network structure
    '''

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


@unique
class State(Enum):
    expired = 0
    failed = 1
    running = 2
    deploying = 3
    future = 4


class VNF(BaseObject):
    '''
    This class is denoted as a VNF
    '''

    def __init__(self, latency: float, computing_resource: int):
        '''
        Initialization
        :param latency: the init latency processed
        :param computing_resource: the init computing_resource requirement processed
        '''
        self.latency = latency
        self.computing_resource = computing_resource
        self.deploy_decision = -1  # -1 represents not deployed

    def __str__(self):
        return "(%f, %d)" % (self.latency, self.computing_resource)


class SFC(BaseObject):
    '''
    This class is denoted as a SFC
    '''

    def __init__(self, vnf_list: List[VNF], latency: float, throughput: int, s: int, d: int, time: float, TTL: int):
        '''
        Initialization
        :param vnf_list: the VNFs contained
        :param latency: totally latency required
        :param throughput: totally throughput required
        :param s: start
        :param d: destination
        :param time: arriving time
        :param TTL: time to live
        '''
        self.vnf_list = vnf_list
        self.latency = latency
        self.throughput = throughput
        self.s = s
        self.d = d
        self.time = time
        self.TTL = TTL
        self.state = State.future

        self.latency_occupied = 0  # including VNF process latency
        self.paths_occupied = []  # links occupied, the throughput occupied must be released, each list item is a tuple of edge such as (1, 2) denote the edge from 1->2 or 2->1

    def __len__(self):
        return len(self.vnf_list)

    def __getitem__(self, index):
        return self.vnf_list[index]

    def __setitem__(self, index, value):
        self.vnf_list[index] = value

    def __str__(self):
        '''
        Display in console with specified format.
        :return: display string
        '''
        return "(VNFs: {}, latency: {}, throughput: {}, from {}->{}, time: {}, TTL: {})".format(self.vnf_list,
                                                                                                self.latency,
                                                                                                self.throughput, self.s,
                                                                                                self.d,
                                                                                                self.time, self.TTL)


class Model(BaseObject):
    '''
    This class is denoted as the model, a model contains following:
    1. the topology of the whole network
    2. the ordered SFCs need to be deployed
    '''

    def __init__(self, topo: nx.Graph, sfc_list: List[SFC]):
        '''
        Initialization
        :param topo: network topology
        :param sfc_list: SFCs set
        '''
        self.topo = topo
        self.sfc_list = sfc_list

    def __str__(self):
        '''
        Display in console with specified format.
        :return: display string
        '''
        return "TOPO-nodes:\n{}\nTOPO-edges:\n{}\nSFCs:\n{}".format(self.topo.nodes.data(), self.topo.edges.data(),
                                                                    self.sfc_list)

    def occupy_path(self, path: Path, throughput: int):
        '''
        occupy the bandwidth of path
        :param path: given path
        :param throughput: given throughput
        :return: nothing
        '''
        for i in range(len(path.path) - 1):
            self.topo[path.path[i]][path.path[i + 1]]["bandwidth"] -= throughput
            Monitor.bandwidth_change(path.path[i], path.path[i + 1],
                                     self.topo[path.path[i]][path.path[i + 1]]["bandwidth"] + throughput,
                                     self.topo[path.path[i]][path.path[i + 1]]["bandwidth"])

    def revert_failed(self, failed_sfc_index: int, failed_vnf_index: int):
        '''
        deal with deploy failed condition, revert the state of vnf and topo
        :param failed_sfc_index: the index of failed sfc
        :param failed_vnf_index: the index of failed vnf
        :return: nothing
        '''
        # computing resource
        for i in range(0, failed_vnf_index):
            before = self.topo.nodes[self.sfc_list[failed_sfc_index].vnf_list[i].deploy_decision]["computing_resource"]
            self.topo.nodes[self.sfc_list[failed_sfc_index].vnf_list[i].deploy_decision]["computing_resource"] += \
            self.sfc_list[failed_sfc_index].vnf_list[i].computing_resource
            after = self.topo.nodes[self.sfc_list[failed_sfc_index].vnf_list[i].deploy_decision]["computing_resource"]
            Monitor.computing_resource_change(self.sfc_list[failed_sfc_index].vnf_list[i].deploy_decision, before,
                                              after)
        # bandwidth
        for path in self.sfc_list[failed_sfc_index].paths_occupied:
            for i in range(len(path.path) - 1):
                before = self.topo[path.path[i]][path.path[i + 1]]["bandwidth"]
                self.topo[path.path[i]][path.path[i + 1]]["bandwidth"] += self.sfc_list[failed_sfc_index].throughput
                after = self.topo[path.path[i]][path.path[i + 1]]["bandwidth"]
                Monitor.bandwidth_change(path.path[i], path.path[i + 1], before, after)
        # state
        self.sfc_list[failed_sfc_index].state = State.failed
        Monitor.change_failed(failed_sfc_index)


class DecisionMaker(ABC):
    '''
    The class used to make deploy decision
    '''

    def __init__(self):
        super(DecisionMaker, self).__init__()

    def is_path_throughtput_available(self, model: Model, path: List, throughput: int):
        '''
        Determine if the throughput requirement of the given path is meet
        :param model: given model
        :param path: given path
        :param throughput: given throughput requirement
        :return: true or false
        '''
        for i in range(len(path) - 1):
            if model.topo[path[i]][path[i + 1]]["bandwidth"] < throughput:
                return False
        return True

    def path_latency(self, model: Model, path: List):
        '''
        Determine if the latency requirement of the given path is meet
        :param model: given model
        :param path: given path
        :return: latency of given path
        '''
        path_latency = 0
        for i in range(len(path) - 1):
            path_latency += model.topo[path[i]][path[i + 1]]["latency"]  # calculate latency of path
        return path_latency

    def is_throughput_and_latency_available(self, model: Model, server_id: int, cur_sfc_index: int, cur_vnf_index: int):
        '''
        Determine if node server_id has enough throughput and latency.
        Requirements:
            throughput:
                1. from pre -> cur
            latency:
                1. req - ocu - VNFs process latency - cur path latency >= 0
        if pre == cur:
            only need to meet latency1
        else:
            need to meet thoughput1 and latency1
        :param model: model
        :param server_id: the server which current vnf will be placed on
        :param cur_sfc_index: the current index of sfc
        :param cur_vnf_index: the current index of vnf
        :return: false(can't be placed) or true
        '''

        # if deploy first vnf
        pre_server_id = model.sfc_list[cur_sfc_index].s if cur_vnf_index == 0 else model.sfc_list[cur_sfc_index][
            cur_vnf_index - 1].deploy_decision

        # if deploy final vnf/ determine final_latency
        final_latency = 0
        if cur_vnf_index == len(model.sfc_list[cur_sfc_index]) - 1:
            if server_id != model.sfc_list[cur_sfc_index].d:  # the server_id isn't the destination node
                final_latency = float("inf")
                for path in nx.all_simple_paths(model.topo, server_id, model.sfc_list[
                    cur_sfc_index].d):  # the server_id is not the destination node
                    cur_latency = self.path_latency(model, path)
                    if cur_latency < final_latency and self.is_path_throughtput_available(model, path, model.sfc_list[
                        cur_sfc_index].throughput):
                        final_latency = cur_latency

        # deploy on the same server
        if pre_server_id == server_id:
            cur_sfc_index
            #  latency1 requirement
            process_latency = 0
            for i in range(cur_vnf_index, len(model.sfc_list[cur_sfc_index])):
                process_latency += model.sfc_list[cur_sfc_index][i].latency
            slack_latency = model.sfc_list[cur_sfc_index].latency - model.sfc_list[
                cur_sfc_index].latency_occupied - process_latency - final_latency  # calculate residual/slack latency
            return True if slack_latency >= 0 else False

        # deploy on the different server
        path_available = 0
        for path in nx.all_simple_paths(model.topo, pre_server_id, server_id):
            # determine if throughput1 is available
            throughput_flag = 1
            for i in range(len(path) - 1):
                if model.topo[path[i]][path[i + 1]]["bandwidth"] < model.sfc_list[cur_sfc_index].throughput:
                    throughput_flag = 0
                    break

            if throughput_flag == 0:
                continue

            # determine if latency1 is available
            path_latency = 0
            process_latency = 0
            for i in range(len(path) - 1):
                path_latency += model.topo[path[i]][path[i + 1]]["latency"]  # calculate latency of path

            for i in range(cur_vnf_index, len(model.sfc_list[cur_sfc_index])):
                process_latency += model.sfc_list[cur_sfc_index][i].latency  # calculate latency of processing

            slack_latency = model.sfc_list[cur_sfc_index].latency - model.sfc_list[
                cur_sfc_index].latency_occupied - process_latency - path_latency - final_latency  # calculate residual/slack latency

            latency_flag = 1 if slack_latency >= 0 else 0

            if latency_flag == 1:
                path_available = 1
                break
        return True if path_available else False

    def available_paths(self, model: Model, start_server_id: int, destination_server_id: int, cur_sfc_index: int,
                        cur_vnf_index: int):
        '''
        Return all the available paths from one server to another in current vnf
        Note: When call this, remember to make sure the two server is not the same
        :param model: model
        :param start_server_id: start server id
        :param destination_server_id: destination server id
        :param cur_sfc_index: index of current sfc
        :param cur_vnf_index: index of current vnf
        :return: all the available paths
        '''

        assert start_server_id != destination_server_id
        paths = []
        for path in nx.all_simple_paths(model.topo, start_server_id, destination_server_id):
            # determine if throughput1 is available
            if self.is_path_throughtput_available(model, path, model.sfc_list[cur_sfc_index].throughput):
                # calculate latency of path
                path_latency = 0
                for i in range(len(path) - 1):
                    path_latency += model.topo[path[i]][path[i + 1]]["latency"]  # calculate latency of path

                # calculate latency of processing
                process_latency = 0
                for i in range(cur_vnf_index, len(model.sfc_list[cur_sfc_index])):
                    process_latency += model.sfc_list[cur_sfc_index][i].latency  # calculate latency of processing

                # if deploy final vnf/ determine final_latency
                final_latency = 0
                if cur_vnf_index == len(model.sfc_list[cur_sfc_index]) - 1:
                    if destination_server_id != model.sfc_list[
                        cur_sfc_index].d:  # the server_id isn't the destination node
                        final_latency = float("inf")
                        for path in nx.all_simple_paths(model.topo, destination_server_id, model.sfc_list[
                            cur_sfc_index].d):  # the server_id is not the destination node
                            cur_latency = self.path_latency(model, path)
                            if cur_latency < final_latency and self.is_path_throughtput_available(model, path,
                                                                                                  model.sfc_list[
                                                                                                      cur_sfc_index].throughput):
                                final_latency = cur_latency

                # calculate slack latency
                slack_latency = model.sfc_list[cur_sfc_index].latency - model.sfc_list[
                    cur_sfc_index].latency_occupied - process_latency - path_latency - final_latency  # calculate residual/slack latency

                if slack_latency >= 0:
                    paths.append(Path(start_server_id, destination_server_id, path, path_latency))

        return paths

    def narrow_node_set(self, model: Model, cur_sfc_index: int, cur_vnf_index: int):
        '''
        Used to narrow available node set
        :param model: model
        :param cur_sfc_index: cur processing sfc index
        :param cur_vnf_index: cur processing vnf index
        :return: server sets
        '''
        server_set = []
        for i in range(len(model.topo.nodes)):
            if model.topo.nodes[i]["computing_resource"] >= model.sfc_list[cur_sfc_index][
                cur_vnf_index].computing_resource and self.is_throughput_and_latency_available(model, i, cur_sfc_index,
                                                                                               cur_vnf_index):
                server_set.append(i)
        return server_set

    def shortest_path(self, model: Model, start_index: int, destination_index: int):
        '''
        return the shortest path
        :param model: given model
        :param start_index: index of start server
        :param destination_index: index of destination server
        :return: shortest path or False
        '''
        final_latency = float("inf")
        final_path = False
        for path in nx.all_simple_paths(model.topo, start_index,
                                        destination_index):  # the server_id is not the destination node
            cur_latency = self.path_latency(model, path)
            if cur_latency < final_latency:
                final_latency = cur_latency
                final_path = Path(start_index, destination_index, path, final_latency)
        return final_path

    @abstractmethod
    def make_decision(self, model: Model, state: State, cur_sfc_index: int, cur_vnf_index: int):
        '''
        make deploy decisions
        :param model: the model
        :param state: current state
        :param cur_sfc_index: cur index of sfc
        :param cur_vnf_index: cur index of vnf
        :return: if success, return the decision list, else return False
        '''
        raise VirtualException()

    @abstractmethod
    def select_path(self, paths: List[Path]):
        '''
        select path from paths
        :param paths: path list
        :return: if success, return the path selected, else return False
        '''
        raise VirtualException()


class RandomDecisionMaker(DecisionMaker):
    '''
    The class used to make random decision
    '''

    def __init__(self):
        super(RandomDecisionMaker, self).__init__()

    def make_decision(self, model: Model, state: State, cur_sfc_index: int, cur_vnf_index: int):
        # First, narrow the available nodes set
        available_node_set = self.narrow_node_set(model, cur_sfc_index, cur_vnf_index)
        if len(available_node_set) == 0:
            return False
        else:
            return random.sample(available_node_set, 1)

    def select_path(self, paths: List[Path]):
        '''
        select path from paths
        :param paths: path list
        :return: path selected or false
        '''
        if len(paths) == 0:
            return False
        else:
            return random.sample(paths, 1)[0]


# test
def main():
    # import generator
    # topo = generator.generate_topology(30)
    # for path in nx.all_simple_paths(topo, 15, 16):
    #     print(path)
    # nx.draw(topo, with_labels=True)
    # plt.show()
    print(random.sample([1], 1))


if __name__ == '__main__':
    main()
