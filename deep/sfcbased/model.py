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

@unique
class State(Enum):
    Undeployed = 0
    Failed = 1
    Normal = 2
    Backup = 3
    Broken = 4


@unique
class VariableState(Enum):
    Uninitialized = 0


@unique
class TestEnv(Enum):
    NoBackup = 0
    Aggressive = 1
    Normal = 2
    MaxReservation = 3
    FullyReservation = 4


@unique
class SFCType(Enum):
    Active = 0
    Standby = 1

class Decision(BaseObject):
    '''
    This class is denoted as a decision
    '''

    def __init__(self, active_server: int, standby_server: int):
        '''
        Initialization
        :param active_server: server index of active instance
        :param standby_server: server index of standby instance, if not backup, then -1
        '''
        self.active_server = active_server
        self.standby_server = standby_server
        self.active_path_s2c = VariableState.Uninitialized
        self.standby_path_s2c = VariableState.Uninitialized
        self.active_path_c2d = VariableState.Uninitialized
        self.standby_path_c2d = VariableState.Uninitialized
        self.update_path = VariableState.Uninitialized

    def set_active_path_s2c(self, path: List):
        self.active_path_s2c = path

    def set_standby_path_s2c(self, path: List):
        self.standby_path_s2c = path

    def set_active_path_c2d(self, path: List):
        self.active_path_c2d = path

    def set_standby_path_c2d(self, path: List):
        self.standby_path_c2d = path

    def set_update_path(self, path: List):
        self.update_path = path


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
    """
    Designed for Monitoring the actions of whole system
    """
    action_list = []
    format_logs = []

    @classmethod
    def state_transition(cls, sfc_index: int, pre_state: State, new_state: State):
        """
        Handle the state transition condition
        :param pre_state: previous state
        :param new_state: new state
        :return: nothing
        """
        cls.log("")
        cls.log("The state of SFC {} changes from {} to {}".format(sfc_index, pre_state, new_state))
        cls.format_log([sfc_index, pre_state, new_state])

    @classmethod
    def log(cls, content: str):
        cls.action_list.append(content)

    @classmethod
    def format_log(cls, content: List):
        cls.format_logs.append(content)

    @classmethod
    def change_failed(cls, sfc_index: int):
        cls.log("")
        cls.log("The state of SFC {} changes to Failed".format(sfc_index))

    @classmethod
    def change_normal(cls, sfc_index: int):
        cls.log("The state of SFC {} changes to Normal".format(sfc_index))

    @classmethod
    def change_backup(cls, sfc_index: int):
        cls.log("The state of SFC {} changes to Backup".format(sfc_index))

    @classmethod
    def change_broken(cls, sfc_index: int):
        cls.log("")
        cls.log("The state of SFC {} changes to Broken".format(sfc_index))

    @classmethod
    def deploy_server(cls, sfc_index: int, server_id: int):
        cls.log("SFC {} deploy on server {}".format(sfc_index, server_id))

    @classmethod
    def path_occupied(cls, path: Path, sfc_index: int):
        cls.log("Path {} is occupied by SFC {}".format(path, sfc_index))

    @classmethod
    def active_computing_resource_change(cls, server_id: int, before: int, after: int):
        cls.log("The active computing resource of server {} from {} changes to {}".format(server_id, before, after))

    @classmethod
    def active_bandwidth_change(cls, start: int, destination: int, before: int, after: int):
        cls.log("The active bandwidth of link from {} to {} changes from {} to {}".format(start, destination, before,
                                                                                          after))

    @classmethod
    def reserved_computing_resource_change(cls, server_id: int, before: int, after: int):
        cls.log("The reserved computing resource of server {} from {} changes to {}".format(server_id, before, after))

    @classmethod
    def reserved_bandwidth_change(cls, start: int, destination: int, before: int, after: int):
        cls.log("The reserved bandwidth of link from {} to {} changes from {} to {}".format(start, destination, before,
                                                                                            after))

    @classmethod
    def print_log(cls):
        for item in cls.action_list:
            print(item)


class Instance(BaseObject):
    """
    This class is denoted as an instance.
    """
    def __init__(self, sfc_index: int, is_active: bool):
        self.sfc_index = sfc_index
        self.is_active = is_active


class ACSFC(BaseObject):
    """
    This class is denoted as an active SFC.
    """

    def __init__(self):
        self.server = VariableState.Uninitialized
        self.downtime = VariableState.Uninitialized
        self.path_s2c = VariableState.Uninitialized
        self.path_c2d = VariableState.Uninitialized


class SBSFC(BaseObject):
    '''
    This class is denoted as a stand-by SFC.
    '''

    def __init__(self):
        self.server = VariableState.Uninitialized
        self.starttime = VariableState.Uninitialized
        self.downtime = VariableState.Uninitialized
        self.path_s2c = VariableState.Uninitialized
        self.path_c2d = VariableState.Uninitialized


class SFC(BaseObject):
    """
    This class is denoted as a SFC
    """

    def __init__(self, computing_resource: int, tp: int, latency: float, update_tp: int, process_latency: float, s: int,
                 d: int, time: float, TTL: int):
        """
        SFC initialization
        :param computing_resource: computing_resource required
        :param tp: totally throughput required
        :param latency: totally latency required
        :param update_tp: update throughput required
        :param process_latency: latency of processing
        :param s: start server
        :param d: destination server
        :param time: arriving time
        :param TTL: time to live
        """
        self.computing_resource = computing_resource
        self.tp = tp
        self.latency = latency
        self.update_tp = update_tp
        self.process_latency = process_latency
        self.s = s
        self.d = d
        self.time = time
        self.TTL = TTL

        self.state = State.Undeployed
        self.update_path = VariableState.Uninitialized
        self.active_sfc = ACSFC()
        self.standby_sfc = SBSFC()

    def __str__(self):
        """
        Display in console with specified format.
        :return: display string
        """
        return "(computing_resource: {}, throughput: {}, latency: {}, update throughput: {}, process latency: {}, from {}->{}, time: {}, TTL: {})".format(
            self.computing_resource,
            self.tp,
            self.latency, self.update_tp,
            self.process_latency,
            self.s, self.d, self.time, self.TTL)

    def set_state(self, sfc_index: int, new_state: State):
        """
        Setting up new state
        :param new_state: new state
        :return: nothing
        """
        Monitor.state_transition(sfc_index, self.state, new_state)
        self.state = new_state


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

class DecisionMaker(BaseObject):
    '''
    The class used to make deploy decision
    '''

    def __init__(self):
        super(DecisionMaker, self).__init__()

    def is_path_throughput_met(self, model: Model, path: List, throughput: int, cur_sfc_type: SFCType, test_env: TestEnv):
        """
        Determine if the throughput requirement of the given path is meet based on current sfc type
        :param model: given model
        :param path: given path
        :param throughput: given throughput requirement
        :param cur_sfc_type: current sfc type
        :return: true or false
        """
        if cur_sfc_type == SFCType.Active:
            for i in range(len(path) - 1):
                if model.topo[path[i]][path[i + 1]]["bandwidth"] - model.topo[path[i]][path[i + 1]]["reserved"] - \
                        model.topo[path[i]][path[i + 1]]["active"] < throughput:
                    return False
            return True
        else:
            for i in range(len(path) - 1):
                if test_env == TestEnv.Aggressive:
                    if model.topo[path[i]][path[i + 1]]["bandwidth"] < throughput:
                        return False
                if test_env == TestEnv.Normal or test_env == TestEnv.MaxReservation:
                    if model.topo[path[i]][path[i + 1]]["bandwidth"] - model.topo[path[i]][path[i + 1]][
                        "active"] < throughput:
                        return False
                if test_env == TestEnv.FullyReservation:
                    if model.topo[path[i]][path[i + 1]]["bandwidth"] - model.topo[path[i]][path[i + 1]]["reserved"] - \
                            model.topo[path[i]][path[i + 1]]["active"] < throughput:
                        return False
            return True

    def is_path_latency_met(self, model: Model, path_s2c: List, path_c2d: List, latency: float):
        """
        Determine if the latency requirement of the given path is meet
        :param model: given model
        :param path_s2c: given path from start server to current server
        :param path_c2d: given path from current server to destination server
        :param latency: given latency
        :return: true or false
        """
        path_latency = 0
        for i in range(len(path_s2c) - 1):
            path_latency += model.topo[path_s2c[i]][path_s2c[i + 1]]["latency"]  # calculate latency of path
        for i in range(len(path_c2d) - 1):
            path_latency += model.topo[path_c2d[i]][path_c2d[i + 1]]["latency"]  # calculate latency of path
        if path_latency <= latency:
            return True
        return False

    def verify_active(self, model: Model, cur_sfc_index: int, cur_server_index: int, test_env: TestEnv):
        """
        Verify if current active sfc can be put on current server based on following two principles
        1. if the remaining computing resource is still enough for this sfc
        2. if available paths still exist
        Both these two principles are met can return true, else false
        :param model: model
        :param cur_sfc_index: current sfc index
        :param cur_server_index: current server index
        :return: true or false
        """

        # principle 1
        if model.topo.nodes[cur_server_index]["computing_resource"] - model.topo.nodes[cur_server_index]["active"] - \
                model.topo.nodes[cur_server_index]["reserved"] < model.sfc_list[cur_sfc_index].computing_resource:
            return False

        # principle 2
        for path_s2c in nx.all_simple_paths(model.topo, model.sfc_list[cur_sfc_index].s, cur_server_index):
            for path_c2d in nx.all_simple_paths(model.topo, cur_server_index, model.sfc_list[cur_sfc_index].d):
                remain_latency = model.sfc_list[cur_sfc_index].latency - model.sfc_list[cur_sfc_index].process_latency
                if self.is_path_latency_met(model, path_s2c, path_c2d, remain_latency) and self.is_path_throughput_met(model,path_s2c,model.sfc_list[cur_sfc_index].tp,SFCType.Active,test_env) and self.is_path_throughput_met(model,path_c2d,model.sfc_list[cur_sfc_index].tp,SFCType.Active,test_env) :
                    return True

    def verify_standby(self, model: Model, cur_sfc_index: int, active_server_index: int, cur_server_index: int, test_env: TestEnv):
        """
        Verify if current stand-by sfc can be put on current server based on following three principles
        1. if the remaining computing resource is still enough for this sfc
        2. if available paths still exist
        3. if available paths for updating still exist
        Both these three principles are met can return true, else false
        When the active instance is deployed, the topology will change and some constraints may not be met, but this is just a really small case so that we don't have to consider it.
        :param model: model
        :param cur_sfc_index: current sfc index
        :param active_server_index: active server index
        :param cur_server_index: current server index
        :return: true or false
        """
        assert test_env != TestEnv.NoBackup

        # principle 1
        if test_env == TestEnv.Aggressive:
            if model.topo.nodes[cur_server_index]["computing_resource"] < model.sfc_list[
                cur_sfc_index].computing_resource:
                return False
        if test_env == TestEnv.Normal or test_env == TestEnv.MaxReservation:
            if model.topo.nodes[cur_server_index]["computing_resource"] - model.topo.nodes[cur_server_index]["active"] < \
                    model.sfc_list[cur_sfc_index].computing_resource:
                return False
        if test_env == TestEnv.FullyReservation:
            if model.topo.nodes[cur_server_index]["computing_resource"] - model.topo.nodes[cur_server_index]["active"] - \
                    model.topo.nodes[cur_server_index]["reserved"] < model.sfc_list[cur_sfc_index].computing_resource:
                return False

        # principle 2
        principle2 = False
        for path_s2c in nx.all_simple_paths(model.topo, model.sfc_list[cur_sfc_index].s, cur_server_index):
            for path_c2d in nx.all_simple_paths(model.topo, cur_server_index, model.sfc_list[cur_sfc_index].d):
                if self.is_path_latency_met(model, path_s2c, path_c2d,
                                            model.sfc_list[cur_sfc_index].latency - model.sfc_list[
                                                cur_sfc_index].process_latency) and self.is_path_throughput_met(model,
                                                                                                                path_s2c,
                                                                                                                model.sfc_list[
                                                                                                                    cur_sfc_index].tp,
                                                                                                                SFCType.Standby, test_env) and self.is_path_throughput_met(model,
                                                                                                                path_c2d,
                                                                                                                model.sfc_list[
                                                                                                                    cur_sfc_index].tp,
                                                                                                                SFCType.Standby, test_env):
                    principle2 = True
        if not principle2:
            return False

        # principle 3
        for path in nx.all_simple_paths(model.topo, active_server_index, cur_server_index):
            if self.is_path_throughput_met(model, path, model.sfc_list[cur_sfc_index].update_tp, SFCType.Active, test_env):
                return True
        return False

    def narrow_decision_set(self, model: Model, cur_sfc_index: int, test_env: TestEnv):
        '''
        Used to narrow available decision set
        :param model: model
        :param cur_sfc_index: cur processing sfc index
        :return: decision sets
        '''
        desision_set = set()
        for i in range(len(model.topo.nodes)):
            if not self.verify_active(model, cur_sfc_index, i, test_env):
                continue
            if test_env == TestEnv.NoBackup:
                desision_set.add(Decision(i, -1))
                continue
            for j in range(len(model.topo.nodes)):
                if self.verify_standby(model, cur_sfc_index, i, j, test_env):
                    desision_set.add(Decision(i, j))
        return desision_set

    def make_decision(self, model: Model, cur_sfc_index: int, test_env: TestEnv):
        '''
        make deploy decisions
        :param model: the model
        :param state: current state
        :param cur_sfc_index: cur index of sfc
        :param cur_vnf_index: cur index of vnf
        :return: if success, return the decision list, else return False
        '''
        decisions = self.narrow_decision_set(model, cur_sfc_index, test_env)
        if len(decisions) == 0:
            return False
        else:
            decision = self.select_decision_from_decisions(decisions)
            paths = self.select_paths(model, cur_sfc_index, decision.active_server, decision.standby_server, test_env)
            if test_env != TestEnv.NoBackup:
                decision.set_active_path_s2c(paths[0][0])
                decision.set_active_path_c2d(paths[0][1])
                decision.set_standby_path_s2c(paths[1][0])
                decision.set_standby_path_c2d(paths[1][1])
                decision.set_update_path(paths[2])
            else:
                decision.set_active_path_s2c(paths[0])
                decision.set_active_path_c2d(paths[1])
            return decision

    def select_decision_from_decisions(self, decisions: set):
        decision = random.sample(decisions, 1)[0]
        return decision

    def select_path(self, path_set: List, coupled: bool):
        '''
        select path from paths
        :param paths: path list
        :return: if success, return the path selected, else return False
        '''
        assert len(path_set) is not 0

        if not coupled:
            min = float("inf")
            min_path = []
            for path in path_set:
                length = len(path)
                if length < min:
                    min = length
                    min_path = path
            return min_path
        else:
            min = float("inf")
            min_path = []
            for path_item in path_set:
                length = len(path_item[0]) + len(path_item[1])
                if length < min:
                    min = length
                    min_path = path_item
            return min_path

    def select_paths(self, model: Model, sfc_index: int, active_index: int, standby_index: int, test_env: TestEnv):
        '''
        select paths for determined active instance server index and stand-by instance server index
        :param model: model
        :param sfc_index: sfc index
        :param active_index: active server index
        :param standby_index: stand-by server index
        :param test_env: test environment
        :return: select path
        '''
        if test_env == TestEnv.NoBackup:
            active_paths = []
            for active_s2c in nx.all_simple_paths(model.topo, model.sfc_list[sfc_index].s, active_index):
                for active_c2d in nx.all_simple_paths(model.topo, active_index, model.sfc_list[sfc_index].d):
                    if self.is_path_latency_met(model, active_s2c, active_c2d,
                                                model.sfc_list[sfc_index].latency - model.sfc_list[
                                                    sfc_index].process_latency) and self.is_path_throughput_met(model,
                                                                                                                active_s2c,
                                                                                                                model.sfc_list[
                                                                                                                    sfc_index].tp,
                                                                                                                SFCType.Active,
                                                                                                                test_env) and self.is_path_throughput_met(
                        model,
                        active_c2d,
                        model.sfc_list[
                            sfc_index].tp,
                        SFCType.Active, test_env):
                        active_paths.append([active_s2c, active_c2d])
            assert len(active_paths) is not 0
            active_path = self.select_path(active_paths, True)
            return active_path


        # calculate paths for active instance
        active_paths = []
        for active_s2c in nx.all_simple_paths(model.topo, model.sfc_list[sfc_index].s, active_index):
            for active_c2d in nx.all_simple_paths(model.topo, active_index, model.sfc_list[sfc_index].d):
                if self.is_path_latency_met(model, active_s2c, active_c2d,
                                            model.sfc_list[sfc_index].latency - model.sfc_list[
                                                sfc_index].process_latency) and self.is_path_throughput_met(model,
                                                                                                                active_s2c,
                                                                                                                model.sfc_list[
                                                                                                                    sfc_index].tp,
                                                                                                                SFCType.Active, test_env) and self.is_path_throughput_met(model,
                                                                                                                active_c2d,
                                                                                                                model.sfc_list[
                                                                                                                    sfc_index].tp,
                                                                                                                SFCType.Active, test_env):
                    active_paths.append([active_s2c, active_c2d])
        assert len(active_paths) is not 0

        # calculate paths for stand-by instance
        standby_paths = []
        for standby_s2c in nx.all_simple_paths(model.topo, model.sfc_list[sfc_index].s, standby_index):
            for standby_c2d in nx.all_simple_paths(model.topo, standby_index, model.sfc_list[sfc_index].d):
                if self.is_path_latency_met(model, standby_s2c, standby_c2d,
                                            model.sfc_list[sfc_index].latency - model.sfc_list[
                                                sfc_index].process_latency) and self.is_path_throughput_met(model,
                                                                                                                standby_s2c,
                                                                                                                model.sfc_list[
                                                                                                                    sfc_index].tp,
                                                                                                                SFCType.Standby, test_env) and self.is_path_throughput_met(model,
                                                                                                                standby_c2d,
                                                                                                                model.sfc_list[
                                                                                                                    sfc_index].tp,
                                                                                                                SFCType.Standby, test_env):
                    standby_paths.append([standby_s2c, standby_c2d])
        assert len(standby_paths) is not 0

        # calculate paths for updating
        update_paths = []
        for path in nx.all_simple_paths(model.topo, active_index, standby_index):
            if self.is_path_throughput_met(model, path, model.sfc_list[sfc_index].update_tp, SFCType.Active, test_env):
                update_paths.append(path)
        assert len(update_paths) is not 0

        # select path
        active_path = self.select_path(active_paths, True)
        standby_path = self.select_path(standby_paths, True)
        update_path = self.select_path(update_paths, False)

        return [active_path, standby_path, update_path]


class RandomDecisionMaker(DecisionMaker):
    '''
    The class used to make random decision
    '''

    def __init__(self):
        super(RandomDecisionMaker, self).__init__()

    def select_decision_from_decisions(self, decisions: set):
        decision = random.sample(decisions, 1)[0]
        return decision














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
