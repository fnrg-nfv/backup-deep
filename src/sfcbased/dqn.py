import torch.nn as nn
from sfcbased.utils import *
from sfcbased.model import *


@unique
class Space(Enum):
    Unlimit = 0


class DQN(nn.Module):
    def __init__(self, state_len: int, action_len: int, tgt: bool, device: torch.device):
        super(DQN, self).__init__()
        self.tgt = tgt
        self.action_len = action_len
        self.device = device
        self.state_len = state_len
        self.LeakyReLU = nn.LeakyReLU()
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.BNs = nn.ModuleList()
        self.num = 50

        self.BNs.append(nn.BatchNorm1d(num_features=self.state_len))
        self.fc1 = nn.Linear(in_features=self.state_len, out_features=self.num)
        self.BNs.append(nn.BatchNorm1d(num_features=self.num))
        self.fc2 = nn.Linear(in_features=self.num, out_features=self.num)
        self.BNs.append(nn.BatchNorm1d(num_features=self.num))
        self.fc3 = nn.Linear(in_features=self.num, out_features=self.num)
        self.fc4 = nn.Linear(in_features=self.num, out_features=self.num)
        self.fc5 = nn.Linear(in_features=self.num, out_features=self.action_len)

        self.init_weights(3e9)

    def init_weights(self, init_w: float):
        for bn in self.BNs:
            bn.weight.data = fanin_init(bn.weight.data.size(), init_w, device=self.device)
            bn.bias.data = fanin_init(bn.bias.data.size(), init_w, device=self.device)
            bn.running_mean.data = fanin_init(bn.running_mean.data.size(), init_w, device=self.device)
            bn.running_var.data = fanin_init(bn.running_var.data.size(), init_w, device=self.device)

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size(), init_w, device=self.device)
        self.fc1.bias.data = fanin_init(self.fc1.bias.data.size(), init_w, device=self.device)

        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size(), init_w, device=self.device)
        self.fc2.bias.data = fanin_init(self.fc2.bias.data.size(), init_w, device=self.device)

        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size(), init_w, device=self.device)
        self.fc3.bias.data = fanin_init(self.fc3.bias.data.size(), init_w, device=self.device)

    def forward(self, x: torch.Tensor):
        # x = self.BNs[0](x)
        x = self.fc1(x)
        x = self.LeakyReLU(x)

        # x = self.BNs[1](x)
        x = self.fc2(x)
        x = self.LeakyReLU(x)

        # x = self.BNs[2](x)
        x = self.fc3(x)
        x = self.LeakyReLU(x)
        # print("output: ", x)

        x = self.fc4(x)
        x = self.LeakyReLU(x)

        x = self.fc5(x)
        return x


class DQNDecisionMaker(DecisionMaker):
    """
    This class is denoted as a decision maker used reinforcement learning
    """

    def narrow_action_index_set(self, model: Model, cur_sfc_index: int, test_env: TestEnv):
        """
        Used to narrow available decision set
        :param test_env: test env
        :param model: model
        :param cur_sfc_index: cur processing sfc index
        :return: action index sets
        """
        action_index_set = []
        for i in range(len(model.topo.nodes)):
            if not self.verify_active(model, cur_sfc_index, i, test_env):
                continue
            if test_env == TestEnv.NoBackup:
                action_index_set.append(i)
                continue
            for j in range(len(model.topo.nodes)):
                if self.verify_standby(model, cur_sfc_index, i, j, test_env):
                    action_index_set.append(i * len(model.topo.nodes) + j)
        return action_index_set

    def __init__(self, net: DQN, tgt_net: DQN, buffer: ExperienceBuffer, action_space: List, gamma: float, epsilon_start: float, epsilon: float, epsilon_final: float, epsilon_decay: float, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.net = net
        self.tgt_net = tgt_net
        self.buffer = buffer
        self.action_space = action_space
        self.epsilon = epsilon
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.gamma = gamma
        self.idx = 0

    def generate_decision(self, model: Model, cur_sfc_index: int, state: List, test_env: TestEnv):
        action_indexs = self.narrow_action_index_set(model, cur_sfc_index, test_env)
        if len(action_indexs) != 0:
            action_indexs = torch.tensor(action_indexs, device=self.device)
        if self.net.tgt:
            state_a = np.array([state], copy=False)  # make state vector become a state matrix
            state_v = torch.tensor(state_a, dtype=torch.float, device=self.device)  # transfer to tensor class
            self.net.eval()
            q_vals_v = self.net(state_v)  # input to network, and get output
            q_vals_v = torch.index_select(q_vals_v, dim=1, index=action_indexs) if len(action_indexs) != 0 else q_vals_v # select columns
            _, act_v = torch.max(q_vals_v, dim=1)  # get the max index
            action_index = action_indexs[int(act_v.item())] if len(action_indexs) != 0 else act_v.item()
        elif np.random.random() < self.epsilon:
            action = random.randint(0, len(self.action_space) - 1)
            action_index = action
        else:
            state_a = np.array([state], copy=False)  # make state vector become a state matrix
            state_v = torch.tensor(state_a, dtype=torch.float, device=self.device)  # transfer to tensor class
            self.net.eval()
            q_vals_v = self.net(state_v)  # input to network, and get output
            q_vals_v = torch.index_select(q_vals_v, dim=1, index=action_indexs) if len(action_indexs) != 0 else q_vals_v # select columns
            _, act_v = torch.max(q_vals_v, dim=1)  # get the max index
            action_index = action_indexs[int(act_v.item())] if len(action_indexs) != 0 else act_v.item()
        action = self.action_space[action_index]
        # print(action)
        decision = Decision()
        decision.active_server = action[0]
        decision.standby_server = action[1]
        self.epsilon = max(self.epsilon_final, self.epsilon_start - self.idx / self.epsilon_decay)
        self.idx += 1
        return decision


class DQNAction(Action):
    def __init__(self, active: int, standby: int):
        super().__init__()
        self.active = active
        self.standby = standby

    def get_action(self):
        return [self.active, self.standby]

    def action2index(self, action_space: List):
        for i in range(len(action_space)):
            if action_space[i][0] == self.active and action_space[i][1] == self.standby:
                return i
        raise RuntimeError("The action space doesn't contain this action")


def calc_loss(batch, net, tgt_net, gamma: float, action_space: List, double: bool, device: torch.device):
    states, actions, rewards, dones, next_states = batch

    # transform each action to index(real action)
    actions = [DQNAction(action[0], action[1]).action2index(action_space) for action in actions]

    states_v = torch.tensor(states, dtype=torch.float).to(device)
    next_states_v = torch.tensor(next_states, dtype=torch.float).to(device)
    actions_v = torch.tensor(actions, dtype=torch.long).to(device)
    rewards_v = torch.tensor(rewards, dtype=torch.float).to(device)
    done_mask = torch.tensor(dones, dtype=torch.bool).to(device)

    # action is a list with one dimension, we should use unsqueeze() to span it
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    if double:
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]

    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()
    expected_state_action_values = next_state_values * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


class DQNEnvironment(Environment):
    def __init__(self):
        super().__init__()

    def get_reward(self, model: Model, sfc_index: int, decision: Decision, test_env: TestEnv):
        if model.sfc_list[sfc_index].state == State.Failed:
            return -1
        if model.sfc_list[sfc_index].state == State.Normal:
            reward = 1
        # reward -= model.topo.nodes(data=True)[decision.standby_server]["fail_rate"]
        # reward = reward - model.topo.nodes(data=True)[decision.standby_server]["fail_rate"]
        length = len(model.sfc_list[sfc_index].standby_sfc.path_c2d) + len(model.sfc_list[sfc_index].standby_sfc.path_s2c)
        return reward

    def get_state(self, model: Model, sfc_index: int):
        """
        Get the state of current network.
        :param model: model
        :param sfc_indexs: sfc indexs
        :param process_capacity: process capacity
        :return: state vector, done
        """
        state = []

        # first part: topo state
        # 1. node state
        max_v = 0
        for node in model.topo.nodes(data=True):
            if node[1]['computing_resource'] > max_v:
                max_v = node[1]['computing_resource']
        for node in model.topo.nodes(data=True):
            state.append(node[1]['fail_rate'])
            state.append(node[1]['computing_resource'] / max_v)
            state.append(node[1]['active'] / max_v)
            if node[1]['reserved'] == float('-inf'):
                state.append(0)
            else:
                state.append(node[1]['reserved'] / max_v)

        # 2. edge state
        max_e = 0
        for edge in model.topo.edges(data=True):
            if edge[2]['bandwidth'] > max_e:
                max_e = edge[2]['bandwidth']

        for edge in model.topo.edges(data=True):
            state.append(edge[2]['latency'])
            state.append(edge[2]['bandwidth'] / max_e)
            state.append(edge[2]['active'] / max_e)
            if edge[2]['reserved'] == float('-inf'):
                state.append(0)
            else:
                state.append(edge[2]['reserved'] / max_e)

        # the sfcs located in this time slot state
        sfc = model.sfc_list[sfc_index] if sfc_index < len(model.sfc_list) else model.sfc_list[sfc_index - 1]
        state.append(sfc.computing_resource / max_v)
        state.append(sfc.tp / max_e)
        state.append(sfc.latency)
        state.append(sfc.update_tp / max_e)
        state.append(sfc.process_latency)
        state.append(sfc.s)
        state.append(sfc.d)
        return state, False

        #second part
        #current sfc hasn't been deployed
        # if sfc_index == len(model.sfc_list) - 1 or model.sfc_list[sfc_index].state == State.Undeployed:
        #     sfc = model.sfc_list[sfc_index]
        #     state.append(sfc.computing_resource)
        #     state.append(sfc.tp)
        #     state.append(sfc.latency)
        #     state.append(sfc.update_tp)
        #     state.append(sfc.process_latency)
        #     state.append(sfc.s)
        #     state.append(sfc.d)
        #
        # #current sfc has been deployed
        # elif model.sfc_list[sfc_index].state == State.Normal or model.sfc_list[sfc_index].state == State.Failed:
        #     sfc = model.sfc_list[sfc_index + 1]
        #     state.append(sfc.computing_resource)
        #     state.append(sfc.tp)
        #     state.append(sfc.latency)
        #     state.append(sfc.update_tp)
        #     state.append(sfc.process_latency)
        #     state.append(sfc.s)
        #     state.append(sfc.d)
