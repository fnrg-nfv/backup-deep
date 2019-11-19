import torch.nn as nn
from sfcbased.utils import *
from sfcbased.model import *


# import torchsnooper

@unique
class Space(Enum):
    Unlimit = 0


class DQN(nn.Module):
    def __init__(self, state_len: int, action_len: int, device: torch.device):
        super(DQN, self).__init__()
        self.action_len = action_len
        self.device = device
        self.state_len = state_len
        self.LeakyReLU = nn.LeakyReLU()
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.BNs = nn.ModuleList()

        self.BNs.append(nn.BatchNorm1d(num_features=self.state_len))
        self.fc1 = nn.Linear(in_features=self.state_len, out_features=50)
        self.BNs.append(nn.BatchNorm1d(num_features=50))
        self.fc2 = nn.Linear(in_features=50, out_features=50)
        self.BNs.append(nn.BatchNorm1d(num_features=50))
        self.fc3 = nn.Linear(in_features=50, out_features=self.action_len)

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

        # print("output: ", x)

        return x


class DQNDecisionMaker(DecisionMaker):
    """
    This class is denoted as a decision maker used reinforcement learning
    """

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
        if np.random.random() < self.epsilon:
            action = random.randint(0, len(self.action_space) - 1)
            action_index = action
        else:
            state_a = np.array([state], copy=False)  # make state vector become a state matrix
            state_v = torch.tensor(state_a, dtype=torch.float, device=self.device)  # transfer to tensor class
            self.net.eval()
            q_vals_v = self.net(state_v)  # input to network, and get output
            _, act_v = torch.max(q_vals_v, dim=1)  # get the max index
            action_index = int(act_v.item())  # returns the value of this tensor as a standard Python number. This only works for tensors with one element.
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


def calc_loss(batch, net, tgt_net, gamma: float, action_space: List, device: torch.device):
    states, actions, rewards, next_states = batch

    # transform each action to index(real action)
    actions = [DQNAction(action[0], action[1]).action2index(action_space) for action in actions]

    states_v = torch.tensor(states, dtype=torch.float).to(device)
    next_states_v = torch.tensor(next_states, dtype=torch.float).to(device)
    actions_v = torch.tensor(actions, dtype=torch.long).to(device)
    rewards_v = torch.tensor(rewards, dtype=torch.float).to(device)

    # action is a list with one dimension, we should use unsqueeze() to span it
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v

    return nn.MSELoss()(state_action_values, expected_state_action_values)


class DQNEnvironment(Environment):
    def __init__(self):
        super().__init__()

    def get_reward(self, model: Model, sfc_index: int, decision: Decision, test_env: TestEnv):
        if model.sfc_list[sfc_index].state == State.Failed:
            return 0
        if model.sfc_list[sfc_index].state == State.Normal:
            return 1

    def get_state(self, model: Model, sfc_index: int):
        """
        Get the state of current network.
        Contains two parts:
        1. information about topology;
        2. information about sfc.
        Mainly two situations:
        1. current sfc hasn't been deployed, then use this sfc's information;
        2. current sfc has been deployed(either success or failed); then use next sfc's information.
        :param model: model
        :param sfc_index: sfc index
        :return: state vector
        """
        state = []

        # first part
        # 1. node state
        for node in model.topo.nodes(data=True):
            state.append(node[1]['computing_resource'])
            state.append(node[1]['active'])
            state.append(node[1]['reserved'])

        # 2. edge state
        for edge in model.topo.edges(data=True):
            state.append(edge[2]['bandwidth'])
            state.append(edge[2]['active'])
            state.append(edge[2]['reserved'])

        # second part
        # current sfc hasn't been deployed
        if sfc_index == len(model.sfc_list) - 1 or model.sfc_list[sfc_index].state == State.Undeployed:
            sfc = model.sfc_list[sfc_index]
            state.append(sfc.computing_resource)
            state.append(sfc.tp)
            state.append(sfc.latency)
            state.append(sfc.update_tp)
            state.append(sfc.process_latency)
            state.append(sfc.s)
            state.append(sfc.d)

        # current sfc has been deployed
        elif model.sfc_list[sfc_index].state == State.Normal or model.sfc_list[sfc_index].state == State.Failed:
            sfc = model.sfc_list[sfc_index + 1]
            state.append(sfc.computing_resource)
            state.append(sfc.tp)
            state.append(sfc.latency)
            state.append(sfc.update_tp)
            state.append(sfc.process_latency)
            state.append(sfc.s)
            state.append(sfc.d)

        return state
