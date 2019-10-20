import torch.nn as nn
from sfcbased.utils import *
from sfcbased.model import *


@unique
class Space(Enum):
    Unlimit = 0


class DQN(nn.Module):

    def __init__(self, state_shape: int, action_space: List):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_shape, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, len(action_space))

        self.bn_input = nn.BatchNorm1d(state_shape)
        self.bn_hidden_1 = nn.BatchNorm1d(10)
        self.bn_hidden_2 = nn.BatchNorm1d(10)
        self.bn_output = nn.BatchNorm1d(len(action_space))
        self.Tanh = nn.Tanh()
        self.init_weights(3e-2)

    def init_weights(self, init_w):
        self.layer1.weight.data = fanin_init(self.layer1.weight.data.size())
        self.layer2.weight.data = fanin_init(self.layer2.weight.data.size())
        self.layer3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = self.bn_input(x)

        x = self.layer1(x)
        x = self.bn_hidden_1(x)
        x = self.Tanh(x)

        x = self.layer2(x)
        x = self.bn_hidden_2(x)
        x = self.Tanh(x)

        x = self.layer3(x)
        x = self.bn_output(x)
        x = self.Tanh(x)

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
            state_v = torch.tensor(state_a, dtype = torch.float).to(self.device)  # transfer to tensor class
            self.net.eval()
            q_vals_v = self.net(state_v)  # input to network, and get output
            _, act_v = torch.max(q_vals_v, dim=1)  # get the max index
            action_index = int(act_v.item())  # returns the value of this tensor as a standard Python number. This only works for tensors with one element.
        action = self.action_space[action_index]
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

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


class DQNEnvironment(Environment):
    def __init__(self):
        super().__init__()

    def get_reward(self, model: Model, sfc_index: int, decision: Decision, test_env: TestEnv):
        if not decision:
            return 0
        alpha = 0.1
        beta = 0.1
        gamma = 0.1
        return 0

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
        # node state
        for node in model.topo.nodes(data=True):
            state.append(node[1]['computing_resource'])
            state.append(node[1]['active'])
            state.append(node[1]['reserved'])

        # edge state
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
