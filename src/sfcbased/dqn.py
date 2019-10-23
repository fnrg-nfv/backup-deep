import torch.nn as nn
from sfcbased.utils import *
from sfcbased.model import *
# import torchsnooper

@unique
class Space(Enum):
    Unlimit = 0


class DQN(nn.Module):

    def __init__(self, model: Model, device):
        super(DQN, self).__init__()

        self.device = device

        self.num_server = len(model.topo.nodes)
        self.num_edge = len(model.topo.edges)

        self.edge_num_server = [0 for _ in range(self.num_server)] # number of edges of each server
        self.edge_index_server = [[] for _ in range(self.num_server)] # index of edges occupied by specific server

        # compute the num of edges occupied by each server and the index of them, so that we can build the forward topology easily
        # note: this part is strongly coupled with the get_start() function, please make them compatible to each other
        # 1. add the server
        start = 0
        for i in range(self.num_server):
            self.edge_index_server[i].extend([start, start + 1])
            start += 2

        # 2. add the edge
        start = self.num_server * 2
        for edge in model.topo.edges:
            self.edge_num_server[edge[0]] += 1
            self.edge_num_server[edge[1]] += 1
            self.edge_index_server[edge[0]].extend([start, start + 1])
            self.edge_index_server[edge[1]].extend([start, start + 1])
            start += 2

        # 3. add the sfc's state
        start = self.num_server * 2 + self.num_edge * 2
        for i in range(self.num_server):
            self.edge_index_server[i].extend(range(start, start + 7))

        # create the layers
        # bn
        self.bn_list = nn.ModuleList()
        for i in range(self.num_server):
            layer_bn = nn.BatchNorm1d(len(self.edge_index_server[i]))
            self.bn_list.append(layer_bn)

        # 1. first layer
        self.layer1_list = nn.ModuleList()
        for i in range(self.num_server):
            layer1 = nn.Linear(len(self.edge_index_server[i]), 3)
            self.layer1_list.append(layer1)

        # 2. second layer
        self.layer2_list = nn.ModuleList()
        for i in range(self.num_server):
            layer2 = nn.Linear(3, 2)
            self.layer2_list.append(layer2)

        # 3. third layer
        self.layer3_list = nn.ModuleList()
        for i in range(self.num_server * self.num_server):
            layer3 = nn.Linear(2, 1)
            self.layer3_list.append(layer3)

        self.Tanh = nn.Tanh()
        self.init_weights(3e2)

    def init_weights(self, init_w):
        for layer in self.bn_list:
            layer.weight.data = fanin_init(layer.weight.data.size(), init_w, device=self.device)
            layer.bias.data = fanin_init(layer.bias.data.size(), init_w, device=self.device)
            layer.running_mean.data = fanin_init(layer.running_mean.data.size(), init_w, device=self.device)
            layer.running_var.data = fanin_init(layer.running_var.data.size(), init_w, device=self.device)
        for layer in self.layer1_list:
            layer.weight.data = fanin_init(layer.weight.data.size(), init_w, device=self.device)
            layer.bias.data = fanin_init(layer.bias.data.size(), init_w, device=self.device)
        for layer in self.layer2_list:
            layer.weight.data = fanin_init(layer.weight.data.size(), init_w, device=self.device)
            layer.bias.data = fanin_init(layer.bias.data.size(), init_w, device=self.device)
        for layer in self.layer3_list:
            layer.weight.data = fanin_init(layer.weight.data.size(), init_w, device=self.device)
            layer.bias.data = fanin_init(layer.bias.data.size(), init_w, device=self.device)

    def forward(self, x: torch.Tensor):
        layer1_outputs = []
        for i in range(self.num_server):
            input = x.index_select(1, torch.tensor(data=self.edge_index_server[i], dtype=torch.long, device=self.device))
            layer1_outputs.append(self.Tanh(self.layer1_list[i](self.bn_list[i](input))))

        layer2_outputs = []
        for i in range(self.num_server):
            input = layer1_outputs[i]
            layer2_outputs.append(self.Tanh(self.layer2_list[i](input)))

        layer3_outputs = []
        for i in range(self.num_server * self.num_server):
            active_index = i // self.num_server
            stand_by_index = i % self.num_server
            active_action = layer2_outputs[active_index].index_select(1, torch.tensor(data=[0], dtype=torch.long, device=self.device))
            standby_action = layer2_outputs[stand_by_index].index_select(1, torch.tensor(data=[1], dtype=torch.long, device=self.device))
            input = torch.cat([active_action, standby_action], 1)
            layer3_outputs.append(self.Tanh(self.layer3_list[i](input)))

        output = torch.cat(layer3_outputs, 1)
        return output


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
            state.append(node[1]['active'])
            state.append(node[1]['reserved'])

        # 2. edge state
        for edge in model.topo.edges(data=True):
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
