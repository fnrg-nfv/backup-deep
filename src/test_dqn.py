from tqdm import tqdm
from generate_topo import *
from train_dqn import REPLAY_SIZE, EPSILON, EPSILON_START, EPSILON_FINAL, EPSILON_DECAY, GAMMA, STATE_LEN, ACTION_LEN

# parameters with rl
ACTION_SHAPE = 2
ACTION_SPACE = generate_action_space(size=topo_size)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SAMPLE_FILE = "model/sample"
TARGET_FILE = "model/target"
EXP_REPLAY_FILE = "model/replay.pkl"


# create model
with open(file_name, 'rb') as f:
    model = pickle.load(f)   # read file and build object
STATE_SHAPE = (len(model.topo.nodes()) + len(model.topo.edges())) * 3 + 7

# create decision maker(agent) & optimizer & environment
# create net and target net
net = DQN(state_len=STATE_LEN, action_len=ACTION_LEN, device=DEVICE)
tgt_net = torch.load(TARGET_FILE)
buffer = ExperienceBuffer(capacity=REPLAY_SIZE)

decision_maker = DQNDecisionMaker(net=net, tgt_net = tgt_net, buffer = buffer, action_space = ACTION_SPACE, epsilon = EPSILON, epsilon_start = EPSILON_START, epsilon_final = EPSILON_FINAL, epsilon_decay = EPSILON_DECAY, device = DEVICE, gamma = GAMMA)

env = DQNEnvironment()

# related
action = VariableState.Uninitialized
reward = VariableState.Uninitialized
state = VariableState.Uninitialized
idx = 0

# main function
if __name__ == "__main__":
    for cur_time in tqdm(range(0, duration)):

        # generate failed instances
        failed_instances = generate_failed_instances_time_slot(model, cur_time, error_rate)

        # handle state transition
        state_transition_and_resource_reclaim(model, cur_time, test_env, failed_instances)

        # deploy sfcs / handle each time slot
        for i in range(len(model.sfc_list)):
            # for each sfc which locate in this time slot
            if cur_time <= model.sfc_list[i].time < cur_time + 1:
                idx += 1
                state = env.get_state(model, i)
                _ = deploy_sfc_item(model, i, decision_maker, cur_time, state, test_env)

    Monitor.print_log()
    # model.print_start_and_down()

    print(model.calculate_fail_rate())
    print(model.calculate_accept_rate())

