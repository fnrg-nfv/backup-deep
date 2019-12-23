from tqdm import tqdm
import os
from generate_topo import *
from train_dqn import REPLAY_SIZE, EPSILON, EPSILON_START, EPSILON_FINAL, EPSILON_DECAY, GAMMA, STATE_LEN, ACTION_LEN, ACTION_SPACE, TARGET_FILE

# parameters with rl
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# for debuging
action_list = []

if load_model:
    with open(model_file_name, 'rb') as f:
        model = pickle.load(f)  # read file and build object
else:
    with open(topo_file_name, 'rb') as f:
        topo = pickle.load(f)  # read file and build object
        sfc_list = generate_sfc_list(topo=topo, process_capacity=process_capacity, size=sfc_size, duration=duration, jitter=jitter)
        model = Model(topo, sfc_list)
STATE_SHAPE = (len(model.topo.nodes()) + len(model.topo.edges())) * 3 + 7

# create decision maker(agent) & optimizer & environment
# create net and target net
tgt_net = torch.load(TARGET_FILE)
buffer = ExperienceBuffer(capacity=REPLAY_SIZE)

decision_maker = DQNDecisionMaker(net=tgt_net, tgt_net = tgt_net, buffer = buffer, action_space = ACTION_SPACE, epsilon = EPSILON, epsilon_start = EPSILON_START, epsilon_final = EPSILON_FINAL, epsilon_decay = EPSILON_DECAY, device = DEVICE, gamma = GAMMA)

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
        # failed_instances = generate_failed_instances_time_slot(model, cur_time)
        failed_instances = []

        # handle state transition
        state_transition_and_resource_reclaim(model, cur_time, test_env, failed_instances)

        # deploy sfcs / handle each time slot
        for i in range(len(model.sfc_list)):
            # for each sfc which locate in this time slot
            if cur_time <= model.sfc_list[i].time < cur_time + 1:
                idx += 1
                state, _ = env.get_state(model=model, sfc_index=i)
                decision = deploy_sfc_item(model, i, decision_maker, cur_time, state, test_env)
                action = DQNAction(decision.active_server, decision.standby_server).get_action()
                action_list.append(action)

    # Monitor.print_log()
    # model.print_start_and_down()
    plot_action_distribution(action_list, num_nodes=topo_size)

    print("fail rate: ", model.calculate_fail_rate())
    print("real fail rate: ", Monitor.calculate_real_fail_rate())
    print("accept rate: ", model.calculate_accept_rate())

    if pf == "Windows":
        os.system("python C:\\Users\\tristone\\PycharmProjects\\backup-deep\\src\\test_dqn.py")
    elif pf == "Linux":
        os.system("python /root/PycharmProjects/backup-deep/src/test_dqn.py")
