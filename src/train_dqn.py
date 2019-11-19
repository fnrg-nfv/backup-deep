from tqdm import tqdm
import torch.optim as optim
import time
import os
from generate_topo import *

# parameters with rl
SAMPLE_FILE = "model/sample"
TARGET_FILE = "model/target"
EXP_REPLAY_FILE = "model/replay.pkl"
LEARNING_FROM_LAST = True if os.path.exists(TARGET_FILE) and os.path.exists(SAMPLE_FILE) and os.path.exists(EXP_REPLAY_FILE) else False
GAMMA = 0.9
BATCH_SIZE = 200

ACTION_SHAPE = 2
REPLAY_SIZE = 1000
EPSILON = 0.0
EPSILON_START = 1.0
EPSILON_FINAL = 0.05
EPSILON_DECAY = sfc_size
LEARNING_RATE = 1e-3
SYNC_INTERVAL = 5
TRAIN_INTERVAL = 100
ACTION_SPACE = generate_action_space(size=topo_size)
ACTION_LEN = len(ACTION_SPACE)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# create model
with open(file_name, 'rb') as f:
    model = pickle.load(f)  # read file and build object
STATE_LEN = (len(model.topo.nodes()) + len(model.topo.edges())) * 3 + 7

# create decision maker(agent) & optimizer & environment
# create net and target net
if LEARNING_FROM_LAST:
    net = torch.load(SAMPLE_FILE)
    tgt_net = torch.load(TARGET_FILE)
    with open(EXP_REPLAY_FILE, 'rb') as f:
        buffer = pickle.load(f)  # read file and build object
else:
    net = DQN(state_len=STATE_LEN, action_len=ACTION_LEN, device=DEVICE)
    tgt_net = DQN(state_len=STATE_LEN, action_len=ACTION_LEN, device=DEVICE)
    for target_param, param in zip(tgt_net.parameters(), net.parameters()):
        target_param.data.copy_(param.data)
    buffer = ExperienceBuffer(capacity=REPLAY_SIZE)

decision_maker = DQNDecisionMaker(net=net, tgt_net=tgt_net, buffer=buffer, action_space=ACTION_SPACE, epsilon=EPSILON, epsilon_start=EPSILON_START, epsilon_final=EPSILON_FINAL, epsilon_decay=EPSILON_DECAY, device=DEVICE, gamma=GAMMA)

optimizer = optim.Adam(decision_maker.net.parameters(), lr=LEARNING_RATE)
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
                decision = deploy_sfc_item(model, i, decision_maker, cur_time, state, test_env)
                action = DQNAction(decision.active_server, decision.standby_server).get_action()
                reward = env.get_reward(model, i, decision, test_env)
                next_state = env.get_state(model, i)

                exp = Experience(state=state, action=action, reward=reward, new_state=next_state)
                decision_maker.buffer.append(exp)

                if len(decision_maker.buffer) < REPLAY_SIZE:
                    continue

                if idx % SYNC_INTERVAL == 0:
                    decision_maker.tgt_net.load_state_dict(decision_maker.net.state_dict())

                if idx % TRAIN_INTERVAL == 0:
                    optimizer.zero_grad()
                    batch = decision_maker.buffer.sample(BATCH_SIZE)
                    loss_t = calc_loss(batch, decision_maker.net, decision_maker.tgt_net, gamma=GAMMA, action_space=ACTION_SPACE, device=DEVICE)
                    loss_t.backward()
                    optimizer.step()

    torch.save(decision_maker.net, SAMPLE_FILE)
    torch.save(decision_maker.tgt_net, TARGET_FILE)
    with open(EXP_REPLAY_FILE, 'wb') as f:  # open file with write-mode
        model_string = pickle.dump(decision_maker.buffer, f)  # serialize and save object

    Monitor.print_log()
    # model.print_start_and_down()

    print("fail rate: ", model.calculate_fail_rate())
    print("accept rate: ", model.calculate_accept_rate())

    time.sleep(10)
    os.system("python -u C:\\Users\\tristone\\PycharmProjects\\backup-deep\\src\\train_dqn.py")
