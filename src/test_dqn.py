from tqdm import tqdm
import torch.optim as optim
from generate_topo import *

# parameters with rl
GAMMA = 0.9
BATCH_SIZE = 500

ACTION_SHAPE = 2
REPLAY_SIZE = 500
EPSILON = 0.0
EPSILON_START = 1.0
EPSILON_FINAL = 0.02
EPSILON_DECAY = duration
LEARNING_RATE = 1e-3
SYNC_INTERVAL = 5
ACTION_SPACE = generate_action_space(size=topo_size)
DEVICE = torch.device("cuda")

# create model
with open(file_name, 'rb') as f:
    model = pickle.load(f)   # read file and build object
STATE_SHAPE = (len(model.topo.nodes()) + len(model.topo.edges())) * 3 + 7

# create decision maker(agent) & optimizer & environment
net = DQN(model=model, device=DEVICE)
tgt_net = DQN(model=model, device=DEVICE)
buffer = ExperienceBuffer(capacity=REPLAY_SIZE)

decision_maker = DQNDecisionMaker(net=net, tgt_net = tgt_net, buffer = buffer, action_space = ACTION_SPACE, epsilon = EPSILON, epsilon_start = EPSILON_START, epsilon_final = EPSILON_FINAL, epsilon_decay = EPSILON_DECAY, device = DEVICE, gamma = GAMMA)

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

                exp =  Experience(state=state, action=action, reward=reward, new_state=next_state)
                decision_maker.buffer.append(exp)

                if len(decision_maker.buffer) < REPLAY_SIZE:
                    continue

                if idx % SYNC_INTERVAL == 0:
                    decision_maker.tgt_net.load_state_dict(decision_maker.net.state_dict())

                optimizer.zero_grad()
                batch = decision_maker.buffer.sample(BATCH_SIZE)
                loss_t = calc_loss(batch, decision_maker.net, decision_maker.tgt_net, gamma=GAMMA, action_space=ACTION_SPACE, device=DEVICE)
                loss_t.backward()
                optimizer.step()

    Monitor.print_log()

    # model.print_start_and_down()

    print(model.calculate_fail_rate())

    print(model.calculate_accept_rate())
