from tqdm import tqdm
from sfcbased import *
import torch
import torch.optim as optim

# meta-parameters
topo_size = 30 # topology size
sfc_size = 800 # number of SFCs
duration = 100 # simulation time
error_rate = 0.1
test_env = TestEnv.FullyReservation

# parameters with rl
GAMMA = 0.99
BATCH_SIZE = 32

ACTION_SHAPE = 2
REPLAY_SIZE = 10000
EPSILON = 0.0
EPSILON_START = 1.0
EPSILON_FINAL = 0.02
EPSILON_DECAY = 10 ** 5
LEARNING_RATE = 1e-4
SYNC_INTERVAL = 5
ACTION_SPACE = generate_action_space(size=topo_size)
DEVICE = torch.device("cpu")

# create model
model = generate_model(topo_size=topo_size, sfc_size=sfc_size, duration=duration)
STATE_SHAPE = (len(model.topo.nodes()) + len(model.topo.edges())) * 3 + 7

# create decision maker(agent) & optimizer & environment
net = DQN(state_shape=STATE_SHAPE, action_shape=ACTION_SHAPE)
tgt_net = DQN(state_shape=STATE_SHAPE, action_shape=ACTION_SHAPE)
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
                action = DQNAction(decision).get_action()
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
                loss_t = calc_loss(batch, decision_maker.net, tgt_net, gamma=GAMMA, device=DEVICE)
                loss_t.backward()
                optimizer.step()



















if __name__ == "__main__":
    device = torch.device("cpu")
    env = Environment(state_shape=STATE_SHAPE, action_shape=ACTION_SHAPE)
    net = DQN(env.state_shape, env.action_shape)
    tgt_net = DQN(env.state_shape, env.action_shape)
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    best_mean_reward = None
    idx = 0
    while idx <= ITERATION:
        idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - idx / EPSILON_DECAY)

        agent.step(net, epsilon, device=device)

        # buffer is not full, don't learn
        if len(buffer) < REPLAY_SIZE:
            continue

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
