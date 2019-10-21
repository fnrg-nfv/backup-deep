from tqdm import tqdm
from generate_topo import *

with open(file_name, 'rb') as f:
    model = pickle.load(f)   # read file and build object

decision_maker = RandomDecisionMaker()

# nx.draw(model.topo, with_labels=True)
# plt.show()

env = NormalEnvironment()

for cur_time in tqdm(range(0, duration)):
    failed_instances = generate_failed_instances_time_slot(model, cur_time, error_rate)
    state = env.get_state(model, 0)
    process_time_slot(model, decision_maker, cur_time, test_env, state, failed_instances)

Monitor.print_log()

# model.print_start_and_down()

print(model.calculate_fail_rate())

print(model.calculate_accept_rate())

print("\nDone!")


