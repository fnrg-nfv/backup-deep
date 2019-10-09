from sfcbased import *
from tqdm import tqdm

# meta-parameters
topo_size = 10 # topology size
sfc_size = 100 # number of SFCs
duration = 100 # simulation time
error_rate = 0.1
test_env = TestEnv.FullyReservation

model = generate_model(topo_size=topo_size, sfc_size=sfc_size, duration=duration)

decision_maker = RandomDecisionMaker()

# nx.draw(model.topo, with_labels=True)
# plt.show()

for cur_time in tqdm(range(0, duration)):
    failed_instances = generate_failed_instances_time_slot(model, cur_time, error_rate)
    process_time_slot(model, decision_maker, cur_time, test_env, failed_instances)

Monitor.print_log()

# model.print_start_and_down()

print(model.calculate_fail_rate())

print("\nDone!")


