from sfcbased import *

# meta-parameters
topo_size = 10 # topology size
sfc_size = 40 # number of SFCs
duration = 100 # simulation time
error_rate = 0.1
test_env = TestEnv.Normal

model = generate_model(topo_size=topo_size, sfc_size=sfc_size, duration=duration)

decision_maker = RandomDecisionMaker()

# nx.draw(model.topo, with_labels=True)
# plt.show()

for cur_time in range(0, duration):
    failed_instances = generate_failed_instances_time_slot(model, error_rate)
    process_time_slot(model, decision_maker, cur_time, test_env, failed_instances)

Monitor.print_log()

print("\nDone!")

