from sfcbased import *
import pickle
from pylab import show
import platform

pf = platform.system()

topo_size = 15 # topology size
sfc_size = 3000 # number of SFCs
duration = 300 # simulation time
process_capacity = 5 # each time only can process 10 sfcs
topo_file_name = "model\\topo.pkl" if pf == "Windows" else "model/topo.pkl" # file name
model_file_name = "model\\model.pkl" if pf == "Windows" else "model/model.pkl" # file name
jitter = True
test_env = TestEnv.MaxReservation
load_model = False

if __name__ == "__main__":
    topo = generate_topology(size=topo_size)
    # model = generate_model(topo_size=topo_size, sfc_size=sfc_size, duration=duration, process_capacity=process_capacity)
    nx.draw(topo)
    show()
    if load_model:
        with open(model_file_name, 'wb') as f:  # open file with write-mode
            sfc_list = generate_sfc_list(topo, process_capacity, size=sfc_size, duration=duration, jitter=jitter)
            model = Model(topo=topo, sfc_list=sfc_list)
            model_string = pickle.dump(model, f)  # serialize and save object
    else:
        with open(topo_file_name, 'wb') as f:  # open file with write-mode
            topo_string = pickle.dump(topo, f)  # serialize and save object

