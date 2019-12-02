from sfcbased import *
import pickle
from pylab import show
import platform

pf = platform.system()

topo_size = 25 # topology size
sfc_size = 6000 # number of SFCs
duration = 500 # simulation time
process_capacity = 5 # each time only can process 10 sfcs
file_name = "model\\topo.pkl" if pf == "Windows" else "model/topo.pkl" # file name
jitter = True
test_env = TestEnv.MaxReservation

if __name__ == "__main__":
    topo = generate_topology(size=topo_size)
    # model = generate_model(topo_size=topo_size, sfc_size=sfc_size, duration=duration, process_capacity=process_capacity)
    nx.draw(topo)
    show()
    with open(file_name, 'wb') as f: # open file with write-mode
        model_string = pickle.dump(topo, f) # serialize and save object

