from sfcbased import *
import pickle

topo_size = 10 # topology size
sfc_size = 10000 # number of SFCs
duration = 3000 # simulation time
file_name = "model.pkl" # file name
error_rate = 0.1
test_env = TestEnv.FullyReservation

if __name__ == "__main__":
    model = generate_model(topo_size=topo_size, sfc_size=sfc_size, duration=duration)
    with open(file_name, 'wb') as f: # open file with write-mode
        model_string = pickle.dump(model, f) # serialize and save object

