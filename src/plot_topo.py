from generate_topo import *

if __name__ == "__main__":
    with open(file_name, 'rb') as f:
        topo = pickle.load(f)  # read file and build object

    print("\nservers: ", len(topo.nodes()))
    print("edges:", len(topo.edges()))
    nx.draw(topo)
    show()