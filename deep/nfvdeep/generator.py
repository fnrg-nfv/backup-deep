import matplotlib.pyplot as plt
import random
import warnings
import matplotlib.cbook

from model import *
import sampler

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
fig, ax = plt.subplots()
fig.set_tight_layout(False)

def generate_topology(size=100):
    '''
    Function used to generate topology.
    Mainly with three resources: computing resources, bandwidth resources and latency resources.
    :param size: node number
    :return: topology
    '''
    topo = nx.Graph()
    # generate V
    for i in range(size):
        topo.add_node(i, computing_resource=random.randint(4000, 8000))
    # generate E
    for i in range(size):
        for j in range(i + 1, size):
            if random.randint(1, 5) == 1:
                topo.add_edge(i, j, bandwidth=random.randint(1000, 10000), latency=random.uniform(2, 5))
    return topo

# random generate 100 service function chains
# number of vnf: 5~10
# vnf computing resource: 500~1000
# sfc latency demand: 10~30 ms
# sfc throughput demand: 32~128 Mbps todo
def generate_sfc_list(topo: nx.Graph, size: int = 100, duration: int = 100):
    '''
    Generate specified number SFCs
    :param topo: network topology
    :param size: the total number SFCs
    :param duration: arriving SFCs duration
    :return: SFC list
    '''
    sfc_list = []
    nodes_len = len(topo.nodes)
    timeshot_list = sampler.uniform(0, duration, size)

    for i in range(size):
        n = random.randint(5, 10) # the number of VNF
        TTL = random.randint(5, 10) # sfc's time to live
        vnf_list = []
        for j in range(n):
            # TODO: the latency of VNF could be larger, and the computing_resource is very important
            vnf_list.append(VNF(latency=random.uniform(0.045, 0.3), computing_resource=random.randint(500, 1000)))
        s = random.randint(1, nodes_len - 1)
        d = random.randint(1, nodes_len - 1)
        # TODO: the throughput requirement is very important
        sfc_list.append(SFC(vnf_list, latency=random.randint(10, 30), throughput=random.randint(32, 128), s=s, d=d, time=timeshot_list[i], TTL=TTL))
    return sfc_list

def generate_model(topo_size: int = 100, sfc_size: int = 100, duration: int = 100):
    '''
    Function used to generate specified number nodes in network topology and SFCs in SFC list
    :param topo_size: nodes number in network topology
    :param sfc_size: SFCs number in SFC list
    :return: Model object
    '''
    topo = generate_topology(size=topo_size)
    sfc_list = generate_sfc_list(topo=topo, size=sfc_size, duration=duration)
    return Model(topo, sfc_list)

# test
def __main():
    topo = generate_topology()
    print("Num of edges: ", len(topo.edges))
    print("Edges: ", topo.edges.data())
    print("Nodes: ", topo.nodes.data())
    print(topo[0])
    nx.draw(topo, with_labels=True)
    plt.show()

if __name__ == '__main__':
    __main()
