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
        computing_resource = random.randint(40000, 80000)
        topo.add_node(i, computing_resource=computing_resource, active=0, reserved=0, max_sbsfc_index=-1, sbsfcs=set())
    # generate E
    for i in range(size):
        for j in range(i + 1, size):
            if random.randint(1, 3) == 1:
                bandwidth = random.randint(1000, 10000)
                topo.add_edge(i, j, bandwidth=bandwidth, active=0, reserved=0, latency=random.uniform(2, 5), max_sbsfc_index=-1, sbsfcs=set())
    return topo


def generate_sfc_list(topo: nx.Graph, size: int = 100, duration: int = 100):
    '''
    Generate specified number SFCs
    :param topo: network topology(used to determine the start server and the destination server of specified SFC)
    :param size: the total number SFCs
    :param duration: arriving SFCs duration
    :return: SFC list
    '''
    sfc_list = []
    nodes_len = len(topo.nodes)

    # list of sample in increasing order
    timeslot_list = sampler.uniform(0, duration, size)

    # generate each sfc
    for i in range(size):
        computing_resource = random.randint(3750, 7500)
        tp = random.randint(32, 128)
        latency = random.randint(10, 30)
        update_tp = tp
        process_latency = random.uniform(0.863, 1.725)
        TTL = random.randint(5, 10)  # sfc's time to live
        s = random.randint(1, nodes_len - 1)
        d = random.randint(1, nodes_len - 1)
        sfc_list.append(SFC(computing_resource, tp, latency, update_tp, process_latency, s, d, timeslot_list[i], TTL))

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
