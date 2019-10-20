import matplotlib.pyplot as plt
import random
import warnings
import matplotlib.cbook
import math

from sfcbased.model import *
import sfcbased.sampler as sampler

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
fig, ax = plt.subplots()
fig.set_tight_layout(False)


def generate_action_space(size: int):
    """
    Generate action space which contains all actions
    :param size: space size
    :return: action space
    """
    action_space = []
    for i in range(size):
        for j in range(size):
            action_space.append([i, j])
    return action_space


def generate_topology(size: int = 100):
    """
    Function used to generate topology.
    Mainly with three resources: computing resources, bandwidth resources and latency resources.
    Make sure the whole network is connected
    Notices:
    1. active: the resources occupied by active instance
    2. reserved: the resources reserved by stand-by instance
    3. max_sbsfc_index: the index of stand-by sfc which has largest reservation, only for MaxReservation
    4. sbsfcs: the stand=by sfc deployed on this server(not started)
    :param size: node number
    :return: topology
    """
    topo = nx.Graph()

    # generate V
    for i in range(size):
        computing_resource = random.randint(10000, 20000)
        topo.add_node(i, computing_resource=computing_resource, active=0, reserved=0, max_sbsfc_index=-1, sbsfcs=set())

    # generate E
    for i in range(size):
        for j in range(i + 1, size):
            # make sure the whole network is connected
            if j == i + 1:
                bandwidth = random.randint(1000, 10000)
                topo.add_edge(i, j, bandwidth=bandwidth, active=0, reserved=0, latency=random.uniform(2, 5), max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
                continue
            if random.randint(1, 5) == 1:
                bandwidth = random.randint(1000, 10000)
                topo.add_edge(i, j, bandwidth=bandwidth, active=0, reserved=0, latency=random.uniform(2, 5), max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    return topo


def generate_sfc_list(topo: nx.Graph, size: int = 100, duration: int = 100):
    """
    Generate specified number SFCs
    :param topo: network topology(used to determine the start server and the destination server of specified SFC)
    :param size: the total number SFCs
    :param duration: arriving SFCs duration
    :return: SFC list
    """
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
    """
    Function used to generate specified number nodes in network topology and SFCs in SFC list
    :param topo_size: nodes number in network topology
    :param sfc_size: SFCs number in SFC list
    :param duration: Duration of model
    :return: Model object
    """
    topo = generate_topology(size=topo_size)
    sfc_list = generate_sfc_list(topo=topo, size=sfc_size, duration=duration)
    return Model(topo, sfc_list)


def generate_failed_instances_time_slot(model: Model, time: int, error_rate: float):
    """
    Random generate failed instances, for:
    1. either active or stand-by instance is running
    2. can't expired in this time slot
    Consider two ways for generating failed instances:
    [×] 1. if the failed instances are decided by server, the instances running on this server will all failed and we can't decide whether our placement is good or not
    [√] 2. if the failed instances are dicided by themselves, then one running active instance failed will make its stand-by instance started, this will occupy the resources
    on the server which this stand-by instance is placed, and influence other stand-by instances, so we use this.
    :param model: model
    :param time: current time
    :param error_rate: error rate
    :return: list of instance
    """
    assert error_rate <= 1

    # get all running instances
    all_running_instances = []
    for i in range(len(model.sfc_list)):
        cur_sfc = model.sfc_list[i]
        if cur_sfc.state == State.Normal and cur_sfc.time + cur_sfc.TTL >= time:
            all_running_instances.append(Instance(i, True))
        if model.sfc_list[i].state == State.Backup and cur_sfc.time + cur_sfc.TTL >= time:
            all_running_instances.append(Instance(i, False))

    # random select
    sample_num = math.ceil(len(all_running_instances) * error_rate)
    failed_instances = random.sample(all_running_instances, sample_num)
    return failed_instances


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
