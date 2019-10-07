from sfcbased.model import *

def detect_cur_state(model: Model, cur_sfc_index: int, cur_vnf_index: int):
    """
    State vector detector.
    :param model: model
    :param cur_sfc_index: current sfc index
    :param cur_vnf_index: current vnf index
    :return: detected state
    """
    state = []

    # node state
    for node in model.topo.nodes(data=True):
        state.append(node[1]['computing_resource'])

    # edge state
    for edge in model.topo.edges(data=True):
        state.append(edge[2]['bandwidth'])
        state.append(edge[2]['latency'])

    # current processing state
    state.append(model.sfc_list[cur_sfc_index].throughput) # VNF's throughput is sfc's throughput
    state.append(model.sfc_list[cur_sfc_index].latency - model.sfc_list[cur_sfc_index].latency_occupied) # residual throughput
    state.append(len(model.sfc_list[cur_sfc_index]) - cur_vnf_index) # number of undeployed VNFs
    state.append(model.sfc_list[cur_sfc_index][cur_vnf_index].computing_resource) # computing resource demand
    state.append(model.sfc_list[cur_sfc_index].TTL) # sfc's TTL

    return state

def main():
    import generator
    topo = generator.generate_topology(30)
    for node in topo.nodes(data=True):
        print(node)
    # nx.draw(topo, with_labels=True)
    # plt.show()
    # print(random.sample([1], 1))

if __name__ == '__main__':
    main()

