import heapq

from sfcbased.model import *


class Result(object):
    def __init__(self, y_r: List[int], x_rs: List[List[int]]):
        self.y_r = y_r
        self.x_rx = x_rs


class Configuration(object):
    def __init__(self, sfc: SFC, route: List[int], place: List[int], latency: int):
        # todo
        self.sfc = sfc
        self.route = route
        self.place = place
        self.latency = latency
        # self.throughput = throughput
        # self.computing_resource = computing_resource


def dijkstra(topo: nx.Graph, s: int) -> {}:  # todo
    """
    Compute single source shortest path with dijkstra algorithm
    :param topo: network topology
    :param s: source node
    :return: shortest path
    """
    ret = {}
    heap = [(0, s)]
    while heap:
        distance, node = heapq.heappop(heap)
        if node not in ret:
            ret[node] = distance

        adj_nodes = topo[node]
        for node in adj_nodes:
            if node not in ret:
                new_distance = adj_nodes[node]['latency'] + distance
                heapq.heappush(heap, (new_distance, node))

    print("Dijkstra: {}".format(ret))
    return ret


def generate_route_list(topo: nx.Graph, sfc: SFC):
    """
    Find all paths meet requirement
    :param topo: network topology
    :param sfc: SFC
    :return: route set
    """
    s = sfc.s
    d = sfc.d
    shortest_distance = dijkstra(topo, d)
    stack = [([s], 0)]
    route_set = []
    while stack:
        route, latency = stack.pop()

        # requirement can't be satisified
        if latency + shortest_distance[route[-1]] > sfc.latency:
            continue

        # final condition: the last node is d
        elif route[-1] == d:
            route_set.append((route, latency))
        else:
            adjs = topo[route[-1]]
            for adj in adjs:
                if adj not in route:
                    new_route = route[:]
                    new_route.append(adj)
                    stack.append((new_route, latency + adjs[adj]['latency']))

    print("Size of path set: %d" % len(route_set))
    return route_set


def generate_configuration(topo: nx.Graph, sfc: SFC) -> List[Configuration]:
    route_list = generate_route_list(topo, sfc)
    vnf_latency = 0 # process latency
    for vnf in sfc.vnf_list:
        vnf_latency += vnf.latency

    configuration_set = []
    n_vnf = len(sfc.vnf_list) # number of VNF
    for route, latency in route_list:
        latency += vnf_latency

        # transimition delay + proccess delay don't meet requirement
        if latency > sfc.latency:
            continue

        queue = [[0]]

        n_node = len(route) # number of nodes

        while queue:
            config = queue.pop()
            if len(config) == n_vnf + 1:
                config.pop(0)
                configuration_set.append(Configuration(sfc, route, config, latency))
                pass
            else:
                for i in range(config[-1], n_node):
                    add = config[:]
                    add.append(i)
                    queue.append(add)

    print("Size of configuration set: %d" % len(configuration_set))

    return configuration_set
