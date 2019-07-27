from model import *
from detecter import *


def deploy_sfcs_in_timeslot(model: Model, decision_maker: DecisionMaker, time: int, step: int):
    '''
    Deploy the sfcs located in given timeslot
    :param sfc_list: list of SFCs
    :param time: given time
    :param step: given step
    '''
    for i in range(len(model.sfc_list)):
        if model.sfc_list[i].time >= time and model.sfc_list[i].time < time + step:
            assert model.sfc_list[i].state == State.future
            model.sfc_list[i].state = State.deploying
            Monitor.change_deploying(i)
            for j in range(len(model.sfc_list[i].vnf_list)):
                state = detect_cur_state(model, cur_sfc_index=i, cur_vnf_index=j)
                pre_server_id = model.sfc_list[i].s if j == 0 else model.sfc_list[i][j - 1].deploy_decision

                # make decision
                decision = decision_maker.make_decision(model, state, cur_sfc_index=i, cur_vnf_index=j)

                # success
                if decision:
                    deploy_index = decision[0]
                    Monitor.deploy_server(i, j, deploy_index)
                    if pre_server_id != deploy_index:
                        path = decision_maker.select_path(
                            decision_maker.available_paths(model, pre_server_id, deploy_index, i, j))
                        assert path is not False
                        model.sfc_list[i].paths_occupied.append(path)  # path occupied
                        Monitor.path_occupied(path, i)
                        model.sfc_list[i].latency_occupied += (
                                path.latency + model.sfc_list[i].vnf_list[j].latency)  # latency occupied
                        Monitor.latency_occupied_change(i, model.sfc_list[i].latency_occupied - path.latency - model.sfc_list[i].vnf_list[j].latency, model.sfc_list[i].latency_occupied)
                        model.occupy_path(path, model.sfc_list[i].throughput)
                    if j == len(model.sfc_list[i].vnf_list) - 1 and deploy_index != model.sfc_list[i].d:  # final vnf
                        final_path = decision_maker.shortest_path(model, deploy_index, model.sfc_list[i].d)
                        assert final_path is not False
                        model.sfc_list[i].paths_occupied.append(final_path)
                        Monitor.path_occupied(final_path, i)
                        model.sfc_list[i].latency_occupied += final_path.latency
                        Monitor.latency_occupied_change(i, model.sfc_list[i].latency_occupied - final_path.latency, model.sfc_list[i].latency_occupied)
                        model.occupy_path(final_path, model.sfc_list[i].throughput)
                    model.sfc_list[i].vnf_list[j].deploy_decision = deploy_index  # deploy_decision
                    model.topo.nodes[deploy_index]["computing_resource"] -= model.sfc_list[i].vnf_list[
                        j].computing_resource  # computing resource
                    Monitor.computing_resource_change(deploy_index, model.topo.nodes[deploy_index]["computing_resource"] + model.sfc_list[i].vnf_list[j].computing_resource, model.topo.nodes[deploy_index]["computing_resource"])
                    if j == len(model.sfc_list[i].vnf_list) - 1:
                        model.sfc_list[i].state = State.running
                        Monitor.change_running(i)
                # failed
                else:
                    # revert
                    model.revert_failed(failed_sfc_index=i, failed_vnf_index=j)
                    break


def remove_expired(model: Model, time: int):
    '''
    Remove expired SFCs
    Notice: We do not need to reset the deploy decision of vnf and the links occupied of sfc, because we just need to
    transfer the state from running to expired
    :param model: model object
    :param time: cur time stamp
    '''
    for i in range(len(model.sfc_list)):
        # not failed, just running
        if model.sfc_list[i].state == State.running and model.sfc_list[i].time + model.sfc_list[i].TTL < time:
            model.sfc_list[i].state = State.expired
            Monitor.change_expired(i)

            # release computing resource
            for j in range(len(model.sfc_list[i].vnf_list)):
                assert model.sfc_list[i].vnf_list[j].deploy_decision != -1
                before = model.topo.nodes[model.sfc_list[i].vnf_list[j].deploy_decision]["computing_resource"]
                model.topo.nodes[model.sfc_list[i].vnf_list[j].deploy_decision]["computing_resource"] += model.sfc_list[i].vnf_list[j].computing_resource
                after = model.topo.nodes[model.sfc_list[i].vnf_list[j].deploy_decision]["computing_resource"]
                Monitor.computing_resource_change(model.sfc_list[i].vnf_list[j].deploy_decision, before, after)

            # release link bandwidth
            for path in model.sfc_list[i].paths_occupied:
                for i in range(len(path.path) - 1):
                    before = model.topo.edges[path.path[i], path.path[i + 1]]["bandwidth"]
                    model.topo.edges[path.path[i], path.path[i + 1]]["bandwidth"] +=  model.sfc_list[i].throughput
                    after = model.topo.edges[path.path[i], path.path[i + 1]]["bandwidth"]
                    Monitor.bandwidth_change(path.path[i], path.path[i + 1], before, after)


def process_time_slot(model: Model, decision_maker: DecisionMaker, time: int, time_step: int):
    '''
    Function used to simulate within given time slot
    :param model: model environment
    :param decision_maker: decision maker
    :param time: time
    :param time_step: time interval
    :return: nothing
    '''

    # First, expired SFCs must be removed, and the resources occupied must be released (computing resource, bandwidth resource)
    remove_expired(model, time)

    # idle time slot
    deploy_sfcs_in_timeslot(model, decision_maker, time, time_step)

