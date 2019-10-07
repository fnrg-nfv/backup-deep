from sfcbased.model import *
from sfcbased.detecter import *


def deploy_sfcs_in_timeslot(model: Model, decision_maker: DecisionMaker, time: int, test_env: TestEnv):
    """
    Deploy the sfcs located in given timeslot.
    :param model: model
    :param decision_maker: make decision
    :param time: time
    :param test_env: test environment
    :return: nothing
    """

    # no backup, don't need to place stand-by instance
    if test_env == TestEnv.NoBackup:
        for i in range(len(model.sfc_list)):
            # for each sfc which locate in this time slot
            if time <= model.sfc_list[i].time < time + 1:
                assert model.sfc_list[i].state == State.Undeployed
                decision = decision_maker.make_decision(model, i, test_env)

                # Undeployed→Failed
                if not decision:
                    model.sfc_list[i].set_state(time, i, State.Failed)

                # Undeployed→Normal
                else:
                    model.sfc_list[i].active_sfc.server = decision.active_server
                    model.sfc_list[i].standby_sfc.server = decision.standby_server
                    deploy_active(model, i, test_env)
                    model.sfc_list[i].set_state(time, i, State.Normal)

    # with backup
    else:
        for i in range(len(model.sfc_list)):
            # for each sfc which locate in this time slot
            if time <= model.sfc_list[i].time < time + 1:
                assert model.sfc_list[i].state == State.Undeployed
                decision = decision_maker.make_decision(model, i, test_env)

                # Undeployed→Failed
                if not decision:
                    model.sfc_list[i].set_state(time, i, State.Failed)

                # Undeployed→Normal
                else:
                    model.sfc_list[i].active_sfc.server = decision.active_server
                    model.sfc_list[i].standby_sfc.server = decision.standby_server
                    model.sfc_list[i].active_sfc.path_s2c = decision.active_path_s2c
                    model.sfc_list[i].active_sfc.path_c2d = decision.active_path_c2d
                    model.sfc_list[i].standby_sfc.path_s2c = decision.standby_path_s2c
                    model.sfc_list[i].standby_sfc.path_c2d = decision.standby_path_c2d
                    model.sfc_list[i].update_path = decision.update_path
                    deploy_active(model, i, test_env)
                    deploy_standby(model, i, test_env)
                    model.sfc_list[i].set_state(time, i, State.Normal)


def deploy_active(model: Model, sfc_index: int, test_env: TestEnv):
    """
    Start active, it must can start because we have examined it
    :param model: model
    :param sfc_index: sfc index
    :param test_env: test environment
    :return: nothing
    """

    # occupy computing resource
    model.topo.nodes[model.sfc_list[sfc_index].active_sfc.server]["active"] += model.sfc_list[
        sfc_index].computing_resource

    # occupy path bandwidth
    path_s2c = model.sfc_list[sfc_index].active_sfc.path_s2c
    path_c2d = model.sfc_list[sfc_index].active_sfc.path_c2d
    for i in range(len(path_s2c) - 1):
        model.topo.edges[path_s2c[i], path_s2c[i + 1]]["active"] += model.sfc_list[sfc_index].tp
    for i in range(len(path_c2d) - 1):
        model.topo.edges[path_c2d[i], path_c2d[i + 1]]["active"] += model.sfc_list[sfc_index].tp

    # No Backup
    if test_env != TestEnv.NoBackup:
        # occupy update bandwidth
        path = model.sfc_list[sfc_index].update_path
        for i in range(len(path) - 1):
            model.topo.edges[path[i], path[i + 1]]["active"] += model.sfc_list[sfc_index].update_tp


def deploy_standby(model: Model, sfc_index: int, test_env: TestEnv):
    """
    Deploy active, it must can be deployed because we have examined it
    :param model: model
    :param sfc_index: sfc index
    :param test_env: test environment
    :return: nothing
    """
    assert test_env != TestEnv.NoBackup

    # MaxReservation
    if test_env == TestEnv.MaxReservation:

        # computing resource reservation
        model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["sbsfcs"].add(sfc_index)
        if model.sfc_list[sfc_index].computing_resource > \
                model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["reserved"]:
            model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["reserved"] = model.sfc_list[
                sfc_index].computing_resource
            model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["max_sbsfc_index"] = sfc_index

        # bandwidth reservation
        path_s2c = model.sfc_list[sfc_index].standby_sfc.path_s2c
        path_c2d = model.sfc_list[sfc_index].standby_sfc.path_c2d
        for i in range(len(path_s2c) - 1):
            model.topo.edges[path_s2c[i], path_s2c[i + 1]]["sbsfcs"].add(sfc_index)
            if model.sfc_list[sfc_index].tp > model.topo.edges[path_s2c[i], path_s2c[i + 1]]["reserved"]:
                model.topo.edges[path_s2c[i], path_s2c[i + 1]]["reserved"] = model.sfc_list[sfc_index].tp
                model.topo.edges[path_s2c[i], path_s2c[i + 1]]["max_sbsfc_index"] = sfc_index
        for i in range(len(path_c2d) - 1):
            model.topo.edges[path_c2d[i], path_c2d[i + 1]]["sbsfcs"].add(sfc_index)
            if model.sfc_list[sfc_index].tp > model.topo.edges[path_c2d[i], path_c2d[i + 1]]["reserved"]:
                model.topo.edges[path_c2d[i], path_c2d[i + 1]]["reserved"] = model.sfc_list[sfc_index].tp
                model.topo.edges[path_c2d[i], path_c2d[i + 1]]["max_sbsfc_index"] = sfc_index

    # FullyReservation
    if test_env == TestEnv.FullyReservation:

        # computing resource reservation
        model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["sbsfcs"].add(sfc_index)
        model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["reserved"] += model.sfc_list[
            sfc_index].computing_resource

        # bandwidth reservation
        path_s2c = model.sfc_list[sfc_index].standby_sfc.path_s2c
        path_c2d = model.sfc_list[sfc_index].standby_sfc.path_c2d
        for i in range(len(path_s2c) - 1):
            model.topo.edges[path_s2c[i], path_s2c[i + 1]]["sbsfcs"].add(sfc_index)
            model.topo.edges[path_s2c[i], path_s2c[i + 1]]["reserved"] += model.sfc_list[sfc_index].tp
        for i in range(len(path_c2d) - 1):
            model.topo.edges[path_c2d[i], path_c2d[i + 1]]["sbsfcs"].add(sfc_index)
            model.topo.edges[path_c2d[i], path_c2d[i + 1]]["reserved"] += model.sfc_list[sfc_index].tp


def active_failed(model: Model, sfc_index: int, test_env: TestEnv):
    """
    Handle the active instance failed condition, including resource reclaiming
    :param model: model
    :param sfc_index: the index of sfc
    :param test_env: test environment
    :return: nothing
    """
    # release computing resource
    model.topo.nodes[model.sfc_list[sfc_index].active_sfc.server]["active"] -= model.sfc_list[
        sfc_index].computing_resource

    # release path bandwidth
    path_s2c = model.sfc_list[sfc_index].active_sfc.path_s2c
    path_c2d = model.sfc_list[sfc_index].active_sfc.path_c2d
    for i in range(len(path_s2c) - 1):
        model.topo.edges[path_s2c[i], path_s2c[i + 1]]["active"] -= model.sfc_list[sfc_index].tp
    for i in range(len(path_c2d) - 1):
        model.topo.edges[path_c2d[i], path_c2d[i + 1]]["active"] -= model.sfc_list[sfc_index].tp

    # release update path bandwidth
    if test_env != TestEnv.NoBackup:
        path = model.sfc_list[sfc_index].update_path
        for i in range(len(path) - 1):
            model.topo.edges[path[i], path[i + 1]]["active"] -= model.sfc_list[sfc_index].update_tp


def standby_start(model: Model, sfc_index: int, test_env: TestEnv):
    """
    Handle the stand-by instance start condition.
    :param model: model
    :param sfc_index: sfc index
    :param test_env: test environment
    :return: start success or not
    """
    assert test_env != TestEnv.NoBackup

    # FullyReservation, is this condition, at any time the stand-by instance can start and we don't need to examine
    if test_env == TestEnv.FullyReservation:
        model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["active"] += model.sfc_list[
            sfc_index].computing_resource
        model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["reserved"] -= model.sfc_list[
            sfc_index].computing_resource
        path_s2c = model.sfc_list[sfc_index].standby_sfc.path_s2c
        path_c2d = model.sfc_list[sfc_index].standby_sfc.path_c2d
        for i in range(len(path_s2c) - 1):
            model.topo.edges[path_s2c[i], path_s2c[i + 1]]["active"] += model.sfc_list[sfc_index].tp
            model.topo.edges[path_s2c[i], path_s2c[i + 1]]["reserved"] -= model.sfc_list[sfc_index].tp
        for i in range(len(path_c2d) - 1):
            model.topo.edges[path_c2d[i], path_c2d[i + 1]]["active"] += model.sfc_list[sfc_index].tp
            model.topo.edges[path_c2d[i], path_c2d[i + 1]]["reserved"] -= model.sfc_list[sfc_index].tp
        return True

    # others(Aggressive, Normal, MaxReservation)
    # examination
    remaining = model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["computing_resource"] - \
                model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["active"]
    if remaining < model.sfc_list[sfc_index].computing_resource:
        return False
    path_s2c = model.sfc_list[sfc_index].standby_sfc.path_s2c
    path_c2d = model.sfc_list[sfc_index].standby_sfc.path_c2d
    for i in range(len(path_s2c) - 1):
        remaining = model.topo.edges[path_s2c[i], path_s2c[i + 1]]["bandwidth"] - \
                    model.topo.edges[path_s2c[i], path_s2c[i + 1]]["active"]
        if remaining < model.sfc_list[sfc_index].tp:
            return False
    for i in range(len(path_c2d) - 1):
        remaining = model.topo.edges[path_c2d[i], path_c2d[i + 1]]["bandwidth"] - \
                    model.topo.edges[path_c2d[i], path_c2d[i + 1]]["active"]
        if remaining < model.sfc_list[sfc_index].tp:
            return False

    # MaxReservation - update nodes/edges reservation state
    if test_env == TestEnv.MaxReservation:

        # remove computing resource reservation
        model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["sbsfcs"].remove(sfc_index)
        if model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["max_sbsfc_index"] == sfc_index:
            maxvalue = float("-inf")
            maxindex = -1
            for index in model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["sbsfcs"]:
                if model.sfc_list[index].computing_resource > maxvalue:
                    maxvalue = model.sfc_list[index].computing_resource
                    maxindex = index
            model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["max_sbsfc_index"] = maxindex
            model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["reserved"] = maxvalue

        # remove bandwidth reservation
        for i in range(len(path_s2c) - 1):
            model.topo.edges[path_s2c[i], path_s2c[i + 1]]["sbsfcs"].remove(sfc_index)
            if model.topo.edges[path_s2c[i], path_s2c[i + 1]]["max_sbsfc_index"] == sfc_index:
                maxvalue = float("-inf")
                maxindex = -1
                for index in model.topo.edges[path_s2c[i], path_s2c[i + 1]]["sbsfcs"]:
                    if model.sfc_list[index].tp > maxvalue:
                        maxvalue = model.sfc_list[index].tp
                        maxindex = index
                model.topo.edges[path_s2c[i], path_s2c[i + 1]]["max_sbsfc_index"] = maxindex
                model.topo.edges[path_s2c[i], path_s2c[i + 1]]["reserved"] = maxvalue
        for i in range(len(path_c2d) - 1):
            model.topo.edges[path_c2d[i], path_c2d[i + 1]]["sbsfcs"].remove(sfc_index)
            if model.topo.edges[path_c2d[i], path_c2d[i + 1]]["max_sbsfc_index"] == sfc_index:
                maxvalue = float("-inf")
                maxindex = -1
                for index in model.topo.edges[path_c2d[i], path_c2d[i + 1]]["sbsfcs"]:
                    if model.sfc_list[index].tp > maxvalue:
                        maxvalue = model.sfc_list[index].tp
                        maxindex = index
                model.topo.edges[path_c2d[i], path_c2d[i + 1]]["max_sbsfc_index"] = maxindex
                model.topo.edges[path_c2d[i], path_c2d[i + 1]]["reserved"] = maxvalue

    # start success
    model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["active"] += model.sfc_list[
        sfc_index].computing_resource
    for i in range(len(path_s2c) - 1):
        model.topo.edges[path_s2c[i], path_s2c[i + 1]]["active"] += model.sfc_list[sfc_index].tp
    for i in range(len(path_c2d) - 1):
        model.topo.edges[path_c2d[i], path_c2d[i + 1]]["active"] += model.sfc_list[sfc_index].tp
    return True


def standby_failed(model: Model, sfc_index: int, test_env: TestEnv):
    """
    Handle the stand-by instance failed condition, including resource reclaiming
    :param model: model
    :param sfc_index: sfc index
    :param test_env: test environment
    :return: nothing
    """
    assert test_env != TestEnv.NoBackup

    # release computing resource
    model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["active"] -= model.sfc_list[
        sfc_index].computing_resource

    # release path bandwidth
    path_s2c = model.sfc_list[sfc_index].standby_sfc.path_s2c
    path_c2d = model.sfc_list[sfc_index].standby_sfc.path_c2d
    for i in range(len(path_s2c) - 1):
        model.topo.edges[path_s2c[i], path_s2c[i + 1]]["active"] -= model.sfc_list[sfc_index].tp
    for i in range(len(path_c2d) - 1):
        model.topo.edges[path_c2d[i], path_c2d[i + 1]]["active"] -= model.sfc_list[sfc_index].tp


def remove_expired_active(model: Model, sfc_index: int, test_env: TestEnv):
    """
    Remove active for time expired
    :param model: model
    :param sfc_index: sfc index
    :param test_env: test environment
    :return: nothing
    """

    # we don't need to handle the backup state for the original active instance is not running
    if model.sfc_list[sfc_index].state == State.Normal:
        active_failed(model, sfc_index, test_env)


def remove_expired_standby(model: Model, sfc_index: int, test_env: TestEnv):
    """
    Remove standby for time expired
    :param test_env:
    :param model: model
    :param sfc_index: sfc index
    :return: nothing
    """
    assert test_env != TestEnv.NoBackup

    # is running
    if model.sfc_list[sfc_index].state == State.Backup:
        standby_failed(model, sfc_index, test_env)
        return

    # not running
    assert model.sfc_list[sfc_index].state == State.Normal

    # FullyReservation - remove reservation
    if test_env == TestEnv.FullyReservation:
        model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["reserved"] -= model.sfc_list[
            sfc_index].computing_resource
        path_s2c = model.sfc_list[sfc_index].standby_sfc.path_s2c
        path_c2d = model.sfc_list[sfc_index].standby_sfc.path_c2d
        for i in range(len(path_s2c) - 1):
            model.topo.edges[path_s2c[i], path_s2c[i + 1]]["reserved"] -= model.sfc_list[sfc_index].tp
        for i in range(len(path_c2d) - 1):
            model.topo.edges[path_c2d[i], path_c2d[i + 1]]["reserved"] -= model.sfc_list[sfc_index].tp

    # MaxReservation - remove reservation
    elif test_env == TestEnv.MaxReservation:

        # remove computing resource reservation
        model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["sbsfcs"].remove(sfc_index)
        if model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["max_sbsfc_index"] == sfc_index:
            maxvalue = float("-inf")
            maxindex = -1
            for index in model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["sbsfcs"]:
                if model.sfc_list[index].computing_resource > maxvalue:
                    maxvalue = model.sfc_list[index].computing_resource
                    maxindex = index
            model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["max_sbsfc_index"] = maxindex
            model.topo.nodes[model.sfc_list[sfc_index].standby_sfc.server]["reserved"] = maxvalue

        # remove bandwidth reservation
        path_s2c = model.sfc_list[sfc_index].standby_sfc.path_s2c
        path_c2d = model.sfc_list[sfc_index].standby_sfc.path_c2d
        for i in range(len(path_s2c) - 1):
            model.topo.edges[path_s2c[i], path_s2c[i + 1]]["sbsfcs"].remove(sfc_index)
            if model.topo.edges[path_s2c[i], path_s2c[i + 1]]["max_sbsfc_index"] == sfc_index:
                maxvalue = float("-inf")
                maxindex = -1
                for index in model.topo.edges[path_s2c[i], path_s2c[i + 1]]["sbsfcs"]:
                    if model.sfc_list[index].tp > maxvalue:
                        maxvalue = model.sfc_list[index].tp
                        maxindex = index
                model.topo.edges[path_s2c[i], path_s2c[i + 1]]["max_sbsfc_index"] = maxindex
                model.topo.edges[path_s2c[i], path_s2c[i + 1]]["reserved"] = maxvalue
        for i in range(len(path_c2d) - 1):
            model.topo.edges[path_c2d[i], path_c2d[i + 1]]["sbsfcs"].remove(sfc_index)
            if model.topo.edges[path_c2d[i], path_c2d[i + 1]]["max_sbsfc_index"] == sfc_index:
                maxvalue = float("-inf")
                maxindex = -1
                for index in model.topo.edges[path_c2d[i], path_c2d[i + 1]]["sbsfcs"]:
                    if model.sfc_list[index].tp > maxvalue:
                        maxvalue = model.sfc_list[index].tp
                        maxindex = index
                model.topo.edges[path_c2d[i], path_c2d[i + 1]]["max_sbsfc_index"] = maxindex
                model.topo.edges[path_c2d[i], path_c2d[i + 1]]["reserved"] = maxvalue


def state_tansition_and_resource_reclaim(model: Model, time: int, test_env: TestEnv, failed_instances: List[Instance]):
    """
    In each time slot, handle state transition and reclaim resources.
    :param model: model
    :param time: time slot
    :param test_env: test environment
    :param failed_instances: failed instances
    :return: nothing
    """

    # random failed condition
    for ins in failed_instances:
        index = ins.sfc_index
        is_active = ins.is_active

        # Normal→Backup and Normal→Broken
        if model.sfc_list[index].state == State.Normal:
            assert is_active is True
            active_failed(model, index, test_env)
            if standby_start(model, index, test_env):
                model.sfc_list[index].set_state(time, index, State.Backup)
            else:
                model.sfc_list[index].set_state(time, index, State.Broken, BrokenReason.StandbyStartFailed)

        # Backup→Broken
        elif model.sfc_list[index].state == State.Backup:
            assert is_active is False
            standby_failed(model, index, test_env)
            model.sfc_list[index].set_state(time, index, State.Broken, BrokenReason.StandbyDamage)

    # time expired condition
    for index in range(len(model.sfc_list)):
        if (model.sfc_list[index].state == State.Normal or model.sfc_list[index].state == State.Backup) and model.sfc_list[index].time + model.sfc_list[
            index].TTL < time:
            remove_expired_active(model, index, test_env)
            remove_expired_standby(model, index, test_env)
            model.sfc_list[index].set_state(time, index, State.Broken, BrokenReason.TimeExpired)


def process_time_slot(model: Model, decision_maker: DecisionMaker, time: int, test_env: TestEnv,
                      failed_instances: List[Instance]):
    """
    Function used to simulate within given time slot
    :param test_env:
    :param failed_instances:
    :param model: model environment
    :param decision_maker: decision maker
    :param time: time
    :return: nothing
    """

    # First, handle the state transition and resources reclaim
    # The transition:
    # 1. - Normal→Backup;
    # 2. - Normal→Broken;
    # 3. - Backup→Broken.
    # are processed in this function
    state_tansition_and_resource_reclaim(model, time, test_env, failed_instances)

    # Deploy sfc in this time slot
    # The transition:
    # 1. - Undeployed→Failed;
    # 2. - Undeployed→Normal.
    # are processed in this function
    deploy_sfcs_in_timeslot(model, decision_maker, time, test_env)
