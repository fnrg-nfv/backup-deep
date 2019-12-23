from generate_topo import *

if __name__ == "__main__":
    topo = nx.Graph()
    cs_low = 20000
    cs_high = 40000
    bandwidth_low = 100
    bandwidth_high = 300
    fail_rate_low = 0.0
    fail_rate_high = 0.4
    latency_low = 2
    latency_high = 5

    # generate V
    for i in range(18):
        computing_resource = random.randint(cs_low, cs_high)
        fail_rate = random.uniform(fail_rate_low, fail_rate_high)
        topo.add_node(i, computing_resource=computing_resource, fail_rate=fail_rate, active=0, reserved=0, max_sbsfc_index=-1, sbsfcs=set())

    # generate E
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(0, 1, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(1, 9, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(1, 17, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(1, 6, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(2, 9, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(2, 3, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(3, 11, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(4, 14, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(5, 17, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(6, 17, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(6, 7, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(7, 17, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(8, 17, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(9, 11, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(8, 9, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(9, 14, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(10, 12, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(10, 11, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(11, 13, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(13, 14, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(14, 17, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(16, 17, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)
    topo.add_edge(14, 15, bandwidth=bandwidth, active=0, reserved=0, latency=latency, max_sbsfc_index=-1, sbsfcs_s2c=set(), sbsfcs_c2d=set())
    bandwidth = random.randint(bandwidth_low, bandwidth_high)
    latency = random.uniform(latency_low, latency_high)

    if load_model:
        with open(model_file_name, 'wb') as f:  # open file with write-mode
            sfc_list = generate_sfc_list(topo, process_capacity, size=sfc_size, duration=duration, jitter=jitter)
            nx.draw(topo)
            show()
            model = Model(topo=topo, sfc_list=sfc_list)
            model_string = pickle.dump(model, f)  # serialize and save object
    else:
        with open(topo_file_name, 'wb') as f:  # open file with write-mode
            nx.draw(topo)
            show()
            topo_string = pickle.dump(topo, f)  # serialize and save object

