# Path
This class is denoted as the path from one server to another.
- **start(int)**: start server;
- **destination(int)**: destination server;
- **path**: path list, e.g. [5, 2, 0];
- **latency(float)**: latency occupied;
- **path_length**: the length of this path.

# Monitor

Designed for Monitoring the actions of whole system.

# Node

This class is denoted as a server.

- **computing_resource**: computing capacity;
- **active**: active computing resource;
- **reserved**: reserved computing resource;
- **max_sbsfc_index**: max stand-by sfc;
- **sbsfcs**: sfcs.

# Edge

This class is denoted as a link.

- **bandwidth**: bandwidth capacity;
- **active**: active bandwidth;
- **reserved**: reserved bandwidth;
- **max_sbsfc_index**: max stand-by sfc;
- **sbsfcs**: sfcs;
- **latency**: link latency.

# Instance:

This class is denoted as an instance.

- **sfc_index(int)**: the sfc index;
- **is_active(bool)**: is active or not.

# SFC

This class is denoted as a SFC.

member variables as following:

- **active_sfc(ACSFC)**: an active sfc;
- **standby_sfc(SBSFC)** a stand-by sfc;
- **computing_resource(int)**: computing_resource required;
- **tp(int)**: totally throughput required;
- **latency(float)**: totally latency required;
- **update_tp(int)**: update throughput required;
- **process_latency(float)**: latency of processing;
- **s(int)**: start server;
- **d(int)**: destination server;
- **time(float)**: arriving time;
- **TTL(int)**: time to live;
- **state(State)**: current state;
- **update_path(Path)**: the update path from active instance to stand-by instance;
- ~~**vnf_list**: the vnfs need to process;~~

# ACSFC

This class is denoted as an active SFC.

- **downtime(float, shouldn't excess the time+TTL)**: down time of this active SFC(if not down then should be 0);
- **server(int)**: the server placed;
- **path_s2c(Path)**: including the path from start server to current server;
- **path_c2d(Path)**: including the path from current server to destination server;
- ~~**latency_occupied**: including VNF process latency, updated when deployed, and reclaimed when deploy failed, expired or broken;~~

# SBSFC

This class is denoted as a stand-by SFC.

- **starttime(float)**: start time of this stand-by SFC;
- **server(int)**: the server placed;
- **downtime(float, shouldn't excess the time+TTL)**: down time of this stand-by SFC(if not down then should be 0);
- **path_s2c(Path)**: including the path from start server to current server;
- **path_c2d(Path)**: including the path from current server to destination server;
- ~~**latency_occupied**: including VNF process latency, updated when deployed, and reclaimed when deploy failed, expired or broken;~~

---

# VNF

This class is denoted as a VNF.

- **latency**: process latency;
- **computing_resource**: computing resource requirement;
- **deploy_decision**: -1 represent not deployed, otherwise the server index.