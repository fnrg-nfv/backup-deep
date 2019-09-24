# Path
This class is denoted as the path from one server to another.
## member variables
- **start**: start server;
- **destination**: destination server;
- **path**: path list;
- **latency**: latency requirement;
- **path_length**: the length of this path.

# Monitor

Designed for Monitoring the actions of whole system.

# VNF

This class is denoted as a VNF.

## member variables

- **latency**: process latency;
- **computing_resource**: computing resource requirement;
- **deploy_decision**: -1 represent not deployed, otherwise the server index.

# SFC

This class is denoted as a SFC.

## member variables

- **vnf_list**: the vnfs need to process;
- **latency**: totally latency required;
- **throughput**: totally throughput required;
- **s**: start server;
- **d**: destination server;
- **time**: arriving time;
- **TTL**: time to live;
- ~~**latency_occupied**: including VNF process latency, **updated when deployed, and reclaimed when deploy failed, expired or broken**;~~
- **path_occupied**: including the path from s to current server and current server to d;