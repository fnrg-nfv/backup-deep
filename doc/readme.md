# Problem
- Online placement model;
- Based on SFC;
- One and only one stand-by;
- Once place failed, not place again;
- Instance broken, not server broken;
- SFC and its stand-by are coupled, they should be placed together; 
- Consider the damage of the stand-by instance.

## Resources

- **Nodes**: Computing resource;
- **Edges**: Bandwidth.

# Solution

- Establish a time-series model that makes decisions based on the arrival time of the SFC；
- Use the reinforcement learning algorithm to determine the location of the node to be placed, and use the greedy algorithm to determine the link occupied;

## Routing algorithm

- Find all paths that meet throughput and latency requirements；
- Choose the shortest one。

## SFC State

|   State    | Description                                                  |
| :--------: | :----------------------------------------------------------- |
| Undeployed | Not yet time to deploy.                                      |
|   Failed   | Deploy failed.                                               |
|   Normal   | The deployment is successful and the active instance is running. |
|   Backup   | The deployment is successful, the active instance is damaged, and the stand-by instance is running. |
|   Broken   | The deployment is successful, the active instance is not running, and the stand-by instance is not running too(if with backup). |

The state transition graph as following:

| State transition  | Handle time    | Condition                                                    |
| :---------------: | -------------- | ------------------------------------------------------------ |
| Undeployed→Failed | present        | When a sfc need to be deployed, and can't be deployed.       |
| Undeployed→Normal | present        | When a sfc need to be deployed, and deployed.                |
|   Normal→Backup   | each time slot | When an active instance failed, the stand-by instance started. |
|   Normal→Broken   | each time slot | When an active instance failed, the stand-by instance can't be started, or without backup, **or time expired**. |
|   Backup→Broken   | each time slot | When a stand-by instance failed, **or time expired**.        |

## `Broken` reasons

- **Time expired**: if a stand-by instance not start;
- Sudden damage of the stand-by instance;
- The stand-by instance did not start successfully when the active instance is damaged(because resource requirements are not met);
- No backup condition when the active instance is damaged;

## Test environment

| Name             | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| NoBackup         | No backup.                                                   |
| Aggressive       | Aggressive method, don't consider current remaining resources. |
| Normal           | Consider current remaining resources, will failed for **new active** or **stand-by→active**. |
| MaxReservation   | Consider current remaining and max reserved, will failed for **stand-by→active**. |
| FullyReservation | Will not failed.                                             |

## Monitor

- Monitor the state change of each sfc;
- Monitor the resources change of each node and edge; 

## Reinforcement Learning method

- On-policy(online);
- Discrete action space; 

## State

## Action

- Location of active instance;
- Location of stand-by instance;
- ~~The link from start server to active instance;~~
- ~~The link from active instance to end server;~~
- ~~The link from start server to stand-by instance;~~
- ~~The link from stand-by instance to end server;~~
- ~~The link from active instance to stand-by instance.~~

## Reward

Mainly three components:

- If this SFC is **accepted**;
- If this SFC's active instance placed good or not;
- If this SFC's stand-by instance placed good or not.

So how to determine an instance is placed good or not? We should consider a function which can compute this, this function has four main parameters:

- The placed server's **computing resource**;
- The **active** area;
- The **max** computing resource of all stand-by instances;
- The number of stand-by instances placed on this server.

# Others

For classes:

[Main classes](https://github.com/fnrg-nfv/backup-deep/blob/master/doc/class.md)

For processes:

[Main processes](https://github.com/fnrg-nfv/backup-deep/blob/master/doc/process.md)

