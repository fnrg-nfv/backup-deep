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

## Notations
## Constraints

## Objectives

# Solution

- 建立一个基于**时序**的模型，即根据SFC的到来时间来做决定；
- 强化学习算法只负责决定节点放置的位置，更新的链路使用贪心算法解决；
- 使用强化学习算法进行解决。
- 在考虑放置备份时，在当前物理资源满足的情况下：考虑**不预留资源/仅预留节点资源/预留节点资源与链路资源**；

## 选路算法

找到所有满足吞吐率和延迟的路径，然后选择最短的。

## SFC State

|   State    | Description                                                  |
| :--------: | :----------------------------------------------------------- |
| Undeployed | Not yet time to deploy.                                      |
|   Failed   | Deploy failed.                                               |
|   Normal   | The deployment is successful and the active instance is running. |
|   Backup   | The deployment is successful, the active instance is damaged, and the stand-by instance is running. |
|   Broken   | The deployment is successful, the active instance is not running, and the stand-by instance is not running too. |

The state transition graph as following:

| State transition  | Handle time    | Condition                                                    |
| :---------------: | -------------- | ------------------------------------------------------------ |
| Undeployed→Failed | present        | When a sfc need to be deployed, and can't be deployed.       |
| Undeployed→Normal | present        | When a sfc need to be deployed, and deployed.                |
|   Normal→Backup   | each time slot | When an active instance failed, the stand-by instance started. |
|   Normal→Broken   | each time slot | When an active instance failed, the stand-by instance can't be started, **or time expired**. |
|   Backup→Broken   | each time slot | When a stand-by instance failed, **or time expired**.        |

## `Broken` reasons

- **Time expired**: if a stand-by instance not start;
- Sudden damage of the stand-by instance;
- The stand-by instance did not start successfully when the active instance is damaged(because resource requirements are not met).

## Some processes

### In each time slot, handle state transition and reclaim resources

**The first two transitions shouldn't be handled in this process.**

- Determine which instance should failed in **previous time slot**, handle the transition: 

  - Normal→Backup;
  - Normal→Broken;
  - Backup→Broken.

  then reclaim the resource.

- Determine which sfc is expired, **if the state is broken, then don't need to bother it**, for it has been handled in previous process, handle the expired condition.

### When a stand-by instance need to be start

- Release the reserved resources occupied by this stand-by instance;

## Test environment

| Name             | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| NoBackup         | No backup.                                                   |
| Aggressive       | aggressive method, don't consider current remaining resources. |
| Normal           | consider current remaining resources, will failed for **new active** or **stand-by→active**. |
| MaxReservation   | consider current remaining and max reserved, will failed for **stand-by→active**. |
| FullyReservation | will not failed.                                             |

## State

## Action

- 主要SFC放置server，即active instance；
- 备份SFC放置server，即stand-by instance；

- ~~从start server到active instance的链路；~~

- ~~从active instance到end server的链路；~~

- ~~从start server到stand-by instance的链路；~~

- ~~从stand-by instance到end server的链路；~~

- ~~从active instance到stand-by instance的状态更新链路。~~
