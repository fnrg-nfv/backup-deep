# Problem
- Online placement model;
- Based on SFC;
- One and only one backup;

## Resources

- **Nodes**: Computing resource;
- **Edges**: Bandwidth.

## Notations
## Constraints

## Objectives

# Solution

- 建立一个基于**时序**的模型，即根据SFC的到来时间来做决定；
- 强化学习算法只负责决定节点放置的位置，更新的链路
- 使用强化学习算法进行解决。
- 在考虑放置备份时，在当前物理资源满足的情况下：考虑**不预留资源/仅预留节点资源/预留节点资源与链路资源**；

## State

## Action

- 主要SFC放置server，即active instance；
- 备份SFC放置server，即stand-by instance；

- ~~从start server到active instance的链路；~~

- ~~从active instance到end server的链路；~~

- ~~从start server到stand-by instance的链路；~~

- ~~从stand-by instance到end server的链路；~~

- ~~从active instance到stand-by instance的状态更新链路。~~

# toThink

- 可以创建一个库，专门用来生成SFC；
- 不考虑再为备份的实例放置备份；
- 算法的可扩展性：当加入新的资源时怎么办（那就不加入新的资源）；
- 坏是instance坏，而不是server坏，也就是整个链路的拓扑是不变的；
- 为什么同样是随机使得服务器坏掉，强化学习一定比随机放置backup的更好呢。一是因为强化学习会考虑更新的开销，另一方面强化学习能够配置更好的backup组合，在服务器坏掉时，不会出现太大的纰漏，纰漏自己去想；
- 列一个表格，定义损坏的broken，完好的，与完全失败的failed；
- 可以总结一下stand-by无法启动的原因；
- 可以总结下脱离服务的原因；
- 当仅能放下一个SFC，而无法放置其备份时，判定为放置失败。