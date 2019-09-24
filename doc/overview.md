# toThink

- stateful or stateless?如果是有状态的，那么所有有相同集合状态（aggregate state）的VNF需要放置在同一个server上，以避免状态更新带来的复杂的开销（如果设计是以SFC为粒度的，那么就无需考虑了）；如果是无状态的，那么对于备份的实例（stand-by instance），是否还需要活动的实例将状态进行迁移？
- computing resource是否包含内存资源，还是只包含计算资源。
- 可以创建一个库，专门用来生成SFC；
- 算法的可扩展性：当加入新的资源时怎么办；
- NFV-deep当中路径是怎么选的？
- 列一个表格，定义损坏的broken，完好的，与完全失败的failed；
- 在模型中加入随机宕机随机恢复的动态过程；
- 为什么同样是随机使得服务器坏掉，强化学习一定比随机放置backup的更好呢。一是因为强化学习会考虑更新的开销，另一方面强化学习能够配置更好的backup组合，在服务器坏掉时，不会出现太大的纰漏，纰漏自己去想；
- 可以总结一下stand-by无法启动的原因；
- 可以总结下脱离服务的原因；
- 宕机就把他的的节点资源重置为0，并且把链路资源重置为0；
- 当仅能放下一个SFC，而无法放置其备份时，如何处理。

# Problem

- NFV-P问题；
- Online-Placement；
- 以SFC为粒度而非以NFV为粒度；
- 考虑每一个SFC都有一个备份；
- 考虑每一个备份在变为活动前不占用**节点资源**（计算资源，内存资源）与**链路资源**；
- 在考虑放置备份时，在当前物理资源满足的情况下：考虑**不预留资源/仅预留节点资源/预留节点资源与链路资源**；

## Objective

### main

- fail rate：考查backup的效果；
- load balance: 节点资源交叉熵、链路资源交叉熵的加权和，防止出现负载过重的情况。

### secondary

- cost: 真正的链路消耗应该是多少？是否像NFVdeep里定义的那样，只要开机便把所有的都算入消耗。
- Requests admitted;
- total throughput;

# Solution

- 建立一个基于**时序**的模型，即根据SFC的到来时间来做决定。

- 使用强化学习算法进行解决。

## State



## Action

- 主要SFC放置server，即active instance；
- 备份SFC放置server，即stand-by instance；

- ~~从start server到active instance的链路；~~

- ~~从active instance到end server的链路；~~

- ~~从start server到stand-by instance的链路；~~

- ~~从stand-by instance到end server的链路；~~

- ~~从active instance到stand-by instance的状态更新链路。~~

  