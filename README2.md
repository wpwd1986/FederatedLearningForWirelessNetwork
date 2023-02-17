# Federated Learning In Wireless
Resource Allocation in Wireless Federated Learning Based on Channel and Learning Qualities.

*By Jichao Leng*

## 项目结构与目录
#### 无线信道仿真模块
*仿真多个用户在一定范围内移动时与基站之间的无线信道状态变化。*
- /channels - 无线环境仿真目录
  - /Random PathLoss
  - /Spatial Channel Model
  - /Winner2 and Random Waypoint Model

#### 联邦学习训练模块
*仿真多用户在信道资源受限情况下的联邦学习性能。*
- /data - 数据集存储目录
  - /MNIST
  - /CIFAR
- /dicts - 场景预设文件目录
  - /Datadict
  - /SNR
- /utils - 仿真配置目录
  - options.py
  - sampling.py
- /models - 训练函数目录
  - Nets.py
  - Update.py
  - ExplainAI.py
  - Fed.py
  - test.py
- main_fed.py - 单基站模式
- main_fed_multi.py - 多基站模式
- /save - 仿真结果保存目录

#### 结果分析模块
*根据仿真结果进行绘图。*
- /plot - 绘图脚本目录
  - Plot_AccSel.m
  - Plot_BalanceFactor.m
  - Plot_UserSNR.m

## 用户选择方法
- rand - Random selection
- chan - Channel quality only
- imp  - Gradient entropy \* loss
- nor  - Gradient norm
- ent  - Gradient entropy
- bal  - Combined Quality
- grad - Gradient divergence
- cadf - Category difference
- cdls - Category diff \* loss
- los  - Loss

