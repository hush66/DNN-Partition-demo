# DNN-Partition-demo

###  文件结构

- DNN_Partition_Server.py:

  RPC服务器，执行切分后模型的右半部分推断

- DNN_Partition_Server.py:

  RPC客户端，执行切分后模型的左半部分

- Model_Pair.py：

  定义了不同切分位置的模型对，模型对名称命名格式为‘NetExit{.}Part{.}’

- untils.py：

  定义了工具函数，如数据集读取

- config.py：

  常量定义

- Branchy_Alexnet_Infer.py：

  模型推断函数定义

pass

### 