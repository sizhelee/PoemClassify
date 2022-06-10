# PoemClassify

《Python程序设计与数据科学导论》2022期末作业

**Github Page: <https://github.com/sizhelee/PoemClassify>**

## 任务介绍

- 古诗作者分类：给定古诗句或片段，预测其作者。
- 本次数据选取自全唐诗，包含了李白、杜甫、杜牧、李商隐、刘禹锡五位诗人的作品。
- 数据已切分为训练集、验证集和测试集，大小分别为11271，1408，1410。

## 模型说明

- LSTM通过在RNN中引入门控机制来解决长程依赖问题，将RNN中的hidden state拓展到了hidden state和cell state两部分，前者主要负责短期信息，后者主要负责长期信息。
- 双向LSTM的目的是在编码每个位置的表示时，不仅利用之前时间片的信息，还利用之后时间片的信息。具体做法为用两个LSTM分别对输入进行处理，一个从前往后，另一个从后往前，最后将两个LSTM的hidden state按时间片concat起来输出到下一层。
- 对于双向LSTM，我们希望能综合利用所有时间片的hidden state进行预测，故考虑用attention对所有时间片的hidden state进行聚合。

## 模型结构

Embedding – BiLSTM – Attention – Linear – LogSoftmax

## 模型实现

### 参数

- 训练batch_size: 128
- 训练epoch: 5
- hidden state size: 512
- 优化器: Adam
- 学习率: 5e-3

### 性能

- **Valid Set: 34.659**

- **Test Set:**

## 训练&测试模型

```bash
python train.py
```
