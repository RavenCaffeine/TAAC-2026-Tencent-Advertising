# InterFormer 论文解读报告

论文：InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction

## 概要

InterFormer 研究 CTR 中异构信息的互利交互：非序列特征如 user profile、ad/item/context 描述长期或静态偏好，行为序列描述动态兴趣。已有模型大多让非序列信息去指导序列建模，但很少让序列信息反过来增强非序列特征；同时，为了控制计算量，很多方法过早对序列或特征做 pooling/sum/concat，造成信息损失。

论文提出 InterFormer，以 interleaving 的方式交替更新非序列表示和序列表示。它由三部分组成：Interaction Arch 负责在序列摘要指导下学习 behavior-aware 的非序列特征交互；Sequence Arch 负责在非序列摘要指导下学习 context-aware 的序列表示；Cross Arch 负责在两种模态之间做选择性摘要和信息交换。

论文的关键价值是强调双向信息流和延迟聚合：保留每种模态的完整 token 表示，在层间逐步交换摘要，而不是一开始就粗暴压缩。

## 详细解读

### 1. 两个核心问题

InterFormer 将已有 CTR 异构建模问题概括为：

1. inter-mode interaction 不足：信息流通常是非序列 -> 序列，缺少序列 -> 非序列。
2. aggressive aggregation：为了降低复杂度，早期把长序列或大量特征压成一个向量，导致细粒度信息丢失。

这两个问题和 HyFormer/OneTrans 的批判对象一致：不是模型没有序列模块或交互模块，而是两者连接方式太粗。

### 2. Interaction Arch

Interaction Arch 面向非序列特征交互。传统 DCN、DeepFM、DHEN 等模块可在这里作为基础交互模型。InterFormer 的特别之处是，它不是只在非序列 token 内部交互，而是接收来自序列侧的 summarization/query，让非序列表示能感知用户当前行为兴趣。

换句话说，用户近期行为可以反过来改变 user/item/context token 的交互方式，使非序列表示变成 behavior-aware。

### 3. Sequence Arch

Sequence Arch 面向行为序列建模。它使用 Personalized FeedForward Network (PFFN) 和 Multi-Head Attention。

PFFN 的核心思想是：用非序列摘要生成或调制作用在序列 token 上的 FFN 权重，让序列表示在更新时感知 user/item/context 等上下文。之后 MHA 再对序列 token 做信息选择和建模。

这比简单地把 candidate item embedding 当 attention query 更通用，因为非序列摘要可以包含更丰富的上下文。

### 4. Cross Arch

Cross Arch 是两种模态的桥。由于完整 token 直接互相交互计算量大、噪声也大，InterFormer 先分别从 Interaction Arch 和 Sequence Arch 中得到摘要，再交换这些摘要以指导下一层更新。

这是一种折中：不把所有 token 全量互注意力，也不一开始就压扁，而是用可控的摘要通道实现双向信息流。

### 5. 实验意义

论文报告在 benchmark 数据上最高约 0.14% AUC 提升，在 Meta Ads 工业数据上有 0.15% NE gain 和 24% QPS gain。工业 CTR 中 0.1% 量级提升通常已经很有价值，说明交互结构比单纯增加参数更关键。

## 如何检验自己掌握了核心要点

可以用下面问题自测：

1. InterFormer 说的“双向信息流”具体是哪两个方向？
2. 为什么早期 pooling/sum 序列会造成 aggressive aggregation？
3. Interaction Arch、Sequence Arch、Cross Arch 分别负责什么？
4. PFFN 为什么叫 personalized？它和普通 Transformer FFN 有什么不同？
5. InterFormer 和 HyFormer 都强调交替更新，它们在实现思想上有什么差异？
6. 如果把 Cross Arch 去掉，模型会退化成什么样的信息流？

能回答这些问题，并能说清它对当前 MVP 的启发：让 item/context 参与 query、让多序列域之间交换信息、保留可 ablation 的开关，就说明已经掌握了核心。
