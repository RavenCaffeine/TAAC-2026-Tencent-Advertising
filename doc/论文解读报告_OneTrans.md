# OneTrans 论文解读报告

论文：OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender

## 概要

OneTrans 是三篇中最“统一 backbone”取向的一篇。它认为工业推荐长期把 sequence modeling 和 feature interaction 分成两条线：序列模块如 LONGER 负责行为历史，交互模块如 Wukong/RankMixer 负责非序列特征。这样会限制双向信息交换，也让工程优化碎片化。

OneTrans 提出把 sequential features 和 non-sequential features 都转成一个 token sequence，送入同一个 Transformer-style stack 中联合建模。为了适配推荐系统中 token 来源差异大这一点，它采用 mixed parameterization：序列 token 共享一套参数，非序列 token 使用 token-specific 参数，以兼顾共享统计强度与异构语义表达。

工程上，OneTrans 通过 causal attention、pyramid pruning、cross-request KV caching、FlashAttention、mixed precision 等优化，使统一 Transformer 在工业 serving 中可落地。线上 A/B 中相对 RankMixer+Transformer baseline 取得 click/order/GMV 提升，并降低 p99 latency。

## 详细解读

### 1. 从 encode-then-interaction 到 one backbone

传统结构：

1. 序列特征进入序列模型，被压成 compressed sequence representation。
2. 非序列特征进入 feature interaction 模块。
3. 二者在后期合并。

OneTrans 的结构：

1. SequentialTokenizer 生成 S tokens。
2. Non-SeqTokenizer 生成 NS tokens。
3. S tokens 和 NS tokens 拼成统一 token 序列。
4. 一个 OneTrans stack 同时完成序列建模和特征交互。

核心转变是：序列和非序列不再是两个模块的输出，而是一开始就是同一个计算图里的 token。

### 2. Tokenization

非序列 tokenization 支持两类：

1. Group-wise Tokenizer：按语义组切分，和 RankMixer 风格接近。
2. Auto-Split Tokenizer：先 concat 所有特征，再统一投影并切成固定数量 token。

序列侧则把多行为序列转为 sequential tokens。统一 tokenization 让推荐模型更像 LLM 的 token 序列，但 token 语义比文本复杂得多。

### 3. Mixed Parameterization

OneTrans 没有简单地对所有 token 使用同一套 Transformer 参数。原因是 RecSys token 异构性强：user age、item id、category、context、行为序列并不像文本 token 那样同质。

设计上：

1. sequential tokens 共享参数，因为它们来自相似的行为序列结构。
2. non-sequential tokens 使用 token-specific 参数，保留每类特征的独特语义。

这也是 OneTrans 与普通 Transformer 直接套用的关键差异。

### 4. 工程优化

统一 Transformer 如果不优化会很重。论文强调几项系统设计：

1. causal attention：使 KV caching 成为可能。
2. cross-request / cross-candidate KV caching：复用用户侧序列计算，候选较多时减少重复开销。
3. pyramid stack：逐层剪掉部分序列 query/token，把信息压到少量 tail tokens 中。
4. FlashAttention、mixed precision、recomputation：继承 LLM 生态成熟优化。

这些优化的意义是：统一建模不只是学术结构，还要在 p99 latency 和 GPU memory 上成立。

### 5. 对当前项目的启发

OneTrans 对 MVP 的启发主要不是马上重写成完整 OneTrans，而是：

1. 入口要支持混合精度和更深 backbone，和 LLM/Transformer 工程实践对齐。
2. 序列 token 与非序列 token 的交互应尽可能提前。
3. 多行为域可以逐层交互，而不是每个域独立压缩后最后拼接。
4. 后续若要进一步改进，可考虑统一 token stack、pyramid pruning、KV cache，而不是只调 batch/lr。

## 如何检验自己掌握了核心要点

可以用下面问题自测：

1. OneTrans 和 HyFormer 都想统一序列建模和特征交互，但统一程度有什么不同？
2. 为什么 OneTrans 不直接使用普通 Transformer 的完全共享参数？
3. Group-wise Tokenizer 和 Auto-Split Tokenizer 各自适合什么情况？
4. causal attention 与 KV caching 为什么绑定在一起？
5. pyramid pruning 牺牲了什么，换来了什么？
6. 如果要把当前 MVP 继续向 OneTrans 演进，第一步应该改模型结构、训练效率，还是 serving cache？为什么？

能回答这些问题，并能把 `AMP`、`RoPE`、`cross-domain query interaction` 和更深 blocks 看成 OneTrans/HyFormer 思路的局部落地，就说明已经理解本文核心。
