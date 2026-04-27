# HyFormer 论文解读报告

论文：HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction

## 概要

HyFormer 关注工业 CTR/推荐模型中的一个核心矛盾：长行为序列建模和异构非序列特征交互通常被拆成两段，即先用 LONGER 一类序列模型把用户历史压成 query/interest token，再用 RankMixer 一类模块和用户、物品、上下文特征做交互。这个范式有效，但信息流偏晚、偏单向，序列压缩前无法充分利用非序列特征，特征交互也只能看到已经被压缩过的序列表示。

论文提出 HyFormer，把长序列建模和特征交互放进统一的分层 backbone。每层交替执行两个动作：Query Decoding 用全局 query token 去 cross-attend 长行为序列的 K/V 表示；Query Boosting 用类似 MLP-Mixer/RankMixer 的 token mixing 强化 query 之间、序列之间、非序列 token 之间的异构交互。这样 query 不再只是一次性生成的候选物品向量，而是在多层中不断被序列信息和特征交互共同刷新。

实验结论是，在 billion-scale 工业数据和线上 A/B 中，HyFormer 相比 LONGER + RankMixer 有更好的 AUC/业务指标，并且随参数量、FLOPs、序列侧信息维度增长时收益更稳定。

## 详细解读

### 1. 要解决的问题

传统工业 LRM 常见结构是：

1. 序列模块先编码用户长行为历史。
2. 得到少量压缩序列 token。
3. 再和 user/item/context 等非序列 token 做特征交互。

论文认为这带来三类瓶颈：

1. query token 信息不足：很多模型用候选物品或少数全局特征生成 query，无法充分表达复杂场景。
2. 交互太晚：序列被压缩后才和异构特征交互，早期序列表示不能被 item/context/user group 共同塑形。
3. scaling 效率不足：加深序列模块或加宽交互模块主要强化局部子模块，不一定强化联合表示。

### 2. 核心设计：Query Decoding

Query Decoding 的直觉是：不要先把序列独立压完，而是让由非序列特征扩展出来的 global/query tokens 每一层都去解码长序列。

具体地，非序列特征经过语义分组 tokenization 后，被 MLP 生成多个 query token。每个 query 通过 cross-attention 从对应序列的 K/V 中提取信息。到了更深层，query 不重新从头生成，而是使用上一层交互后的 query 继续去问序列，从而形成逐层 refinement。

这和 baseline “生成 query -> 压缩序列 -> 交互”不同。HyFormer 的 query 在层间被不断更新，序列理解与特征交互互相影响。

### 3. 核心设计：Query Boosting

Query Boosting 负责对解码后的 query token 和非序列 token 做高效 token mixing。它的作用类似 RankMixer：用硬件友好的 MLP/token mixing 在小 token 集合上做高阶交互。

重要点不是 token mixing 本身，而是它被放在每个 HyFormer layer 中。也就是说，模型不是只在最后做一次融合，而是反复执行：

Query Decoding：query 从长序列中取信息。
Query Boosting：query 与其他 query、非序列 token 交互。
下一层再用更新后的 query 继续解码序列。

### 4. 对 CTR/PCVR 建模的启发

对当前 PCVRHyFormer baseline 来说，论文最有价值的启发是：

1. 多个 query token 比单个 query 更适合表达多兴趣、多行为域。
2. item/target 信息应参与 query 生成，否则“候选物品相关的历史行为”难以被精准召回。
3. 多行为序列域之间不应完全独立，各域 query 需要交互。
4. 特征交互应分层发生，而不是只在最终拼接后发生。

## 如何检验自己掌握了核心要点

可以用下面问题自测：

1. 能否用一句话解释 HyFormer 相比 LONGER + RankMixer 的结构差异？
2. 能否画出一层 HyFormer 的数据流：NS tokens -> query generation -> query decoding -> query boosting -> next layer？
3. 能否解释为什么“晚融合”会损失信息，而不是只说“Transformer 更强”？
4. 能否区分 Query Decoding 和 Query Boosting 的职责？
5. 能否说明为什么增加 query 数量会带来表达能力与 serving 成本之间的权衡？
6. 能否设计 ablation：关闭 target-aware query、关闭 cross-domain query interaction、减少 query 数量，分别预期会影响什么？

如果你能不看论文回答以上问题，并能把它映射到当前代码中的 `TargetAwareQueryGenerator`、`CrossDomainInteraction`、`MultiSeqHyFormerBlock`，基本就掌握了本文核心。
