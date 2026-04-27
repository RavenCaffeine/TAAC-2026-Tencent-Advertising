# MVP 训练代码对照检查报告

## 检查结论

MVP 相比 `training` 原 baseline 有明确的结构和训练策略改进，但原始入口存在两个会导致改进没有真正生效的问题。我已修复：

1. `mvp/train.py` 原先把 `training` 插到 `sys.path` 最前面，导致实际导入 baseline 的 `model.py/trainer.py`。现在改为 append，让 MVP 目录优先。
2. `mvp/run.sh` 原先调用 `../training/train.py`，不会使用 MVP 新增参数和训练器。现在改为调用 `mvp/train.py`。
3. `--no_target_aware` 原先仍实例化 Target-Aware generator，开关无效。现在补充 baseline-style `MultiSeqQueryGenerator`，可以真实 ablation。

当前环境没有安装 `torch`，因此无法完成真实 forward/backward 或端到端训练；但已完成 Python 语法编译检查：

```bash
python -m py_compile mvp/model.py mvp/trainer.py mvp/train.py training/model.py training/trainer.py training/train.py
```

该检查已通过。要验证完整运行，需要在带 PyTorch 的训练环境中执行 `bash mvp/run.sh ...` 或 `python mvp/train.py ...`。

## 与 baseline 的主要差异

### 1. 模型结构改进

对应文件：`mvp/model.py`

1. Target-Aware Query Generation：`TargetAwareQueryGenerator` 使用 item NS tokens 生成 target gate，对每个序列域的 query token 进行调制。它对应 HyFormer/InterFormer 中“候选物品或非序列上下文应参与序列检索”的思想。
2. Cross-Domain Interaction：`CrossDomainInteraction` 在每层 query decoding 后，把所有行为域的 query token concat 后做 self-attention + FFN，再切回各域。它对应三篇论文共同强调的“多模态/多域双向交互”。
3. 更深更宽默认配置：`d_model=128`、`num_hyformer_blocks=3`、`num_heads=8`，相比 baseline 默认 `d_model=64`、`blocks=2`、`heads=4`，表达能力更强。
4. 输出头加深：MVP classifier 从单层主干变成两层投影与归一化组合，容量更大。

### 2. 训练策略改进

对应文件：`mvp/trainer.py`

1. AMP：CUDA 环境下启用 `GradScaler` 与 `autocast`，与 OneTrans 强调的 mixed precision 工程优化一致。
2. Warmup + Cosine LR：新增 `WarmupCosineScheduler`，比固定 lr 更适合更深模型。
3. Label Smoothing：BCE 标签平滑，降低过拟合和过度自信。
4. Gradient Accumulation：支持更大 effective batch size。
5. Gradient Clipping 参数化：`max_grad_norm` 可配置。

### 3. 与三篇论文的契合度

HyFormer：MVP 已实现 target-aware query、逐层 query decoding + query boosting、跨域 query interaction，方向较契合。但还不是完整 HyFormer，因为 query boosting 仍沿用 RankMixer 风格，且没有论文级别的 serving/KV cache 设计。

InterFormer：MVP 有部分双向交互思想，尤其是 item/context 参与 query 和跨域 query 交互。但它没有完整的 Interaction Arch / Sequence Arch / Cross Arch 三分结构，也没有 PFFN。

OneTrans：MVP 吸收了 mixed precision、更深 Transformer、统一交互的局部思想，但没有统一 token backbone、mixed parameterization、pyramid pruning、cross-request KV cache。

## 是否“有改进”

从代码结构上看：有改进。它不是单纯调参，而是在 baseline 的 HyFormer-like 框架上加入了更贴近论文的 target-aware query 和 cross-domain query interaction。

从可证明的实验结果上看：当前还不能断言线上/离线指标一定提升。需要至少做以下 ablation：

1. baseline：`training/train.py` 原配置。
2. MVP full：`mvp/train.py` 默认配置。
3. MVP no target-aware：加 `--no_target_aware`。
4. MVP no cross-domain：加 `--no_cross_domain`。
5. MVP capacity-only：关闭 target-aware/cross-domain，但保留更大 d_model/blocks，用于区分结构收益和模型容量收益。

建议比较 Valid AUC、LogLoss、训练耗时、显存峰值和收敛轮数。

## 运行检查建议

在带 PyTorch 和数据路径的环境中执行：

```bash
bash mvp/run.sh \
  --data_dir /path/to/train_data \
  --ckpt_dir /path/to/ckpt \
  --log_dir /path/to/log \
  --num_epochs 1 \
  --train_ratio 0.01 \
  --valid_ratio 0.01 \
  --num_workers 0 \
  --eval_every_n_steps 0
```

最小 smoke test 通过标准：

1. 日志中出现 `MVP Args`。
2. 日志中出现 `enable_cross_domain=True` 和 `enable_target_aware=True`。
3. `Total parameters` 大于 baseline 同等小配置。
4. 第一轮训练能完成一次 train step 和一次 validation。
5. `--no_target_aware` 后模型实例化为 `MultiSeqQueryGenerator`，不是 `TargetAwareQueryGenerator`。

## 仍需注意的风险

1. 当前本机 Python 环境缺少 `torch`，未能执行真实训练。
2. `mvp/README.md` 和部分注释存在乱码，不影响运行，但影响协作阅读。
3. AMP 只在 CUDA 下启用，CPU 环境会自动关闭，这是合理行为。
4. Gradient accumulation 在默认 `1` 时无问题；若设置大于 `1`，建议确保 dataloader step 数能被整除，或后续补充 epoch 尾部 flush。
