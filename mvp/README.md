# MVP 改进版 Baseline

## 改进清单

1. **混合精度训练 (AMP)**：GradScaler + autocast，训练速度提升 ~1.5x
2. **学习率调度器**：Warmup + CosineAnnealing
3. **Target-Aware Attention**：在 Query 生成中引入候选物品信息
4. **跨域序列交互层**：多域 Q tokens 交叉注意力
5. **Label Smoothing**：缓解过拟合
6. **梯度累积**：支持更大等效 batch_size
7. **更大默认模型容量**：d_model=128, 3层 blocks

## 文件结构

- `model.py` — 改进后的模型（Target-Aware + 跨域交互）
- `trainer.py` — 改进后的训练器（AMP + LR Schedule + Label Smoothing + 梯度累积）
- `run.sh` — 改进后的启动脚本

## 使用方式

将 `dataset.py` 和 `utils.py` 从 `training/` 复制过来，然后运行：
```bash
cp ../training/dataset.py .
cp ../training/utils.py .
cp ../training/train.py .  # 或使用改进后的 train.py
bash run.sh
```
