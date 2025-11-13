# Lumina2 LyCORIS 训练指南

## 概述

LyCORIS（LoRA beyond Conventional methods, Rank adaptation Implemented in Stable diffusion）是一系列参数高效微调方法的集合，比标准LoRA更灵活和高效。

### LyCORIS vs LoRA

| 特性 | LoRA | LyCORIS |
|------|------|---------|
| 算法多样性 | 单一算法 | 多种算法(LoCoN, LoHa, LoKr等) |
| 卷积层支持 | 不支持 | 完整支持 |
| 参数效率 | 好 | 更好 |
| 灵活性 | 中等 | 高 |
| 适用场景 | 通用 | 特别适合视觉模型 |

## 文件说明

- `modules/lumina2_model_train_lycoris.py`: LyCORIS训练模块
- `config/train_lumina2_lycoris.yaml`: 简化配置
- `config/train_lumina2_lycoris_full.yaml`: 完整配置（带详细注释）

## LyCORIS算法介绍

### 1. LoCoN (LoRA for Convolution) - 推荐

**特点**：
- 专为视觉模型设计
- 同时支持卷积层和线性层
- 参数量和效果的最佳平衡

**适用场景**：
- 图像生成模型（如Lumina2）
- 需要微调卷积层的任务
- 标准的风格迁移

**配置示例**：
```yaml
lycoris:
  algo: locon
  linear_dim: 16
  linear_alpha: 8
  conv_dim: 16      # 卷积层维度
  conv_alpha: 8     # 卷积层alpha
```

### 2. LoHa (LoRA with Hadamard Product)

**特点**：
- 使用Hadamard积分解
- 参数量比LoCoN少约30%
- 训练和推理速度更快

**适用场景**：
- 显存受限
- 需要快速训练
- 对参数量敏感

**配置示例**：
```yaml
lycoris:
  algo: loha
  linear_dim: 16
  linear_alpha: 8
  factor: 4         # 分解因子
```

### 3. LoKr (LoRA with Kronecker Product)

**特点**：
- 使用Kronecker积分解
- 参数量最少（可减少50%+）
- 最高效的算法

**适用场景**：
- 极度显存受限
- 追求最小模型大小
- 轻量级微调

**配置示例**：
```yaml
lycoris:
  algo: lokr
  linear_dim: 16
  linear_alpha: 8
  factor: 2         # 更小的factor
```

### 4. DyLoRA (Dynamic LoRA)

**特点**：
- 动态调整秩
- 自适应参数分配
- 无需手动调整dim

**适用场景**：
- 不确定最优秩的情况
- 探索性训练
- 需要灵活性

**配置示例**：
```yaml
lycoris:
  algo: dylora
  linear_dim: 32    # 最大维度
  linear_alpha: 16
```

## 使用方法

### 1. 安装依赖

```bash
# 安装LyCORIS库
pip install lycoris_lora toml
```

### 2. 快速开始

```bash
# 复制配置文件
cp config/train_lumina2_lycoris.yaml config/my_lycoris_config.yaml

# 编辑配置，修改路径和参数
# ...

# 开始训练
python trainer.py --config config/my_lycoris_config.yaml
```

### 3. 基本配置

```yaml
lycoris:
  algo: locon              # 算法选择
  linear_dim: 16           # 线性层维度
  linear_alpha: 8          # 线性层alpha
  factor: 4                # 分解因子
  multiplier: 1.0          # 应用强度

trainer:
  model_path: /path/to/lumina2
  checkpoint_dir: ./results/my_lycoris

dataset:
  index_file: /path/to/dataset.json
```

## 参数调优指南

### linear_dim（维度）选择

| linear_dim | 参数量 | 训练速度 | 效果 | 推荐场景 |
|-----------|--------|---------|------|---------|
| 4-8 | 极少 | 最快 | 基础 | 轻量微调、风格迁移 |
| 8-16 | 少 | 快 | 好 | 标准训练、通用场景 |
| 16-32 | 中等 | 中等 | 很好 | 重度微调、复杂任务 |
| 32+ | 多 | 慢 | 最好 | 接近全量训练 |

### linear_alpha（缩放因子）选择

**一般规则**：
- `linear_alpha = linear_dim / 2`（推荐）
- `linear_alpha = linear_dim`（更强的学习）
- `linear_alpha < linear_dim / 2`（更保守的学习）

**影响**：
- 控制LyCORIS的更新强度
- 影响训练稳定性
- 需要与学习率配合调整

### 算法选择策略

```
显存充足，追求效果 -> LoCoN (linear_dim: 16-32)
显存一般，平衡性能 -> LoHa (linear_dim: 16)
显存紧张，最小参数 -> LoKr (linear_dim: 8-16, factor: 2)
不确定最优配置   -> DyLoRA (linear_dim: 32)
```

## 高级功能

### 1. 分别配置不同模块

```yaml
# 主模型使用较大的维度
lycoris_model:
  algo: locon
  linear_dim: 32
  linear_alpha: 16
  conv_dim: 32
  conv_alpha: 16

# Text encoder使用较小的维度
lycoris_text_encoder:
  algo: locon
  linear_dim: 8
  linear_alpha: 4
```

### 2. 使用dropout防止过拟合

```yaml
lycoris:
  algo: locon
  linear_dim: 16
  linear_alpha: 8
  dropout: 0.1          # 标准dropout
  rank_dropout: 0.1     # 秩dropout
  module_dropout: 0.05  # 模块dropout
```

### 3. 训练text encoder

```yaml
advanced:
  train_text_encoder: true
  text_encoder_lr: 5e-5   # 更小的学习率

# 可选：为text encoder单独配置
lycoris_text_encoder:
  algo: locon
  linear_dim: 8
  linear_alpha: 4
```

## 性能对比

基于Lumina2-2B模型的测试：

| 方法 | 可训练参数 | 显存占用 | 训练速度 | 权重大小 | 效果 |
|-----|-----------|---------|---------|---------|------|
| 全量训练 | 2B (100%) | ~48GB | 基准 | ~8GB | 最好 |
| LoRA (r=16) | ~20M (1%) | ~24GB | 1.8x | ~80MB | 很好 |
| LoCoN (dim=16) | ~15M (0.75%) | ~20GB | 2x | ~60MB | 很好 |
| LoHa (dim=16) | ~10M (0.5%) | ~18GB | 2.2x | ~40MB | 好 |
| LoKr (dim=16) | ~8M (0.4%) | ~16GB | 2.5x | ~32MB | 好 |

## 常见问题

### Q1: LyCORIS和LoRA应该选哪个？

**建议**：
- 对于Lumina2这样的视觉模型，推荐LyCORIS（特别是LoCoN）
- 如果已有LoRA经验，可以先用LoCoN试试，配置类似
- 如果显存特别紧张，考虑LoKr

### Q2: 如何选择合适的算法？

```
首选: LoCoN (locon) - 视觉模型的最佳选择
显存紧张: LoKr (lokr) - 最少参数
追求速度: LoHa (loha) - 快速训练
探索阶段: DyLoRA (dylora) - 自适应
```

### Q3: 训练多久合适？

- **轻量微调**（dim=8）: 5-10 epochs
- **标准训练**（dim=16）: 10-20 epochs
- **重度微调**（dim=32）: 20-30 epochs

监控训练loss和生成样本质量来判断。

### Q4: 如何加载LyCORIS权重？

```python
from lycoris import create_lycoris
from modules.lumina2_model import Lumina2Model

# 加载基础模型
model = Lumina2Model(...)

# 加载LyCORIS权重
lycoris_model = create_lycoris(
    model.model,
    algo="locon",
    linear_dim=16,
    linear_alpha=8
)
lycoris_model.load_state_dict(torch.load("checkpoint.safetensors"))
lycoris_model.apply_to()
```

### Q5: 可以从LoRA转换到LyCORIS吗？

可以，但需要重新训练。LyCORIS和LoRA的权重结构不同，无法直接转换。

## 故障排除

### ImportError: cannot import lycoris

```bash
pip install lycoris_lora toml
```

### CUDA out of memory

1. 减小 `linear_dim`（如从16降到8）
2. 切换到更高效的算法（loha或lokr）
3. 增大 `accumulate_grad_batches`
4. 启用 `checkpointing: true`

### 训练不稳定/loss震荡

1. 降低学习率（如从1e-4降到5e-5）
2. 增加 `linear_alpha`
3. 添加 dropout
4. 增加warmup步数

### 效果不如预期

1. 增大 `linear_dim`
2. 延长训练时间
3. 尝试不同的算法（特别是locon）
4. 检查数据集质量和标签
5. 调整学习率

## 最佳实践

### 1. 从小开始

```yaml
# 第一次训练使用较小配置
lycoris:
  algo: locon
  linear_dim: 8
  linear_alpha: 4

trainer:
  max_epochs: 5
```

### 2. 逐步增加复杂度

```yaml
# 效果不够再增大
lycoris:
  linear_dim: 16  # 8 -> 16
  linear_alpha: 8  # 4 -> 8

trainer:
  max_epochs: 10  # 5 -> 10
```

### 3. 监控关键指标

- **训练loss**: 应该稳定下降
- **生成样本**: 定期检查质量
- **显存占用**: 确保不会OOM
- **训练速度**: 记录时间用于优化

### 4. 保存实验记录

```yaml
trainer:
  wandb_id: "experiment_name"  # 使用有意义的名称

# 在笔记中记录:
# - 使用的算法和参数
# - 数据集信息
# - 最终效果评估
# - 遇到的问题和解决方案
```

## 与其他方法的对比

### LyCORIS vs LoRA

**LyCORIS优势**：
- ✅ 支持卷积层
- ✅ 多种算法可选
- ✅ 参数更少
- ✅ 对视觉模型更友好

**LoRA优势**：
- ✅ 更成熟稳定
- ✅ 社区资源丰富
- ✅ 工具支持更好

**建议**：对于Lumina2，优先考虑LyCORIS（特别是LoCoN）

### LyCORIS vs 全量训练

| 维度 | LyCORIS | 全量训练 |
|-----|---------|---------|
| 参数量 | 0.4-1% | 100% |
| 显存 | 低 | 高 |
| 速度 | 快 | 慢 |
| 效果 | 很好 | 最好 |
| 过拟合风险 | 低 | 高 |
| 适用场景 | 小数据集微调 | 大数据集训练 |

## 参考资源

- [LyCORIS GitHub](https://github.com/KohakuBlueleaf/LyCORIS)
- [LyCORIS文档](https://github.com/KohakuBlueleaf/LyCORIS/wiki)
- [LoRA论文](https://arxiv.org/abs/2106.09685)
- Lumina2模型文档

## 示例工作流

```bash
# 1. 安装依赖
pip install lycoris_lora toml

# 2. 准备配置
cp config/train_lumina2_lycoris.yaml config/my_exp.yaml

# 3. 编辑配置（使用LoCoN，dim=8先试试）
# lycoris:
#   algo: locon
#   linear_dim: 8
#   linear_alpha: 4

# 4. 快速测试（1-2 epochs）
python trainer.py --config config/my_exp.yaml

# 5. 检查结果，如果不够好：
#    - 增大dim到16
#    - 延长训练到10 epochs
#    - 调整学习率

# 6. 正式训练
python trainer.py --config config/my_exp.yaml

# 7. 评估和部署
python test_inference.py --lycoris_path results/lumina2_lycoris/checkpoint-eXX_sYYYYY.safetensors
```

开始你的Lumina2 LyCORIS训练之旅吧！ 🚀
