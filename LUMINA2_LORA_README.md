# Lumina2 LoRA Training Guide

本文档介绍如何使用LoRA（Low-Rank Adaptation）方式训练Lumina2模型。

## 概述

LoRA是一种高效的模型微调方法，通过在模型的注意力层添加低秩矩阵来实现参数高效的训练。相比全量微调，LoRA具有以下优势：

- **内存效率高**：只训练少量参数（通常<1%的模型参数）
- **训练速度快**：更少的参数需要更新
- **易于部署**：LoRA权重可以独立保存和加载
- **防止过拟合**：参数量小，更不容易过拟合

## 文件说明

- `modules/lumina2_model_train_lora.py`: LoRA训练模块实现
- `config/train_lumina2_lora.yaml`: LoRA训练配置文件示例

## 配置说明

### LoRA参数配置

```yaml
use_lora: true
lora_params:
  r: 16                    # LoRA rank，控制低秩矩阵的秩，越大表达能力越强但参数越多
  lora_alpha: 16           # LoRA alpha，缩放因子，通常设置为与r相同
  lora_dropout: 0.0        # LoRA层的dropout率
  bias: "none"             # 是否训练bias，可选："none", "all", "lora_only"
  target_modules:          # 目标模块，指定哪些层应用LoRA
    - "to_q"               # Query投影层
    - "to_k"               # Key投影层
    - "to_v"               # Value投影层
    - "to_out.0"           # 输出投影层
```

### 重要配置项

```yaml
advanced:
  use_ema: false           # LoRA训练通常不需要EMA
  train_text_encoder: false  # 是否同时训练text encoder
  text_encoder_lr: 5e-5    # text encoder的学习率（如果训练）

optimizer:
  params:
    lr: 1e-4               # LoRA可以使用比全量训练更高的学习率
```

## 使用方法

### 1. 安装依赖

确保已安装peft库：

```bash
pip install peft
```

### 2. 准备配置文件

复制并修改`config/train_lumina2_lora.yaml`：

```bash
cp config/train_lumina2_lora.yaml config/my_lumina2_lora.yaml
```

根据你的需求修改配置：
- 调整`model_path`和相关路径
- 设置`dataset.index_file`指向你的数据集
- 调整`lora_params`中的参数
- 设置`checkpoint_dir`保存路径

### 3. 开始训练

使用trainer.py启动训练：

```bash
python trainer.py --config config/my_lumina2_lora.yaml
```

### 4. 检查训练进度

训练过程中会：
- 自动保存检查点到`checkpoint_dir`
- 生成样本图片到`sampling.save_dir`（如果启用）
- 记录训练日志到wandb（如果配置了wandb_id）

## 参数调优建议

### LoRA Rank (r)

- **r=4-8**: 适合简单风格迁移或微调
- **r=16-32**: 适合一般场景，平衡性能和参数量
- **r=64+**: 适合复杂任务，但接近全量训练的参数量

### 学习率

- LoRA通常可以使用比全量训练更高的学习率
- 建议范围：`1e-4` 到 `5e-4`
- 如果训练不稳定，可以降低到`5e-5`

### Batch Size

- LoRA训练内存占用更少，可以适当增大batch size
- 建议根据GPU显存调整：
  - 8GB: batch_size=1-2
  - 16GB: batch_size=2-4
  - 24GB: batch_size=4-8

### Target Modules

默认配置针对注意力层：
```python
["to_q", "to_k", "to_v", "to_out.0"]
```

如果需要更强的表达能力，可以添加更多层：
```python
["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
```

## 权重保存与加载

### 保存的文件

训练完成后会保存：
- `checkpoint-eXX_sYYYYY.safetensors`: LoRA权重
- `checkpoint-eXX_sYYYYY_text_encoder.safetensors`: Text encoder权重（如果训练了）

### 加载权重进行推理

```python
from peft import PeftModel
from modules.lumina2_model import Lumina2Model

# 加载基础模型
base_model = Lumina2Model(...)

# 加载LoRA权重
model = PeftModel.from_pretrained(base_model.model, "path/to/lora/checkpoint")
```

## 与全量训练的对比

| 特性 | 全量训练 | LoRA训练 |
|------|---------|---------|
| 训练参数量 | 100% | <1% |
| 内存占用 | 高 | 低 |
| 训练速度 | 慢 | 快 |
| 过拟合风险 | 高 | 低 |
| 表达能力 | 最强 | 较强 |
| 权重大小 | 数GB | 数MB |

## 故障排除

### 1. 导入错误

```
ImportError: cannot import name 'LoraConfig' from 'peft'
```

解决方案：安装或更新peft
```bash
pip install -U peft
```

### 2. 内存不足

- 减小batch_size
- 减小LoRA rank (r)
- 启用gradient checkpointing（配置中已默认支持）

### 3. 训练不收敛

- 降低学习率
- 增加warmup步数
- 检查数据集质量
- 增加LoRA rank

### 4. 生成质量不佳

- 增加训练步数
- 增加LoRA rank和alpha
- 添加更多target_modules
- 考虑训练text_encoder

## 参考资源

- [PEFT官方文档](https://huggingface.co/docs/peft)
- [LoRA论文](https://arxiv.org/abs/2106.09685)
- Lumina2原始论文和文档

## 代码实现说明

### 核心类：SupervisedFineTuneLoRA

继承自`Lumina2Model`，主要特点：

1. **init_lora()**: 初始化LoRA适配器
   - 使用PEFT库的`get_peft_model`
   - 冻结基础模型参数
   - 只训练LoRA参数

2. **forward()**: 前向传播
   - 与原始训练流程相同
   - 使用`self.lora_model`替代`self.model`

3. **save_checkpoint()**: 保存检查点
   - 只保存LoRA参数（显著减小文件大小）
   - 支持safetensors格式

4. **load_checkpoint()**: 加载检查点
   - 智能识别LoRA权重和基础模型权重
   - 支持从预训练检查点恢复训练

### 与其他模块的一致性

代码风格保持与以下模块一致：
- `train_general_llm.py`: LoRA初始化方式
- `train_lycoris.py`: 参数优化器设置
- `lumina2_model_train.py`: 训练流程和forward实现

所有改动都遵循项目的编码规范和设计模式。
