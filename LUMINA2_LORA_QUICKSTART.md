# Lumina2 LoRA 训练快速入门

## 1. 环境准备

### 安装依赖

```bash
# 安装PEFT库（用于LoRA）
pip install peft

# 如果还没有安装其他依赖
pip install -r requirements.txt
```

### 验证安装

```bash
python test_lumina2_lora.py
```

如果所有测试都通过，说明环境配置正确。

## 2. 准备数据

确保你的数据集已经准备好，格式与Lumina2训练一致：
- 图像数据
- 对应的文本描述
- 数据集索引JSON文件

## 3. 配置训练参数

### 快速开始（使用简化配置）

复制并编辑简化配置：

```bash
cp config/train_lumina2_lora.yaml config/my_lora_config.yaml
```

修改关键参数：

```yaml
# 修改路径
trainer:
  model_path: /path/to/your/lumina2/model
  checkpoint_dir: ./my_lora_output
  
dataset:
  index_file: /path/to/your/dataset.json

# 调整LoRA参数（可选）
lora_params:
  r: 16              # rank，可以尝试8, 16, 32
  lora_alpha: 16     # 通常与r相同
```

### 完整配置（高级用户）

使用完整配置文件获得更多控制：

```bash
cp config/train_lumina2_lora_full.yaml config/my_lora_config.yaml
```

详细参数说明请参考配置文件中的注释。

## 4. 开始训练

### 单GPU训练

```bash
python trainer.py --config config/my_lora_config.yaml
```

### 多GPU训练（推荐）

配置文件中设置：

```yaml
lightning:
  devices: 8  # 使用8张GPU
  strategy: "deepspeed"
```

然后运行：

```bash
python trainer.py --config config/my_lora_config.yaml
```

## 5. 监控训练

### 使用WandB（推荐）

在配置文件中设置：

```yaml
trainer:
  wandb_id: "my_project_name"
```

然后登录WandB查看训练曲线和生成的样本。

### 查看本地日志

训练日志会保存在终端输出和日志文件中。

### 查看生成的样本

如果启用了采样，样本图片会保存在：

```
sampling.save_dir  # 默认: samples/lumina2_lora/
```

## 6. 常见参数调整

### 显存不足

```yaml
trainer:
  batch_size: 1                    # 减小batch size
  accumulate_grad_batches: 16      # 增加梯度累积
  checkpointing: true              # 启用gradient checkpointing

lora_params:
  r: 8                            # 减小LoRA rank
```

### 训练速度慢

```yaml
trainer:
  batch_size: 4                   # 增大batch size
  use_xformers: true              # 使用xformers加速

dataset:
  num_workers: 4                  # 减少dataloader workers

sampling:
  enabled: false                  # 禁用采样
  # 或减少采样频率
  every_n_steps: 5000
```

### 提升训练效果

```yaml
lora_params:
  r: 32                           # 增大rank
  target_modules:                 # 添加更多层
    - "to_q"
    - "to_k"
    - "to_v"
    - "to_out.0"
    - "ff.net.0.proj"
    - "ff.net.2"

optimizer:
  params:
    lr: 2e-4                      # 调整学习率

trainer:
  max_epochs: 50                  # 延长训练
```

## 7. 检查点和恢复训练

### 自动保存检查点

检查点会自动保存到：

```
checkpoint_dir/checkpoint-eXX_sYYYYY.safetensors
```

### 从检查点恢复

在配置文件中设置：

```yaml
trainer:
  resume: true
```

或者：

```yaml
model:
  resume: /path/to/checkpoint-eXX_sYYYYY.safetensors
```

## 8. 使用训练好的LoRA

训练完成后，你会得到LoRA权重文件（通常几十MB）。

### 加载方法

```python
from peft import PeftModel
from modules.lumina2_model import Lumina2Model

# 加载基础模型
config = {...}  # 你的配置
base_model = Lumina2Model(config, device, model_path)

# 加载LoRA权重
lora_model = PeftModel.from_pretrained(
    base_model.model, 
    "/path/to/checkpoint-eXX_sYYYYY"
)

# 现在可以使用lora_model进行推理
```

## 9. 性能对比

| 指标 | 全量微调 | LoRA (r=16) | LoRA (r=8) |
|------|---------|------------|-----------|
| 可训练参数 | ~2B | ~20M (1%) | ~10M (0.5%) |
| 显存占用 | 高 | 中 | 低 |
| 训练速度 | 慢 | 快 | 更快 |
| 权重大小 | ~8GB | ~80MB | ~40MB |
| 效果 | 最好 | 很好 | 好 |

## 10. 疑难解答

### ImportError: cannot import name 'LoraConfig'

```bash
pip install peft
```

### CUDA out of memory

- 减小 `batch_size`
- 增大 `accumulate_grad_batches`
- 启用 `checkpointing: true`
- 减小 `lora_params.r`

### 训练损失不下降

- 检查数据质量
- 增加学习率
- 增大 `lora_params.r`
- 检查数据集路径是否正确

### 生成质量不佳

- 延长训练时间
- 增大 `lora_params.r` 和 `lora_alpha`
- 添加更多 `target_modules`
- 检查采样配置

## 11. 更多资源

- 完整文档：`LUMINA2_LORA_README.md`
- 配置示例：`config/train_lumina2_lora.yaml`
- 详细配置：`config/train_lumina2_lora_full.yaml`
- 测试脚本：`test_lumina2_lora.py`

## 12. 最佳实践

1. **从小开始**：先用小的rank (r=8)和短时间训练验证流程
2. **监控指标**：关注train_loss和生成的样本质量
3. **定期保存**：设置合理的checkpoint_freq和checkpoint_steps
4. **版本控制**：保存每个实验的配置文件
5. **数据质量**：高质量数据比大参数更重要

## 13. 示例工作流

```bash
# 1. 验证环境
python test_lumina2_lora.py

# 2. 准备配置
cp config/train_lumina2_lora.yaml config/my_experiment.yaml
# 编辑 my_experiment.yaml

# 3. 开始训练
python trainer.py --config config/my_experiment.yaml

# 4. 监控训练（另一个终端）
watch -n 10 'ls -lh results/lumina2_lora/'

# 5. 训练完成后测试
python test_inference.py --lora_path results/lumina2_lora/checkpoint-eXX_sYYYYY.safetensors
```

开始你的Lumina2 LoRA训练之旅吧！ 🚀
