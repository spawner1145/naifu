# Lumina2 LyCORIS 快速入门

## 1分钟了解LyCORIS

LyCORIS是LoRA的增强版本，特别适合图像生成模型：
- ✅ **更少参数**：比LoRA少30-50%
- ✅ **更好效果**：对视觉模型优化
- ✅ **多种算法**：LoCoN、LoHa、LoKr等

## 快速开始（3步）

### 步骤1：安装

```bash
pip install lycoris_lora toml
```

### 步骤2：配置

```bash
cp config/train_lumina2_lycoris.yaml config/my_config.yaml
```

编辑`my_config.yaml`，修改路径：
```yaml
trainer:
  model_path: /path/to/your/lumina2
  checkpoint_dir: ./my_lycoris_output

dataset:
  index_file: /path/to/your/dataset.json
```

### 步骤3：训练

```bash
python trainer.py --config config/my_config.yaml
```

就这么简单！

## 核心配置说明

### 算法选择

```yaml
lycoris:
  algo: locon    # 推荐！视觉模型最佳
  # algo: loha   # 更少参数
  # algo: lokr   # 最少参数
```

**选择建议**：
- 😊 不确定？用 `locon`
- 💾 显存紧张？用 `lokr`
- ⚡ 追求速度？用 `loha`

### 维度设置

```yaml
lycoris:
  linear_dim: 16      # 主要参数
  linear_alpha: 8     # 通常是dim的一半
```

**维度建议**：
| 场景 | linear_dim | 说明 |
|-----|-----------|------|
| 轻量微调 | 4-8 | 快速、参数少 |
| 标准训练 | 8-16 | 推荐配置 |
| 重度微调 | 16-32 | 效果最好 |

## 常用配置模板

### 模板1：标准配置（推荐新手）

```yaml
lycoris:
  algo: locon
  linear_dim: 16
  linear_alpha: 8
  
trainer:
  batch_size: 2
  accumulate_grad_batches: 8
  max_epochs: 10
  
optimizer:
  params:
    lr: 1e-4
```

**特点**：平衡、稳定、效果好

### 模板2：高效配置（显存有限）

```yaml
lycoris:
  algo: lokr        # 最高效
  linear_dim: 8     # 较小维度
  linear_alpha: 4
  factor: 2         # 小因子
  
trainer:
  batch_size: 1     # 小批次
  accumulate_grad_batches: 16
```

**特点**：显存占用少、速度快

### 模板3：高质量配置（追求效果）

```yaml
lycoris:
  algo: locon
  linear_dim: 32    # 大维度
  linear_alpha: 16
  conv_dim: 32      # 卷积层也大
  conv_alpha: 16
  
trainer:
  batch_size: 4
  max_epochs: 20
  
optimizer:
  params:
    lr: 5e-5        # 较小学习率
```

**特点**：效果最好、训练时间长

## 显存参考

| GPU | 推荐batch_size | 推荐linear_dim | 算法 |
|-----|---------------|---------------|------|
| 8GB | 1 | 8 | lokr |
| 16GB | 2 | 16 | locon |
| 24GB | 4 | 16-32 | locon |
| 40GB+ | 8 | 32 | locon |

## 常见问题速查

### ❌ 显存不足

```yaml
# 方案1：减小维度
linear_dim: 8    # 16 -> 8

# 方案2：换算法
algo: lokr       # locon -> lokr

# 方案3：小批次+累积
batch_size: 1
accumulate_grad_batches: 16
```

### ❌ 训练太慢

```yaml
# 方案1：增大批次
batch_size: 4
accumulate_grad_batches: 4

# 方案2：减少采样
sampling:
  every_n_steps: 5000  # 1000 -> 5000
  
# 方案3：减少workers
dataset:
  num_workers: 4       # 8 -> 4
```

### ❌ 效果不好

```yaml
# 方案1：增大维度
linear_dim: 32       # 16 -> 32
linear_alpha: 16     # 8 -> 16

# 方案2：延长训练
max_epochs: 20       # 10 -> 20

# 方案3：调整学习率
lr: 5e-5            # 1e-4 -> 5e-5
```

## 算法对比速查表

| 算法 | 参数量 | 速度 | 效果 | 推荐场景 |
|-----|--------|-----|------|---------|
| locon | 中 | 中 | 最好 | ⭐ 首选，通用 |
| loha | 少 | 快 | 好 | 速度优先 |
| lokr | 最少 | 最快 | 较好 | 显存受限 |
| dylora | 多 | 慢 | 好 | 探索阶段 |

## 训练检查清单

开始训练前检查：

- [ ] 已安装 `lycoris_lora toml`
- [ ] 配置文件中的路径都正确
- [ ] 数据集索引文件存在
- [ ] 显存足够（参考上表）
- [ ] 设置了合适的batch_size
- [ ] 选择了合适的算法和维度

训练过程中监控：

- [ ] loss在下降
- [ ] 生成的样本质量在提升
- [ ] 显存没有溢出
- [ ] 训练速度正常

## 从LoRA迁移

如果你之前用过LoRA：

```yaml
# LoRA配置
lora_params:
  r: 16
  lora_alpha: 16

# 等价的LyCORIS配置
lycoris:
  algo: locon
  linear_dim: 16
  linear_alpha: 16
```

主要区别：
- LyCORIS支持卷积层
- 可以选择不同算法
- 参数量可以更少

## 下一步

✅ **基础使用**：
- 阅读 `LUMINA2_LYCORIS_README.md` 了解详细信息
- 尝试不同的算法和维度

✅ **进阶优化**：
- 查看 `config/train_lumina2_lycoris_full.yaml` 所有参数
- 学习各算法的原理和适用场景

✅ **问题排查**：
- 参考 README 中的故障排除章节
- 加入社区讨论

## 完整示例

```bash
# 1. 安装
pip install lycoris_lora toml

# 2. 准备配置
cp config/train_lumina2_lycoris.yaml config/test.yaml

# 编辑 test.yaml，设置你的路径...

# 3. 小规模测试（2 epochs）
# 修改配置：max_epochs: 2
python trainer.py --config config/test.yaml

# 4. 检查结果
# 查看 samples/ 目录下的生成图片
# 查看 wandb 上的训练曲线

# 5. 如果效果OK，正式训练
# 修改配置：max_epochs: 10-20
python trainer.py --config config/test.yaml
```

## 性能预期

使用推荐配置（locon, dim=16），在典型数据集上：

- **训练速度**：比全量训练快2-3倍
- **显存占用**：降低50%左右
- **权重大小**：约60MB（vs 全量8GB）
- **训练效果**：接近全量训练（95%+）

## 小贴士

💡 **第一次使用**：
- 从小维度开始（dim=8）
- 先训练2-3个epochs
- 查看生成效果再决定是否继续

💡 **显存优化**：
- lokr > loha > locon（参数量）
- 减小dim比减小batch_size更有效

💡 **效果优化**：
- locon效果最好
- 适当增大dim（16->32）
- 延长训练时间

💡 **速度优化**：
- 增大batch_size
- 减少sampling频率
- 使用loha或lokr

就是这样！现在你可以开始训练了 🚀

有问题？查看完整文档：`LUMINA2_LYCORIS_README.md`
