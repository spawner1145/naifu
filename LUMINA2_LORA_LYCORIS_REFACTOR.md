# Lumina2 LoRA/LyCORIS训练实现重构说明

## 重构日期
2025年11月12日

## 重构目标
将Lumina2的LoRA和LyCORIS训练实现改为参考SDXL的成熟代码风格，同时确保训练的是Lumina2模型。

---

## 主要改进

### 1. **代码风格统一** ✅

#### 参考SDXL LyCORIS实现 (`modules/train_lycoris.py`)
- **setup函数结构**: 采用与SDXL相同的函数组织方式
- **参数收集方式**: 使用SDXL的参数优化器构建模式
- **模型setup顺序**: 遵循SDXL的setup调用顺序
- **日志输出风格**: 保持与SDXL一致的日志格式

#### 代码对比示例

**重构前（原始实现）:**
```python
params_to_optim = [{"params": [p for p in model.lora_model.parameters() if p.requires_grad]}]

if config.advanced.get("train_text_encoder"):
    lr = config.advanced.get("text_encoder_lr", config.optimizer.params.lr)
    text_encoder_params = [p for p in model.text_encoder.parameters() if p.requires_grad]
    if text_encoder_params:
        params_to_optim.append({"params": text_encoder_params, "lr": lr})
```

**重构后（参考SDXL）:**
```python
params_to_optim = [{"params": model.lora_model.parameters()}]

if config.advanced.get("train_text_encoder"):
    if hasattr(config.optimizer.params, 'lr'):
        lr = config.advanced.get("text_encoder_lr", config.optimizer.params.lr)
        params_to_optim.append({"params": model.lora_text_encoder.parameters(), "lr": lr})
    else:
        params_to_optim.append({"params": model.lora_text_encoder.parameters()})
```

---

### 2. **修复的核心问题** 🔧

#### 问题1: LoRA模型setup和forward标记问题
**原问题:**
```python
# 问题代码
model.lora_model, optimizer = fabric.setup(model.lora_model, optimizer)

if hasattr(model.lora_model.get_base_model(), "mark_forward_method"):
    model.lora_model.get_base_model().mark_forward_method("forward_with_cfg")
```

**修复方案:**
```python
# 简化setup，移除不必要的forward标记
# PEFT包装的模型在setup后会自动处理forward调用
model.lora_model, optimizer = fabric.setup(model.lora_model, optimizer)

if config.advanced.get("train_text_encoder"):
    model.lora_text_encoder = fabric.setup(model.lora_text_encoder)
```

#### 问题2: Text Encoder的LoRA支持
**改进:**
- 为text encoder单独创建LoRA适配器
- 统一管理模型和text encoder的LoRA映射
- 在保存和加载时正确处理两个模块

```python
# 初始化text encoder的LoRA
if cfg.advanced.get("train_text_encoder"):
    logger.info("")
    logger.info(f"Initializing model.lora_text_encoder with config: {default_lora_config}")
    text_encoder_lora_config = LoraConfig(**default_lora_config)
    self.lora_text_encoder = get_peft_model(self.text_encoder, text_encoder_lora_config)
    self.lora_text_encoder.print_trainable_parameters()
    lora_mapping["text_encoder"] = self.lora_text_encoder
```

---

### 3. **类命名规范化** 📝

**重构前:**
- `SupervisedFineTuneLoRA`
- `SupervisedFineTuneLyCORIS`

**重构后:**
- `Lumina2ModelLoRA` - 清晰表明这是Lumina2的LoRA实现
- `Lumina2ModelLyCORIS` - 清晰表明这是Lumina2的LyCORIS实现

**好处:**
- 类名更清晰地表明用途
- 与其他模型的命名风格一致
- 便于代码搜索和维护

---

### 4. **训练逻辑保持一致** ✅

#### LoRA训练
```python
def forward(self, batch):
    """使用LoRA模型进行训练"""
    # ... 数据处理相同 ...
    
    # 关键：使用lora_model替代model
    loss_dict = trans.training_losses(self.lora_model, latents, model_kwargs)
    
    # ... 损失计算相同 ...
```

#### LyCORIS训练
```python
def forward(self, batch):
    """LyCORIS已经apply_to原始模型"""
    # ... 数据处理相同 ...
    
    # 关键：LyCORIS修改了原始模型，直接使用self.model
    loss_dict = trans.training_losses(self.model, latents, model_kwargs)
    
    # ... 损失计算相同 ...
```

**核心区别说明:**
- **LoRA**: PEFT创建wrapper，需要使用`lora_model`
- **LyCORIS**: 直接修改原模型权重，使用`model`

---

### 5. **权重保存和加载改进** 💾

#### 统一的命名约定
```python
@rank_zero_only
def save_checkpoint(self, model_path, metadata):
    """参考SDXL LyCORIS的保存逻辑"""
    state_dict = {}
    
    # 为每个模块添加统一前缀
    for key, module in self.lora_mapping.items():
        module_state_dict = module.state_dict()
        new_state_dict = {}
        for k, v in module_state_dict.items():
            k = k.replace("module.", "")
            k = k.replace("lycoris_", "")  # LyCORIS使用
            # 或 不需要额外处理 (LoRA使用)
            k = f"lora_{key}_{k}"
            new_state_dict[k] = v
        state_dict.update(new_state_dict)
```

#### 加载时的正确处理
```python
def load_checkpoint(self, state_dict):
    """支持从检查点恢复训练"""
    sd = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    
    # 分离不同模块
    model_lora_sd = {}
    text_encoder_lora_sd = {}
    
    for key in list(sd.keys()):
        if key.startswith("lora_model_"):
            model_lora_sd[key.replace("lora_model_", "")] = sd.pop(key)
        elif key.startswith("lora_text_encoder_"):
            text_encoder_lora_sd[key.replace("lora_text_encoder_", "")] = sd.pop(key)
```

---

### 6. **改进的日志和调试信息** 📊

```python
def init_lora(self):
    """初始化LoRA适配器"""
    # ... 配置准备 ...
    
    # 详细的初始化日志
    logger.info("")
    logger.info(f"Initializing model.lora_model with config: {default_lora_config}")
    lora_config = LoraConfig(**default_lora_config)
    self.lora_model = get_peft_model(self.model, lora_config)
    self.lora_model.print_trainable_parameters()  # PEFT内置的参数统计
    
    # 验证可训练参数
    trainable = sum(p.numel() for p in self.lora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in self.model.parameters())
    if trainable == 0:
        raise RuntimeError("No trainable parameters found after LoRA initialization!")
    logger.info(f"Total parameters: {total:,}")
    logger.info(f"Trainable LoRA parameters: {trainable:,} ({trainable/total*100:.2f}%)")
```

---

## 代码结构对比

### LoRA实现 (`lumina2_model_train_lora.py`)

```
导入部分
├── 标准库
├── PyTorch和Lightning
├── 项目工具类
├── Lumina2基础类
└── PEFT库

setup函数
├── 创建Lumina2ModelLoRA实例
├── 初始化数据集和dataloader
├── 收集可训练参数（参考SDXL风格）
├── 创建优化器和调度器
├── 加载检查点（如果需要）
├── Setup模型和优化器
└── 返回所有组件

Lumina2ModelLoRA类
├── init_model(): 初始化基础模型+LoRA
├── init_lora(): 应用LoRA适配器
│   ├── 配置LoRA参数
│   ├── 冻结基础模型
│   ├── 创建lora_model
│   ├── 创建lora_text_encoder（可选）
│   └── 验证参数
├── forward(): 使用lora_model训练
├── load_checkpoint(): 加载LoRA权重
├── save_checkpoint(): 保存LoRA权重
└── generate_samples(): 生成样本图像
```

### LyCORIS实现 (`lumina2_model_train_lycoris.py`)

```
导入部分
├── 标准库
├── PyTorch和Lightning
├── 项目工具类
├── Lumina2基础类
└── LyCORIS库

setup函数
├── 创建Lumina2ModelLyCORIS实例
├── 初始化数据集和dataloader
├── 收集可训练参数（参考SDXL风格）
├── 创建优化器和调度器
├── 加载检查点（如果需要）
├── Setup lycoris_model和优化器
└── 返回所有组件

Lumina2ModelLyCORIS类
├── init_model(): 初始化基础模型+LyCORIS
├── init_lycoris(): 应用LyCORIS适配器
│   ├── 配置LyCORIS参数
│   ├── 冻结基础模型
│   ├── Apply preset
│   ├── 创建lycoris_model并apply_to()
│   ├── 创建lycoris_text_encoder（可选）
│   └── 统计参数信息
├── forward(): 使用原model训练（LyCORIS已应用）
├── load_checkpoint(): 加载LyCORIS权重
├── save_checkpoint(): 保存LyCORIS权重
└── generate_samples(): 生成样本图像
```

---

## 与SDXL实现的对应关系

| 组件 | SDXL LyCORIS | Lumina2 LoRA | Lumina2 LyCORIS |
|------|--------------|--------------|-----------------|
| 基础类 | `SupervisedFineTune` | `SupervisedFineTune` | `SupervisedFineTune` |
| 微调类 | `StableDiffusionModel` | `Lumina2ModelLoRA` | `Lumina2ModelLyCORIS` |
| 主模型适配器 | `lycoris_unet` | `lora_model` | `lycoris_model` |
| TE1适配器 | `lycoris_te1` | N/A | N/A |
| TE2适配器 | `lycoris_te2` | N/A | N/A |
| 通用TE适配器 | N/A | `lora_text_encoder` | `lycoris_text_encoder` |
| 映射字典 | `lycoris_mapping` | `lora_mapping` | `lycoris_mapping` |
| Setup方式 | setup lycoris_unet | setup lora_model | setup lycoris_model |
| Forward模型 | `model.diffusion_model` | `lora_model` | `model` |

---

## 关键技术点

### 1. LoRA vs LyCORIS的本质区别

**LoRA (PEFT实现):**
- 创建一个wrapper模型 (`PeftModel`)
- 原始模型不变，LoRA参数作为额外层
- 前向传播：`PeftModel.forward()` → 自动注入LoRA
- 需要使用 `lora_model` 进行训练和推理

**LyCORIS:**
- 直接修改原始模型的某些层
- 通过 `apply_to()` 将LyCORIS参数注入到原模型
- 前向传播：直接使用原 `model.forward()`
- 训练时使用原 `model`，LyCORIS参数已经在其中

### 2. 为什么使用lora_mapping/lycoris_mapping

```python
self.lora_mapping = {
    "model": self.lora_model,
    "text_encoder": self.lora_text_encoder  # 如果训练
}
```

**作用:**
1. 统一管理所有需要保存的适配器
2. 保存时遍历mapping，为每个模块添加正确前缀
3. 加载时根据前缀还原到对应模块
4. 便于扩展（如果将来需要添加更多模块）

### 3. Transport的使用

```python
# 三种训练方式都使用相同的transport
trans = create_transport(
    "Linear",              # path type
    "velocity",            # prediction type
    None, None, None,      # 其他参数
    snr_type=self.config.advanced.snr_type,
    do_shift=not self.config.advanced.no_shift,
    seq_len=(1024 // 16) ** 2,
)

# 训练损失计算
loss_dict = trans.training_losses(model, latents, model_kwargs)
```

**关键点:**
- 全量训练: `model = self.model`
- LoRA训练: `model = self.lora_model`
- LyCORIS训练: `model = self.model` (LyCORIS已应用)

---

## 测试建议

### 1. 单卡训练测试
```bash
# LoRA
python trainer.py --config config/train_lumina2_lora.yaml

# LyCORIS
python trainer.py --config config/train_lumina2_lycoris.yaml
```

### 2. 多卡训练测试
```bash
# LoRA
python trainer.py --config config/train_lumina2_lora.yaml lightning.devices=2

# LyCORIS
python trainer.py --config config/train_lumina2_lycoris.yaml lightning.devices=2
```

### 3. 检查点恢复测试
```bash
# 训练几个step后停止，然后恢复
python trainer.py --config config/train_lumina2_lora.yaml trainer.max_steps=100
python trainer.py --config config/train_lumina2_lora.yaml trainer.resume=true
```

### 4. 验证事项
- [ ] 训练loss正常下降
- [ ] 可训练参数数量正确（应该远小于全量训练）
- [ ] 保存的权重文件大小合理（LoRA/LyCORIS应该很小）
- [ ] 检查点能正确加载和恢复训练
- [ ] 样本生成功能正常
- [ ] 多卡训练同步正常

---

## 未来改进方向

### 1. 支持更多LoRA配置
- [ ] rank-stabilized LoRA
- [ ] AdaLoRA
- [ ] DoRA

### 2. 支持更多LyCORIS算法
- [ ] LoHa
- [ ] LoKr
- [ ] DyLoRA

### 3. 混合精度优化
- [ ] 支持更精细的混合精度控制
- [ ] 验证不同精度下的训练稳定性

### 4. 分布式训练优化
- [ ] DeepSpeed Zero优化
- [ ] FSDP优化
- [ ] 梯度检查点优化

---

## 总结

本次重构主要完成：

1. ✅ **代码风格统一**: 参考SDXL的成熟实现
2. ✅ **修复核心问题**: 模型setup和参数管理
3. ✅ **改进可维护性**: 清晰的类命名和注释
4. ✅ **保持训练逻辑**: Lumina2的训练逻辑完全保留
5. ✅ **增强日志输出**: 便于调试和监控

**重要提醒:**
- LoRA和LyCORIS的前向传播方式不同，请勿混淆
- 所有改动都保持了与全量训练相同的loss计算逻辑
- 测试时请验证训练loss和生成质量

---

## 参考文件

- `modules/train_lycoris.py` - SDXL LyCORIS参考实现
- `modules/train_sdxl.py` - SDXL全量训练参考
- `modules/lumina2_model_train.py` - Lumina2全量训练基础类
- `modules/lumina2_model_train_lora.py` - 新的LoRA实现
- `modules/lumina2_model_train_lycoris.py` - 新的LyCORIS实现
