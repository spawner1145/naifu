#!/usr/bin/env python
"""
Lumina2 LoRA模块测试脚本

用于验证LoRA训练模块是否能正常导入和初始化
"""

import sys
import os

def test_imports():
    """测试必要的库是否已安装"""
    print("=" * 80)
    print("测试必要的库导入")
    print("=" * 80)
    
    # 测试基础库
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import lightning as pl
        print(f"✓ Lightning version: {pl.__version__}")
    except ImportError as e:
        print(f"✗ Lightning import failed: {e}")
        return False
    
    # 测试PEFT库
    try:
        import peft
        print(f"✓ PEFT version: {peft.__version__}")
    except ImportError as e:
        print(f"✗ PEFT not installed: {e}")
        print("  请运行: pip install peft")
        return False
    
    # 测试transformers
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    # 测试safetensors
    try:
        import safetensors
        print(f"✓ Safetensors version: {safetensors.__version__}")
    except ImportError as e:
        print(f"✗ Safetensors import failed: {e}")
        return False
    
    print()
    return True

def test_module_import():
    """测试LoRA训练模块是否能正常导入"""
    print("=" * 80)
    print("测试Lumina2 LoRA训练模块导入")
    print("=" * 80)
    
    try:
        from modules.lumina2_model_train_lora import SupervisedFineTuneLoRA, setup
        print("✓ 成功导入 SupervisedFineTuneLoRA")
        print("✓ 成功导入 setup 函数")
        
        # 检查类的关键方法
        methods = ['init_lora', 'forward', 'save_checkpoint', 'load_checkpoint', 'generate_samples']
        for method in methods:
            if hasattr(SupervisedFineTuneLoRA, method):
                print(f"✓ 方法 {method} 存在")
            else:
                print(f"✗ 方法 {method} 不存在")
                return False
        
        print()
        return True
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_files():
    """测试配置文件是否存在"""
    print("=" * 80)
    print("测试配置文件")
    print("=" * 80)
    
    config_files = [
        "config/train_lumina2_lora.yaml",
        "config/train_lumina2_lora_full.yaml"
    ]
    
    all_exist = True
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ 配置文件存在: {config_file}")
        else:
            print(f"✗ 配置文件不存在: {config_file}")
            all_exist = False
    
    print()
    return all_exist

def test_peft_lora():
    """测试PEFT LoRA功能"""
    print("=" * 80)
    print("测试PEFT LoRA基础功能")
    print("=" * 80)
    
    try:
        import torch
        import torch.nn as nn
        from peft import LoraConfig, get_peft_model
        
        # 创建一个简单的模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        # 创建模型
        model = SimpleModel()
        original_params = sum(p.numel() for p in model.parameters())
        print(f"✓ 创建测试模型，参数量: {original_params}")
        
        # 应用LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=["linear"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        lora_model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        print(f"✓ 成功应用LoRA")
        print(f"✓ 可训练参数量: {trainable_params} ({trainable_params/original_params*100:.2f}%)")
        
        # 测试前向传播
        x = torch.randn(2, 10)
        output = lora_model(x)
        print(f"✓ 前向传播成功，输出形状: {output.shape}")
        
        print()
        return True
    except Exception as e:
        print(f"✗ PEFT LoRA测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n")
    print("*" * 80)
    print("Lumina2 LoRA训练模块测试")
    print("*" * 80)
    print()
    
    results = []
    
    # 运行测试
    results.append(("库导入测试", test_imports()))
    results.append(("PEFT LoRA功能测试", test_peft_lora()))
    results.append(("模块导入测试", test_module_import()))
    results.append(("配置文件测试", test_config_files()))
    
    # 打印总结
    print("=" * 80)
    print("测试总结")
    print("=" * 80)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ 所有测试通过！可以开始使用Lumina2 LoRA训练。")
        print()
        print("使用方法:")
        print("  python trainer.py --config config/train_lumina2_lora.yaml")
    else:
        print("✗ 部分测试失败，请检查上述错误信息。")
        print()
        print("常见问题:")
        print("  1. 如果PEFT未安装: pip install peft")
        print("  2. 如果PyTorch版本过旧: pip install --upgrade torch")
        print("  3. 如果配置文件不存在: 检查是否在正确的目录下运行")
    
    print()
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
