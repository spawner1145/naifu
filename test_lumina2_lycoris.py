#!/usr/bin/env python
"""
Lumina2 LyCORIS模块测试脚本

用于验证LyCORIS训练模块是否能正常导入和初始化
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
    
    # 测试LyCORIS库
    try:
        from lycoris import create_lycoris, LycorisNetwork
        print(f"✓ LyCORIS库导入成功")
    except ImportError as e:
        print(f"✗ LyCORIS not installed: {e}")
        print("  请运行: pip install lycoris_lora toml")
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
    """测试LyCORIS训练模块是否能正常导入"""
    print("=" * 80)
    print("测试Lumina2 LyCORIS训练模块导入")
    print("=" * 80)
    
    try:
        from modules.lumina2_model_train_lycoris import SupervisedFineTuneLyCORIS, setup
        print("✓ 成功导入 SupervisedFineTuneLyCORIS")
        print("✓ 成功导入 setup 函数")
        
        # 检查类的关键方法
        methods = ['init_lycoris', 'forward', 'save_checkpoint', 'load_checkpoint', 'generate_samples']
        for method in methods:
            if hasattr(SupervisedFineTuneLyCORIS, method):
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
        "config/train_lumina2_lycoris.yaml",
        "config/train_lumina2_lycoris_full.yaml"
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

def test_lycoris_functionality():
    """测试LyCORIS基础功能"""
    print("=" * 80)
    print("测试LyCORIS基础功能")
    print("=" * 80)
    
    try:
        import torch
        import torch.nn as nn
        from lycoris import create_lycoris, LycorisNetwork
        
        # 创建一个简单的模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.linear = nn.Linear(64, 10)
            
            def forward(self, x):
                x = self.conv(x)
                x = x.mean([2, 3])
                x = self.linear(x)
                return x
        
        # 创建模型
        model = SimpleModel()
        original_params = sum(p.numel() for p in model.parameters())
        print(f"✓ 创建测试模型，参数量: {original_params:,}")
        
        # 测试不同的LyCORIS算法
        algorithms = ["locon", "loha", "lokr"]
        
        for algo in algorithms:
            try:
                # 应用LyCORIS
                LycorisNetwork.apply_preset({"target_name": ".*"})
                lycoris_model = create_lycoris(
                    model,
                    algo=algo,
                    linear_dim=8,
                    linear_alpha=4,
                    factor=2,
                    multiplier=1.0
                )
                
                trainable_params = sum(p.numel() for p in lycoris_model.parameters() if p.requires_grad)
                print(f"✓ 成功应用 {algo.upper()}")
                print(f"  可训练参数: {trainable_params:,} ({trainable_params/original_params*100:.2f}%)")
                
                # 测试前向传播
                lycoris_model.apply_to()
                x = torch.randn(2, 3, 32, 32)
                output = model(x)
                print(f"  前向传播成功，输出形状: {output.shape}")
                
            except Exception as e:
                print(f"✗ {algo.upper()} 算法测试失败: {e}")
                return False
        
        print()
        return True
    except Exception as e:
        print(f"✗ LyCORIS功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_algorithm_comparison():
    """比较不同算法的参数量"""
    print("=" * 80)
    print("LyCORIS算法参数量对比")
    print("=" * 80)
    
    try:
        import torch
        import torch.nn as nn
        from lycoris import create_lycoris, LycorisNetwork
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(512, 512)
                self.linear2 = nn.Linear(512, 512)
                self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
            
            def forward(self, x):
                return x
        
        model = TestModel()
        base_params = sum(p.numel() for p in model.parameters())
        
        print(f"基础模型参数量: {base_params:,}")
        print()
        
        configs = [
            {"algo": "locon", "linear_dim": 16, "linear_alpha": 8, "factor": 4},
            {"algo": "loha", "linear_dim": 16, "linear_alpha": 8, "factor": 4},
            {"algo": "lokr", "linear_dim": 16, "linear_alpha": 8, "factor": 2},
        ]
        
        print(f"{'算法':<10} {'参数量':<15} {'占比':<10} {'说明':<30}")
        print("-" * 70)
        
        for config in configs:
            LycorisNetwork.apply_preset({"target_name": ".*"})
            lycoris_model = create_lycoris(TestModel(), **config)
            lycoris_params = sum(p.numel() for p in lycoris_model.parameters())
            ratio = lycoris_params / base_params * 100
            
            descriptions = {
                "locon": "标准配置，适合视觉模型",
                "loha": "参数较少，训练快速",
                "lokr": "参数最少，最高效"
            }
            
            print(f"{config['algo']:<10} {lycoris_params:>12,}   {ratio:>6.2f}%   {descriptions[config['algo']]}")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ 算法对比测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("\n")
    print("*" * 80)
    print("Lumina2 LyCORIS训练模块测试")
    print("*" * 80)
    print()
    
    results = []
    
    # 运行测试
    results.append(("库导入测试", test_imports()))
    results.append(("LyCORIS功能测试", test_lycoris_functionality()))
    results.append(("算法对比测试", test_algorithm_comparison()))
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
        print("✓ 所有测试通过！可以开始使用Lumina2 LyCORIS训练。")
        print()
        print("使用方法:")
        print("  python trainer.py --config config/train_lumina2_lycoris.yaml")
        print()
        print("快速入门:")
        print("  查看 LUMINA2_LYCORIS_QUICKSTART.md")
        print()
        print("完整文档:")
        print("  查看 LUMINA2_LYCORIS_README.md")
    else:
        print("✗ 部分测试失败，请检查上述错误信息。")
        print()
        print("常见问题:")
        print("  1. 如果LyCORIS未安装: pip install lycoris_lora toml")
        print("  2. 如果PyTorch版本过旧: pip install --upgrade torch")
        print("  3. 如果配置文件不存在: 检查是否在正确的目录下运行")
    
    print()
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
