#!/usr/bin/env python
"""
Lumina2 LoRA训练启动脚本示例

使用方法：
    python train_lumina2_lora_example.py

或者使用标准训练脚本：
    python trainer.py --config config/train_lumina2_lora.yaml
"""

import os
import sys

def main():
    # 配置文件路径
    config_path = "config/train_lumina2_lora.yaml"
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"错误: 配置文件 {config_path} 不存在")
        print("请先创建配置文件，可以参考 config/train_lumina2_lora.yaml")
        sys.exit(1)
    
    print("=" * 80)
    print("Lumina2 LoRA训练")
    print("=" * 80)
    print(f"配置文件: {config_path}")
    print()
    
    # 导入trainer
    try:
        from trainer import main as trainer_main
    except ImportError as e:
        print(f"错误: 无法导入trainer模块: {e}")
        sys.exit(1)
    
    # 设置命令行参数
    sys.argv = ["trainer.py", "--config", config_path]
    
    # 启动训练
    trainer_main()

if __name__ == "__main__":
    main()
