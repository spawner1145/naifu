import safetensors
import torch
import os
import lightning as pl
import torch.nn.functional as F
from omegaconf import OmegaConf
from common.utils import get_class, get_latest_checkpoint, load_torch_file
from common.logging import logger
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import DataLoader
from IndexKits.index_kits.sampler import DistributedSamplerWithStartIndex, BlockDistributedSampler
from data_loader.arrow2_load_stream_ import TextImageArrowStream
from safetensors.torch import save_file

from modules.lumina2_model_train import SupervisedFineTune
from models.lumina.transport import create_transport

import random
import math

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    raise ImportError(
        f"\n\nError importing peft: {e}\nTry install peft using `pip install peft`"
    )


def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = Lumina2ModelLoRA(
        model_path=model_path, 
        config=config, 
        device=fabric.device
    )

    world_size = fabric.world_size
    logger.info(f"loading dataset from {config.dataset.index_file}")
    dataset = TextImageArrowStream(args="args",
                                   resolution=config.trainer.resolution,
                                   random_flip=config.dataset.random_flip,
                                   log_fn=logger.info,
                                   index_file=config.dataset.index_file,
                                   multireso=config.dataset.multireso,
                                   batch_size=config.trainer.batch_size,
                                   world_size=world_size
                                   )

    if config.dataset.multireso:
        sampler = BlockDistributedSampler(dataset, num_replicas=world_size, rank=fabric.global_rank, seed=config.trainer.seed,
                                          shuffle=True, drop_last=True, batch_size=config.trainer.batch_size)
    else:
        sampler = DistributedSamplerWithStartIndex(dataset, num_replicas=world_size, rank=fabric.global_rank, seed=config.trainer.seed,
                                                   shuffle=True, drop_last=True)
        
    dataloader = DataLoader(dataset, batch_size=config.trainer.batch_size, shuffle=False, sampler=sampler,
                        num_workers=config.dataset.num_workers, pin_memory=True, drop_last=True)
    
    # 收集可训练参数 - 参考SDXL LyCORIS风格
    params_to_optim = [{"params": model.lora_model.parameters()}]
    
    if config.advanced.get("train_text_encoder"):
        if hasattr(config.optimizer.params, 'lr'):
            lr = config.advanced.get("text_encoder_lr", config.optimizer.params.lr)
            params_to_optim.append({"params": model.lora_text_encoder.parameters(), "lr": lr})
        else:
            params_to_optim.append({"params": model.lora_text_encoder.parameters()})

    optim_param = config.optimizer.params
    optimizer = get_class(config.optimizer.name)(params_to_optim, **optim_param)
    scheduler = None
    if config.get("scheduler"):
        scheduler = get_class(config.scheduler.name)(
            optimizer, **config.scheduler.params
        )
        
    if config.trainer.get("resume"):
        latest_ckpt = get_latest_checkpoint(config.trainer.checkpoint_dir)
        remainder = {}
        if latest_ckpt:
            logger.info(f"Loading weights from {latest_ckpt}")
            remainder = sd = load_torch_file(ckpt=latest_ckpt, extract=False)
            if latest_ckpt.endswith(".safetensors"):
                remainder = safetensors.safe_open(latest_ckpt, "pt").metadata()
            model.load_checkpoint(sd.get("state_dict", sd))
            config.global_step = remainder.get("global_step", 0)
            config.current_epoch = remainder.get("current_epoch", 0)

    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")

    # 参考SDXL LyCORIS的setup方式 - 直接setup LoRA包装后的模型
    model.lora_model, optimizer = fabric.setup(model.lora_model, optimizer)
    model.lora_mapping["model"] = model.lora_model
    
    if config.advanced.get("train_text_encoder"):
        model.lora_text_encoder = fabric.setup(model.lora_text_encoder)
        model.text_encoder = model.lora_text_encoder
        model.lora_mapping["text_encoder"] = model.lora_text_encoder

    model._fabric_wrapped = fabric

    if hasattr(model, "setup"):
        model.setup(fabric)
    
    dataloader = fabric.setup_dataloaders(dataloader)
    return model, dataset, dataloader, optimizer, scheduler


class Lumina2ModelLoRA(SupervisedFineTune):
    """
    Lumina2 LoRA训练模型
    参考SDXL LyCORIS的实现风格，确保代码清晰和可维护
    """
    
    def init_model(self):
        """初始化模型并应用LoRA"""
        # 先初始化基础模型
        super().init_model()
        # 再应用LoRA
        self.init_lora()
    
    def init_lora(self):
        """
        初始化LoRA适配器
        参考SDXL LyCORIS的init_lycoris方法风格
        """
        cfg = self.config
        use_lora = cfg.get("use_lora", True)
        
        if not use_lora:
            logger.warning("use_lora is False, but using LoRA training module. Enabling LoRA.")
        
        # 获取LoRA配置
        raw_lora_params = cfg.get("lora_params", {})
        if raw_lora_params is None:
            raw_lora_params = {}
        elif not isinstance(raw_lora_params, dict):
            raw_lora_params = OmegaConf.to_container(raw_lora_params, resolve=True)

        default_target_modules = ["qkv", "out"]
        target_modules = raw_lora_params.get("target_modules") or default_target_modules
        if isinstance(target_modules, str):
            target_modules = [target_modules]

        default_lora_config = {
            "r": raw_lora_params.get("r", 16),
            "lora_alpha": raw_lora_params.get("lora_alpha", 16),
            "target_modules": target_modules,
            "lora_dropout": raw_lora_params.get("lora_dropout", 0.0),
            "bias": raw_lora_params.get("bias", "none"),
            "task_type": "CAUSAL_LM",
        }
        
        # 存储LoRA映射，便于后续保存
        self.lora_mapping = lora_mapping = {}
        
        # 冻结原始模型参数
        self.model.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # 应用LoRA到主模型
        logger.info("")
        logger.info(f"Initializing model.lora_model with config: {default_lora_config}")
        lora_config = LoraConfig(**default_lora_config)
        self.lora_model = get_peft_model(self.model, lora_config)
        self.lora_model.print_trainable_parameters()
        lora_mapping["model"] = self.lora_model

        base_model = None
        if hasattr(self.lora_model, "get_base_model"):
            base_model = self.lora_model.get_base_model()
        elif hasattr(self.lora_model, "base_model"):
            base_model = self.lora_model.base_model
        if base_model is not None and hasattr(base_model, "mark_forward_method"):
            base_model.mark_forward_method("forward_with_cfg")
        
        # 如果需要训练text encoder
        if cfg.advanced.get("train_text_encoder"):
            logger.info("")
            logger.info(f"Initializing model.lora_text_encoder with config: {default_lora_config}")
            text_encoder_lora_config = LoraConfig(**default_lora_config)
            self.lora_text_encoder = get_peft_model(self.text_encoder, text_encoder_lora_config)
            self.lora_text_encoder.print_trainable_parameters()
            lora_mapping["text_encoder"] = self.lora_text_encoder
            self.text_encoder = self.lora_text_encoder
        
        # 验证是否有可训练参数
        trainable = sum(p.numel() for p in self.lora_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        if trainable == 0:
            raise RuntimeError(
                "No trainable parameters found after LoRA initialization! "
                f"请检查 target_modules 设置: {target_modules}"
            )
        logger.info(f"Total parameters: {total:,}")
        logger.info(f"Trainable LoRA parameters: {trainable:,} ({trainable/total*100:.2f}%)")
    
    def get_module(self):
        """返回LoRA模型用于训练"""
        return self.lora_model
    
    def load_checkpoint(self, state_dict):
        """
        加载检查点，支持LoRA权重
        参考SDXL LyCORIS的load_checkpoint方法
        """
        sd = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        
        # 分离不同模块的权重
        model_lora_sd = {}
        text_encoder_lora_sd = {}
        
        for key in list(sd.keys()):
            if key.startswith("lora_model_"):
                model_lora_sd[key.replace("lora_model_", "")] = sd.pop(key)
            elif key.startswith("lora_text_encoder_"):
                text_encoder_lora_sd[key.replace("lora_text_encoder_", "")] = sd.pop(key)
        
        # 加载权重
        if model_lora_sd:
            logger.info(f"Loading model LoRA weights: {len(model_lora_sd)} parameters")
            self.lora_model.load_state_dict(model_lora_sd, strict=False)
        
        if text_encoder_lora_sd and self.config.advanced.get("train_text_encoder"):
            logger.info(f"Loading text encoder LoRA weights: {len(text_encoder_lora_sd)} parameters")
            self.lora_text_encoder.load_state_dict(text_encoder_lora_sd, strict=False)
    
    def forward(self, batch):
        """
        前向传播函数 - 使用LoRA模型训练
        保持与全量训练相同的逻辑，只是模型换成lora_model
        """
        images = batch["pixels"].to(self.target_device)       
        prompts = batch["prompts"]

        # 创建transport - 与全量训练完全一致
        trans = create_transport(
            "Linear",
            "velocity",
            None,
            None,
            None,
            snr_type=self.config.advanced.snr_type,
            do_shift=not self.config.advanced.no_shift,
            seq_len=(1024 // 16) ** 2,
        )

        # 编码文本提示
        prompt_embeds, prompt_masks = self.encode_prompt(
            prompts, 
            self.text_encoder,
            self.tokenizer,
            proportion_empty_prompts=0.1
        )

        # 对图像进行VAE编码
        latents = self.encode_images(images)

        model_kwargs = dict(cap_feats=prompt_embeds, cap_mask=prompt_masks)
        
        # 使用LoRA模型进行训练 - 关键：这里使用lora_model替代model
        loss_dict = trans.training_losses(self.lora_model, latents, model_kwargs)

        loss_1024 = loss_dict["loss"].sum() / self.batch_size
        loss = loss_1024 

        # 记录训练损失
        self.log("train_loss", loss, prog_bar=True)
        self.log("loss_1024", loss_1024, prog_bar=True)
        
        # 添加梯度裁剪
        if hasattr(self.config.trainer, 'grad_clip') and self.config.trainer.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.lora_model.parameters(), 
                max_norm=self.config.trainer.grad_clip
            )
            self.log("grad_norm", grad_norm, prog_bar=True)
            
        return loss
    
    @rank_zero_only
    def save_checkpoint(self, model_path, metadata):
        """
        保存LoRA权重
        参考SDXL LyCORIS的保存逻辑
        """
        cfg = self.config.trainer
        state_dict = {}

        # 构建LoRA state_dict - 类似SDXL LyCORIS的方式
        for key, module in self.lora_mapping.items():
            module_state_dict = module.state_dict()
            new_state_dict = {}
            for k, v in module_state_dict.items():
                # 移除module.前缀（如果有）
                k = k.replace("module.", "")
                # 添加lora_{key}_前缀
                k = f"lora_{key}_{k}"
                new_state_dict[k] = v

            state_dict.update(new_state_dict)

        # 添加元数据
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (int, float, str)):
                    metadata[k] = str(v)

        # 保存文件
        if cfg.get("save_format") == "safetensors":
            model_path += ".safetensors"
            save_file(state_dict, model_path, metadata=metadata)
            logger.info(f"Saved LoRA weights to {model_path}")
        else:
            model_path += ".ckpt"
            torch.save({"state_dict": state_dict, **metadata}, model_path)
            logger.info(f"Saved LoRA weights to {model_path}")
    
    def generate_samples(self, logger, current_epoch, global_step):
        """
        生成样本图像
        在推理时使用LoRA模型生成图像
        """
        if hasattr(self, "_fabric_wrapped"):
            if self._fabric_wrapped.world_size > 1:
                return self.generate_samples_dist(logger, current_epoch, global_step)
        
        return self.generate_samples_seq(logger, current_epoch, global_step)
    
    def generate_samples_seq(self, logger, current_epoch, global_step):
        """序列化生成样本（使用LoRA模型）"""
        # 临时替换model为lora_model以便生成
        original_model = self.model
        self.model = self.lora_model
        
        try:
            result = super().generate_samples_seq(logger, current_epoch, global_step)
        finally:
            self.model = original_model
        
        return result
    
    def generate_samples_dist(self, logger, current_epoch, global_step):
        """分布式生成样本（使用LoRA模型）"""
        original_model = self.model
        self.model = self.lora_model
        
        try:
            result = super().generate_samples_dist(logger, current_epoch, global_step)
        finally:
            self.model = original_model
        
        return result
