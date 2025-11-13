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
    import logging
    logging.getLogger("LyCORIS").addHandler(logger.handlers[0])
    
    from lycoris import create_lycoris, LycorisNetwork
except ImportError as e:
    raise ImportError(
        f"\n\nError import lycoris: {e} \nTry install lycoris using `pip install lycoris_lora toml`"
    )


def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = Lumina2ModelLyCORIS(
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
    
    # 参考SDXL LyCORIS的参数收集方式
    params_to_optim = [{"params": model.lycoris_model.parameters()}]
    
    if config.advanced.get("train_text_encoder"):
        if hasattr(config.optimizer.params, 'lr'):
            lr = config.advanced.get("text_encoder_lr", config.optimizer.params.lr)
            params_to_optim.append({"params": model.lycoris_text_encoder.parameters(), "lr": lr})
        else:
            params_to_optim.append({"params": model.lycoris_text_encoder.parameters()})

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

    # 参考SDXL LyCORIS的setup方式
    model.lycoris_model, optimizer = fabric.setup(model.lycoris_model, optimizer)
    model.lycoris_mapping["model"] = model.lycoris_model
    
    if config.advanced.get("train_text_encoder"):
        model.lycoris_text_encoder = fabric.setup(model.lycoris_text_encoder)
        model.lycoris_mapping["text_encoder"] = model.lycoris_text_encoder
    
    model._fabric_wrapped = fabric
    
    dataloader = fabric.setup_dataloaders(dataloader)
    return model, dataset, dataloader, optimizer, scheduler


class Lumina2ModelLyCORIS(SupervisedFineTune):
    """
    Lumina2 LyCORIS训练模型
    参考SDXL LyCORIS的实现风格，确保代码清晰和可维护
    """
    
    def init_model(self):
        """初始化模型并应用LyCORIS"""
        # 先初始化基础模型
        super().init_model()
        # 再应用LyCORIS
        self.init_lycoris()
    
    def init_lycoris(self):
        """
        初始化LyCORIS适配器
        参考SDXL的init_lycoris方法，保持相同的代码风格
        """
        cfg = self.config
        
        # 获取LyCORIS配置
        default_cfg = cfg.get("lycoris", {})
        
        # 设置默认参数
        if not default_cfg:
            default_cfg = {
                "linear_dim": 16,
                "linear_alpha": 8,
                "algo": "locon",
                "factor": 4,
                "multiplier": 1.0,
            }
            logger.info(f"Using default LyCORIS config: {default_cfg}")
        
        # 存储LyCORIS映射，便于后续保存 - 与SDXL保持一致
        self.lycoris_mapping = lycoris_mapping = {}
        
        # 冻结原始模型参数
        self.model.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # 应用preset以选择所有层 - 与SDXL保持一致
        LycorisNetwork.apply_preset({"target_name": ".*"})

        # 初始化主模型的LyCORIS - 与SDXL日志风格一致
        logger.info("")
        logger.info(f"Initializing model.lycoris_model with {default_cfg}")
        self.lycoris_model = create_lycoris(self.model, **cfg.get("lycoris_model", default_cfg))
        lycoris_mapping["model"] = self.lycoris_model
        
        # 应用到设备并启用训练 - 与SDXL保持一致
        self.lycoris_model.to(self.target_device).apply_to()
        self.lycoris_model.requires_grad_(True)
        
        # 如果需要训练text encoder
        if cfg.advanced.get("train_text_encoder"):
            logger.info("")
            logger.info(f"Initializing model.lycoris_text_encoder with {default_cfg}")
            self.lycoris_text_encoder = create_lycoris(self.text_encoder, **cfg.get("lycoris_text_encoder", default_cfg))
            lycoris_mapping["text_encoder"] = self.lycoris_text_encoder
            
            self.lycoris_text_encoder.to(self.target_device).apply_to()
            self.lycoris_text_encoder.requires_grad_(True)
        
        # 打印可训练参数信息 - 与SDXL类似的统计方式
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.lycoris_model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable LyCORIS parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    def get_module(self):
        """返回LyCORIS模型用于训练"""
        return self.lycoris_model
    
    def load_checkpoint(self, state_dict):
        """
        加载检查点，支持LyCORIS权重
        参考SDXL LyCORIS的load_checkpoint方法
        """
        sd = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        
        # 分离不同模块的权重 - 与SDXL保持一致的命名约定
        model_lycoris_sd = {}
        text_encoder_lycoris_sd = {}
        
        for key in list(sd.keys()):
            if key.startswith("lora_model_"):
                model_lycoris_sd[key.replace("lora_model_", "lycoris_")] = sd.pop(key)
            elif key.startswith("lora_text_encoder_"):
                text_encoder_lycoris_sd[key.replace("lora_text_encoder_", "lycoris_")] = sd.pop(key)
        
        # 加载权重
        if model_lycoris_sd:
            logger.info(f"Loading model LyCORIS weights: {len(model_lycoris_sd)} parameters")
            self.lycoris_model.load_state_dict(model_lycoris_sd, strict=False)
        
        if text_encoder_lycoris_sd and self.config.advanced.get("train_text_encoder"):
            logger.info(f"Loading text encoder LyCORIS weights: {len(text_encoder_lycoris_sd)} parameters")
            self.lycoris_text_encoder.load_state_dict(text_encoder_lycoris_sd, strict=False)
    
    def forward(self, batch):
        """
        前向传播函数 - LyCORIS已经apply_to原始模型
        保持与全量训练相同的逻辑
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
        
        # LyCORIS已经通过apply_to()应用到原始模型，直接使用self.model
        # 这是与LoRA的关键区别 - LyCORIS修改了原始模型的权重
        loss_dict = trans.training_losses(self.model, latents, model_kwargs)

        loss_1024 = loss_dict["loss"].sum() / self.batch_size
        loss = loss_1024 

        # 记录训练损失
        self.log("train_loss", loss, prog_bar=True)
        self.log("loss_1024", loss_1024, prog_bar=True)
        
        # 添加梯度裁剪
        if hasattr(self.config.trainer, 'grad_clip') and self.config.trainer.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.lycoris_model.parameters(), 
                max_norm=self.config.trainer.grad_clip
            )
            self.log("grad_norm", grad_norm, prog_bar=True)
            
        return loss
    
    @rank_zero_only
    def save_checkpoint(self, model_path, metadata):
        """
        保存LyCORIS权重
        完全参考SDXL LyCORIS的保存逻辑
        """
        cfg = self.config.trainer
        state_dict = {}

        # 构建LyCORIS state_dict - 与SDXL完全一致的方式
        for key, module in self.lycoris_mapping.items():
            module_state_dict = module.state_dict()
            new_state_dict = {}
            for k, v in module_state_dict.items():
                # 移除module.前缀（如果有）
                k = k.replace("module.", "")
                # 移除lycoris_前缀
                k = k.replace("lycoris_", "")
                # 添加lora_{key}_前缀以保持兼容性
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
            logger.info(f"Saved LyCORIS weights to {model_path}")
        else:
            model_path += ".ckpt"
            torch.save({"state_dict": state_dict, **metadata}, model_path)
            logger.info(f"Saved LyCORIS weights to {model_path}")
    
    def generate_samples(self, logger, current_epoch, global_step):
        """
        生成样本图像
        LyCORIS已经应用到模型上，可以直接使用
        """
        if hasattr(self, "_fabric_wrapped"):
            if self._fabric_wrapped.world_size > 1:
                return self.generate_samples_dist(logger, current_epoch, global_step)
        
        return self.generate_samples_seq(logger, current_epoch, global_step)
