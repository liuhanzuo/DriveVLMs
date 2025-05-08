import torch
from typing import Optional, List
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class DriveLMNusPhi4Config:
    # =============== 模型核心配置 ===============
    model_name: str = "microsoft/Phi-4-multimodal-instruct"
    model_preparation: str = "prepare_model_and_processor_phi4"
    collate_fn_train: str = "drivelm_nus_phi4_collate_fn"
    collate_fn_val: str = None
    
    # =============== LoRA 专项配置 ===============
    use_lora: bool = True                       # 是否启用LoRA
    lora_r: int = 32                            # LoRA秩
    lora_alpha: int = 64                        # LoRA alpha值 (默认建议2倍rank) <<<
    lora_dropout: float = 0.1                   # LoRA层dropout率 <<<
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", 
            "v_proj",
            "vision_projection",
            "text_projection"
        ]
    )
    lora_task_type: str = "CAUSAL_LM"           # 任务类型（根据模型调整）<<<
    
    # =============== 训练超参数 ===============
    num_train_epochs: int = 3
    batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 8
    lr: float = 5e-6
    warmup_steps: int = 500
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0
    
    # =============== 系统配置 ===============
    seed: int = 42
    dtype = torch.bfloat16
    quantization: bool = False                   # 是否量化（QLoRA需设为True）<<<
    use_flash_attention: bool = False            # 是否启用Flash Attention
    
    # =============== 日志与保存 ===============
    wandb_project: Optional[str] = "DriveVLMs"          # W&B项目名（None则禁用）<<<
    run_name: str = f"FULL-{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    output_dir: str = "/data2/private-data/zhangn/pretrained/phi4/" + f"{run_name}"
    save_steps: int = 500                        # 保存间隔步数
    log_steps: int = 10                          # 日志记录间隔
    print_steps: int = 10                        # 控制台打印间隔
    
    # =============== 恢复/调试 ===============
    resume_from_checkpoint: bool = False
    save_lora_adapter_when_checkpointing: bool = True  # 是否保存LoRA适配器 <<<
    find_unused_parameters: bool = True          # DDP相关

    # =============== 数据集 ===============
    dataset_name: str = "data/DriveLM_nuScenes/split/train"
    max_seq_length: int = 2048                   # 最大序列长度 <<<
    image_size: int = 224                        # 输入图像尺寸 <<<
    
    # =============== 实验性功能 ===============
    gradient_checkpointing: bool = True          # 梯度检查点（节省显存）<<<
    fsdp_config: Optional[dict] = field(default_factory=lambda: None)           # FSDP配置（如需分布式）<<<

config = DriveLMNusPhi4Config()