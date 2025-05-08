import os
import json
import torch
from tqdm import tqdm
from functools import partial
from accelerate.utils import set_seed
from accelerate import Accelerator, DistributedType
from torch.nn.parallel import DistributedDataParallel
import argparse
from peft import LoraConfig, get_peft_model
from drivevlms.build import build_collate_fn, build_preparation
from drivevlms.utils import (prepare_training_dataloader,
                             prepare_optimizer_and_scheduler,
                             save_checkpoint,
                             load_checkpoint,
                             save_lora_adapter,
                             write_log_to_json,
                             load_dataclass_config
                            )
from accelerate.utils import (get_mixed_precision_context_manager,
                              convert_outputs_to_fp32)
from types import MethodType
import wandb

class MyAccelerator(Accelerator):
    def prepare_model(self, model, device_placement=None):
        if device_placement is None:
            device_placement = self.device_placement and self.distributed_type != DistributedType.FSDP

        self._models.append(model)

        if self.native_amp:
            model._original_forward = model.forward
            autocast_context = get_mixed_precision_context_manager(self.native_amp, self.autocast_handler)
            if self.fp8_backend == "MSAMP" or not hasattr(model.forward, "__func__"):
                model_forward_func = model.forward
                model.forward = convert_outputs_to_fp32(autocast_context(model_forward_func))
            else:
                model_forward_func = model.forward.__func__
                new_forward = autocast_context(model_forward_func)
                model.forward = MethodType(new_forward, model)
                model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)

        model = model.to(self.device)
        if self.distributed_type in [DistributedType.MULTI_GPU, DistributedType.MULTI_CPU]:
            model = DistributedDataParallel(
                model,
                device_ids=[self.local_process_index] if torch.cuda.is_available() else None,
                output_device=self.local_process_index if torch.cuda.is_available() else None,
                find_unused_parameters=True
            )
        return model

def prepare_lora_model(config, model):
    """Prepare the model for LoRA fine-tuning."""
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=config.lora_task_type,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def train(args):
    # load config
    config = load_dataclass_config(args.config)
    wandb.init(
        project=config.wandb_project,  # 项目名称
        name=config.run_name,         # 当前运行的名称
        config=vars(config),          # 配置信息
    )
    set_seed(config.seed)

    # set accelerate for ddp training
    mixed_precision = "bf16" if config.dtype == torch.bfloat16 else "no"
    if config.find_unused_parameters:
        accelerator = MyAccelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=["tensorboard", "wandb"],
            project_dir=config.output_dir,
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=["tensorboard", "wandb"],
            project_dir=config.output_dir,
        )

    # prepare model and processor
    prepare_model_and_processor = build_preparation(config.model_preparation)
    model, processor = prepare_model_and_processor(config)
    
    # Convert model to LoRA
    model = prepare_lora_model(config, model)

    # Build the data collation function and dataloader
    collate_fn = build_collate_fn(config.collate_fn_train)
    train_collate_fn = partial(collate_fn, processor=processor, dtype=config.dtype)
    dataloader = prepare_training_dataloader(config, train_collate_fn)

    # Prepare optimizer and scheduler
    num_training_steps = (
        len(dataloader) * config.num_train_epochs // config.gradient_accumulation_steps
    )
    optimizer, lr_scheduler = prepare_optimizer_and_scheduler(
        config, model, num_training_steps
    )

    num_training_steps = num_training_steps // accelerator.num_processes

    dataloader, model, optimizer, scheduler = accelerator.prepare(
        dataloader, model, optimizer, lr_scheduler
    )

    progress_bar = tqdm(
        total=num_training_steps, disable=not accelerator.is_local_main_process
    )
    starting_epoch = 1
    global_step = 0
    resumed = False

    skipped_dataloader = dataloader
    if config.resume_from_checkpoint and os.path.exists(
        f"{config.output_dir}/training_info.json"
    ):
        with open(f"{config.output_dir}/training_info.json", "r") as f:
            training_info = json.load(f)

        starting_epoch = training_info["epoch"]
        global_step = training_info["step"]
        load_checkpoint(accelerator, training_info["latest_checkpoint"])

        progress_bar.update(global_step)
        accelerator.print(
            f"Resumed from checkpoint: {training_info['latest_checkpoint']}"
        )

        resumed = True
        skip_batch_count = (
            global_step * config.gradient_accumulation_steps % len(dataloader)
        )
        skipped_dataloader = accelerator.skip_first_batches(
            dataloader, num_batches=skip_batch_count
        )

    accelerator.print(f"Starting epoch: {starting_epoch}, Global step: {global_step}")

    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_loss_count = 0
    dataloader_step = 0
    grad_norm = None

    accelerator.init_trackers(
        project_name=config.wandb_project,
        init_kwargs={
            "wandb": {"name": config.run_name}
        },
        config=vars(config),
    )

    for epoch in range(1, config.num_train_epochs + 1):
        if epoch < starting_epoch:
            continue
        train_dataloader = skipped_dataloader if epoch == starting_epoch else dataloader
        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                output = model(**batch)
                loss = output.loss
                total_loss += loss.detach().cpu()
                total_loss_count += 1

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                dataloader_step += 1
                if dataloader_step % config.gradient_accumulation_steps == 0:
                    global_step += 1
                    progress_bar.update(1)

                    if global_step % config.print_steps == 0:
                        accelerator.print(
                            f"Epoch {epoch}, Step {global_step}, Loss {loss.item()}"
                        )

                    if resumed or global_step % config.log_steps == 0:
                        log_data = {
                            "train/loss": accelerator.gather(total_loss)
                            .detach()
                            .sum()
                            .item()
                            / accelerator.num_processes
                            / total_loss_count,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/global_step": global_step,
                            "train/epoch": global_step
                            / num_training_steps
                            * config.num_train_epochs,
                            "train/grad_norm": (
                                grad_norm.detach().item()
                                if isinstance(grad_norm, torch.Tensor)
                                else grad_norm
                            ),
                        }
                        accelerator.log(log_data, step=global_step)
                        wandb.log(log_data, step=global_step)
                        write_log_to_json(log_data, step=global_step, file_path=args.log_path)
                        resumed = False
                        accelerator.wait_for_everyone()

                    if global_step % config.save_steps == 0:
                        save_checkpoint(
                            accelerator, model, epoch, global_step, config, loss.item()
                        )
                        # Save LoRA adapter specifically
                        save_lora_adapter(accelerator, model, f"{config.output_dir}/step-{global_step}")
                    
                    total_loss = torch.tensor(0.0, device=accelerator.device)
                    total_loss_count = 0
                    torch.cuda.empty_cache()

        # Save at end of each epoch
        epoch_path = f"{config.output_dir}/epoch-{epoch}"
        save_checkpoint(
            accelerator, model, epoch, global_step, config, loss.item(), checkpoint_dir=epoch_path
        )
        save_lora_adapter(accelerator, model, epoch_path)

    # Final save
    final_path = f"{config.output_dir}/final_model"
    save_lora_adapter(accelerator, model, final_path)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        final_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    wandb.save(f"{final_path}/pytorch_model.bin")  # 保存模型权重
    wandb.save(args.config)  # 保存配置文件
    accelerator.end_training()
    wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the config file")
    parser.add_argument("--log_path",
                        default='./lora_finetune_log.json',
                        help="Path to the training log")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)
