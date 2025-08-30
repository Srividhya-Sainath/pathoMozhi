""" Main training script """

import argparse
import os
import random
import wandb

import torch
torch.cuda.empty_cache()

import numpy as np

from data import get_data
from distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP

from train_utils import (
    train_one_epoch,
    save_checkpoint,
    create_feature_loader,
)
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from pathoMozhi import create_model_and_transforms

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main():
    parser = argparse.ArgumentParser()

    # training args
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--delete_previous_checkpoint", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant", choices=["constant", "linear", "cosine"])
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--precision", type=str, default="fp32", choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--lambda_gate", type=float, default=0.0, help="Weight for the gate loss, if applicable.")

    # data
    parser.add_argument("--vision_features", type=str, required=True)
    parser.add_argument("--jsonl_file", type=str, required=True)
    parser.add_argument("--train_num_samples", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=1)

    # model
    parser.add_argument("--lm_path", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--tokenizer_path", type=str, default="facebook/opt-30b")
    parser.add_argument("--cross_attn_every_n_layers", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=312)
    parser.add_argument("--freeze_lm_embeddings", action="store_true")
    parser.add_argument("--gate_learning_rate", type=float, default=None)
    parser.add_argument("--perceiver", type=str, default=None, help="Optional path to a pretrained perceiver checkpoint")

    # distributed
    parser.add_argument("--dist-url", type=str, default="env://")
    parser.add_argument("--dist-backend", type=str, default="nccl")

    # wandb
    parser.add_argument("--report_to_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--save_checkpoints_to_wandb", action="store_true")

    args = parser.parse_args()
    args.cls = args.cls.lower()

    if "{epoch}" not in args.vision_features:
        raise ValueError("The --vision_features argument must include '{epoch}' for epoch-based dynamic path resolution.")

    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    random_seed(args.seed, args.rank)
    model, tokenizer = create_model_and_transforms(
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
    )
    if args.rank == 0:
        print("\n[Model Parameter Names - In Perceiver Resampler]")
        for name, _ in model.named_parameters():
            print(name)

    if args.perceiver is not None:
        def strip_prefix_if_present(state_dict, prefix="perceiver."):
            return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

        ckpt = torch.load(args.perceiver, map_location="cpu")
        excluded_keys = ["classifier", "attn_weights"]
        ckpt = {k: v for k, v in ckpt.items() if not any(ex in k.lower() for ex in excluded_keys)}
        ckpt = strip_prefix_if_present(ckpt)
        msg = model.perceiver.load_state_dict(ckpt, strict=False)
        if args.rank == 0:
            print(f"Loaded Perceiver weights with {len(msg.missing_keys)} missing and {len(msg.unexpected_keys)} unexpected keys.")
            print(f"Missing keys: {msg.missing_keys}")
            print(f"Unexpected keys: {msg.unexpected_keys}")

        # Unfreeze all perceiver layers
        for block in model.perceiver.layers:
            for submodule in block:
                for param in submodule.parameters():
                    param.requires_grad = True

    if args.rank == 0:
        print(f"Start running training on rank {args.rank}.")
        print(f"Initializing distributed training with {args.world_size} GPUs.")

    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        msd = checkpoint["model_state_dict"]
        msd = {k.replace("module.", ""): v for k, v in msd.items()}
        resume_from_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(msd, strict=False)

    model = model.to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    if args.gradient_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing
        )
        import functools

        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            offload_to_cpu=True,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            ddp_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: getattr(m, "_use_gradient_checkpointing", False),
        )

    def get_grouped_params(named_params, base_lr, wd, gate_lr=None, gate_lr_mult=1.0):
        decay, no_decay, gate = [], [], []
        for n, p in named_params:
            if not p.requires_grad:
                continue
            if n.endswith("attn_gate") or n.endswith("ff_gate"):
                gate.append(p)
            elif p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        lr_gate = gate_lr if gate_lr is not None else base_lr * gate_lr_mult
        return [
            {"params": decay, "weight_decay": wd, "lr": base_lr},
            {"params": no_decay, "weight_decay": 0.0, "lr": base_lr},
            {"params": gate, "weight_decay": 0.0, "lr": lr_gate},
        ]

    params_to_optimize = list(
        filter(lambda x: x[1].requires_grad and not getattr(x[1], "exclude_from_optimizer", False),
               ddp_model.named_parameters())
    )

    optimizer = torch.optim.AdamW(
        get_grouped_params(params_to_optimize, args.learning_rate, args.weight_decay, gate_lr=args.gate_learning_rate),
        betas=(0.9, 0.999)
    )

    if args.rank == 0:
        for i, group in enumerate(optimizer.param_groups):
            print(f"\n[Param Group {i}] LR: {group['lr']}, Weight Decay: {group['weight_decay']}")
            matched_names = []
            for param in group['params']:
                for name, p in ddp_model.named_parameters():
                    if p is param:
                        matched_names.append(name)
                        break
            print("Matched Params:")
            for name in matched_names:
                print(f"  - {name}")
            if not matched_names:
                print("  (No matched parameters)")

    if args.resume_from_checkpoint is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError:
            print("WARNING: Could not load optimizer state due to mismatch.")

    import math
    global_batch_size = args.batch_size * args.world_size
    steps_per_epoch = math.ceil(args.train_num_samples / global_batch_size)
    total_training_steps = steps_per_epoch * args.num_epochs

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_training_steps
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_training_steps
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    if args.resume_from_checkpoint is not None and "lr_scheduler_state_dict" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    ddp_model.train()

    for epoch in range(resume_from_epoch, args.num_epochs):
        train_feature_loader = create_feature_loader(args.vision_features, epoch=epoch, augment=False)
        train_dataset = get_data(args, train_feature_loader, tokenizer, epoch=epoch)
        if args.rank == 0:
            print(f"[Epoch {epoch}] Using vision feature path: {args.vision_features.format(epoch=epoch)}")
        train_dataset.set_epoch(epoch)
        train_loader = train_dataset.dataloader

        train_one_epoch(
            args=args,
            model=ddp_model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            device_id=device_id,
            wandb=wandb,
        )
        save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, args)
    save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, args)

if __name__ == "__main__":
    main()
