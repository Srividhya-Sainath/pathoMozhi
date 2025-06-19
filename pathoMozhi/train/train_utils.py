import time
import h5py
from contextlib import suppress
import torch
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.api import FullOptimStateDictConfig
import os
import wandb

def create_feature_loader(base_path_template: str, epoch: int = 0):
    """
    Loads .pt features from the specified base path template.
    """
    def feature_loader(file_path: str):
        filename = file_path.replace(".h5", ".pt")

        pt_file = os.path.join(base_path_template.format(epoch=epoch), filename)

        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"File not found: {filename}")

        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=True)
        except (EOFError, RuntimeError) as e:
            print(f"[ERROR: torch.load failed] Epoch: {epoch}, File: {pt_file}, Error: {e}")
            raise e

        if isinstance(data, dict) and "features" in data:
            feats = data["features"]
        elif isinstance(data, torch.Tensor):
            feats = data
        else:
            raise ValueError(f"Invalid .pt structure for file: {pt_file}")

        if feats.ndim != 2 or feats.shape[1] != 768:
            raise ValueError(f"Expected shape [N, 768] in file {pt_file}, but got {feats.shape}")

        feats = feats.to(dtype=torch.float32)
        return {
            "file_path": file_path,
            "feature": feats
        }

    return feature_loader

def mask_tokens(input_ids, tokenizer, mlm_probability=0.15, protected_mask=None):
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    if protected_mask is not None:
        masked_indices &= ~protected_mask
    labels[~masked_indices] = -100
    input_ids[masked_indices] = tokenizer.mask_token_id
    return input_ids, labels

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_mp_policy_dtype(precision: str):
    if "bfloat16" in precision or "bf16" in precision:
        return torch.bfloat16
    elif precision == "fp16":
        return torch.float16
    else:
        return torch.float32

def get_autocast(precision, cache_enabled=True):
    if precision == "amp":
        return torch.cuda.amp.autocast(cache_enabled=cache_enabled)
    elif precision in ["amp_bfloat16", "amp_bf16"]:
        return lambda: torch.cuda.amp.autocast(
            dtype=torch.bfloat16, cache_enabled=cache_enabled
        )
    else:
        return suppress

def train_one_epoch(
    args,
    model,
    epoch,
    train_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    # setup loader
    num_batches_per_epoch = train_loader.num_batches
    print("Number of batches in training dataset: ", num_batches_per_epoch)
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(
        args.precision, cache_enabled=(not args.fsdp)
    )  # if fsdp, disable cache to save memory
    cast_dtype = get_cast_dtype(args.precision)

    # setup model
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]

    
    model.train()

    # setup logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # loop through dataloader
    for num_steps, batch in tqdm(
        enumerate(train_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        global_step = num_steps + epoch * num_batches_per_epoch

        images = batch["images"].to(device_id, dtype=cast_dtype, non_blocking=True)
        input_ids = batch["input_ids"].to(device_id, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device_id, non_blocking=True)
        labels = batch["labels"].to(device_id, non_blocking=True)

        mlm_input_ids, mlm_labels = mask_tokens(
            input_ids.clone(),
            tokenizer,
            mlm_probability=args.mlm_probability,
            protected_mask=batch["image_token_mask"].to(device_id)
        )

        with autocast():
            # AR loss
            ar_loss = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]

            # MLM loss
            mlm_loss = model(
                vision_x=images,
                lang_x=mlm_input_ids,
                attention_mask=attention_mask,
                labels=mlm_labels,
            )[0]

            loss = ar_loss + args.mlm_loss_weight * mlm_loss

            # if loss is nan, skip this batch
            if torch.isnan(loss):
                print("loss is nan, skipping this batch")
                print("input_ids: ", tokenizer.batch_decode(input_ids))
                print("labels: ", labels)
                print("images: ", images)
                optimizer.zero_grad(set_to_none=True)
                continue

        divided_loss = loss / args.gradient_accumulation_steps
        (divided_loss * args.loss_multiplier).backward()

        if (not args.freeze_lm_embeddings) and (
            not args.fsdp or args.fsdp_use_orig_params
        ):
            # Mask gradients for input embeddings s.t. we only update the added tokens <image> , <|endofchunk|> and <|endofquestion|>
            if args.fsdp:
                embed_grad = model.lang_encoder.get_input_embeddings().weight.grad
            else:
                embed_grad = (
                    model.module.lang_encoder.get_input_embeddings().weight.grad
                )
            if embed_grad is not None: # Added an if condition to check if embed_grad is None
                zero_mask = torch.zeros_like(embed_grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(zero_mask[endofchunk_token_id])
                if args.fsdp:
                    model.lang_encoder.get_input_embeddings().weight.grad = (
                        embed_grad * zero_mask
                    )
                else:
                    model.module.lang_encoder.get_input_embeddings().weight.grad = (
                        embed_grad * zero_mask
                    )

        # clip gradient norm
        if args.fsdp:
            """
            The way we clip gradients with FSDP is different than the non-FSDP case,
            because during FSDP, gradient norms are computed over certain submodules,
            rather than the entire model.
            At least for OPT-125M, this didn't seem to make a difference in performance.
            """
            model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            # rank 0 logging
            if args.rank == 0 and args.report_to_wandb:
                samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    * args.world_size
                    / step_time_m.val
                )
                samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    / step_time_m.val
                )
                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "epoch": epoch,
                        "samples_per_second": samples_per_second,
                        "samples_per_second_per_gpu": samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                        "ar_loss": ar_loss.item(),
                        "mlm_loss": mlm_loss.item(),
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss": loss.item(),
                        "global_step": global_step,
                    },
                    commit=True,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. AR Loss: {ar_loss.item():.3f}, MLM Loss: {mlm_loss.item():.3f}"
            )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def filter_state_dict_to_trainable(model, state_dict):
    """
    Remove non-trainable parameters from model state dict.
    Exception: Embeddings will not be removed, even if frozen.
    This is because we need the new <image> <|endofchunk|> tokens to
    be consistent across initializations.
    """
    for (
        name,
        p,
    ) in model.named_parameters():  # won't work for fsdp + use_orig_params=False
        if "fsdp" in name:
            continue
        if "embed" in name or isinstance(p, torch.nn.Embedding):
            continue
        if not p.requires_grad:
            name = name.replace("._checkpoint_wrapped_module", "")
            if name in state_dict:
                del state_dict[name]
            else:
                print(f"WARNING: filtering but {name} not in state_dict")

    # also remove the keys in state_dict generated from
    # lang_encoder.old_decoder_blocks and lang_encoder.gated_cross_attn_layers
    # because these are already saved in lang_encoder.model...
    to_delete = [
        n
        for n in state_dict.keys()
        if ("lang_encoder.old_decoder_blocks" in n)
        or ("lang_encoder.gated_cross_attn_layers" in n)
        or ("vision_encoder" in n)
    ]
    for name in to_delete:
        del state_dict[name]
    return state_dict


def save_checkpoint(model, optimizer, lr_scheduler, epoch, args):
    """
    Save training checkpoint with model, optimizer, and lr_scheduler state.
    """
    if args.fsdp:
        FSDP.set_state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True),
        )
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer, group=args.my_group)

    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if args.rank == 0:
        if not (args.fsdp and not args.fsdp_use_orig_params):
            model_state = filter_state_dict_to_trainable(model, model_state)

        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)

        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        }

        print(f"Saving checkpoint to {args.run_name}/checkpoint_{epoch}.pt")
        torch.save(checkpoint_dict, f"{args.run_name}/checkpoint_{epoch}.pt")
        if args.report_to_wandb and args.save_checkpoints_to_wandb:
            wandb.save(f"{args.run_name}/checkpoint_{epoch}.pt")

        if args.delete_previous_checkpoint:
            if epoch > 0:
                os.remove(f"{args.run_name}/checkpoint_{epoch-1}.pt")