import time
from contextlib import suppress
import torch
from tqdm import tqdm
import os
import wandb

def create_feature_loader(base_path_template: str, epoch: int = 0, augment: bool = False):
    """
    Loads .pt features from the specified list of base path templates.
    """
    base_path_template = base_path_template.split(",")
    def feature_loader(file_path: str):
        filename = file_path.replace(".h5", ".pt")
        pt_file = None
        for base_path in base_path_template:
            candidate_path = os.path.join(base_path.format(epoch=epoch), filename)
            if os.path.exists(candidate_path):
                pt_file = candidate_path
                break
        if pt_file is None:
            raise FileNotFoundError(f"File not found in any base path: {filename}")
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
        if augment:
            idx = torch.randperm(feats.size(0))
            feats = feats[idx]
            proj_matrix = torch.randn(feats.size(1), feats.size(1), device=feats.device)
            feats = feats @ proj_matrix
            noise_std = 0.01
            feats = feats + torch.randn_like(feats) * noise_std
        feats = feats.to(dtype=torch.float32)
        return {
            "file_path": file_path,
            "feature": feats
        }

    return feature_loader

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
    cls_loss_fns,
):
    # setup loader
    num_batches_per_epoch = train_loader.num_batches
    print("Number of batches in training dataset: ", num_batches_per_epoch)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    # setup model
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]

    organ_loss_fn, diag_loss_fn = cls_loss_fns

    model.train()

    # setup logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # loop through dataloader
    for local_step, batch in tqdm(
        enumerate(train_loader),
        disable=args.rank != 0,
        total=num_batches_per_epoch,
    ):
        global_step = epoch * num_batches_per_epoch + local_step

        images = batch["images"].to(device_id, dtype=cast_dtype, non_blocking=True)
        input_ids = batch["input_ids"].to(device_id, non_blocking=True)
        if args.rank == 0 and epoch == 0 and local_step == 0:
            print("Sample input_ids (decoded):")
            decoded = tokenizer.batch_decode(input_ids[:8], skip_special_tokens=False)
            for line in decoded:
                print(line)
        attention_mask = batch["attention_mask"].to(device_id, non_blocking=True)
        labels = batch["labels"].to(device_id, non_blocking=True)

        with autocast():
            output = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = output["loss"]

            organ_label = batch["organ_label"].to(device_id)
            diagnosis_label = batch["diagnosis_label"].to(device_id)

            loss_cls1 = torch.tensor(0.0, device=device_id)
            loss_cls2 = torch.tensor(0.0, device=device_id)

            if "cls_logits1" in output:
                cls_logits1 = output["cls_logits1"]
                loss_cls1 = organ_loss_fn(cls_logits1, organ_label)
                loss = loss + loss_cls1

            if "cls_logits2" in output:
                cls_logits2 = output["cls_logits2"]
                loss_cls2 = diag_loss_fn(cls_logits2, diagnosis_label)
                loss = loss + loss_cls2
            elif args.cls == "diagnosisnoclass":
                # Use the last hidden state from the output
                if output.get("hidden_states") is not None:
                    hidden_states = output["hidden_states"][-1]  # shape (B, T, D)
                    cls_logits2 = model.cls_head2(hidden_states[:, -1, :])  # Use the last token representation
                    loss_cls2 = diag_loss_fn(cls_logits2, diagnosis_label)
                    loss = loss + loss_cls2
                    output["cls_logits2"] = cls_logits2  # for logging
                else:
                    print("Warning: output.hidden_states is None, skipping diagnosisnoclass loss.")
            elif args.cls == "diagnosisAttn":
                if output.get("hidden_states") is not None:
                    hidden_states = output["hidden_states"][-1]
                    attn_weights = model.attn_pool(hidden_states).softmax(dim=1)
                    pooled = (hidden_states * attn_weights).sum(dim=1)
                    cls_logits2 = model.cls_head2(pooled)
                    loss_cls2 = diag_loss_fn(cls_logits2, diagnosis_label)
                    loss = loss + loss_cls2
                    output["cls_logits2"] = cls_logits2  # for logging
                else:
                    print("Warning: output.hidden_states is None, skipping diagnosisAttn loss.")
            if args.lambda_gate > 0:
                #print("[DEBUG] Applying gate regularization")
                all_attn_gates = [
                    layer.attn_gate for layer in model.module.lang_encoder.gated_cross_attn_layers
                    if layer is not None
                ]
                gate_reg_loss = -torch.stack([gate.tanh() for gate in all_attn_gates]).mean()
                loss = loss + args.lambda_gate * gate_reg_loss

            # if loss is nan, skip this batch
            if torch.isnan(loss):
                print("loss is nan, skipping this batch")
                print("input_ids: ", tokenizer.batch_decode(input_ids))
                print("labels: ", labels)
                print("images: ", images)
                optimizer.zero_grad(set_to_none=True)
                continue

        divided_loss = loss / args.gradient_accumulation_steps
        (divided_loss).backward()

        if args.rank == 0 and epoch == 0 and local_step == 0:
            print("\nTrainable layers with non-zero gradients:")
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None and param.grad.abs().sum() > 0:
                    print(f"✓ {name}")

        if not args.freeze_lm_embeddings:
            embed_grad = model.module.lang_encoder.get_input_embeddings().weight.grad
            zero_mask = torch.zeros_like(embed_grad)
            zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
            zero_mask[endofchunk_token_id] = torch.ones_like(zero_mask[endofchunk_token_id])
            model.module.lang_encoder.get_input_embeddings().weight.grad = embed_grad * zero_mask

        # clip gradient norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((local_step + 1) % args.gradient_accumulation_steps) == 0) or (
            local_step == num_batches_per_epoch - 1
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
                wandb_log = {
                    "data_time": data_time_m.avg,
                    "step_time": step_time_m.avg,
                    "epoch": epoch,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "lr_decay": optimizer.param_groups[0]["lr"],
                    "lr_no_decay": optimizer.param_groups[1]["lr"],
                    "lr_gate": optimizer.param_groups[2]["lr"],
                    "global_step": global_step,
                    "ar_loss": output["loss"].item(),
                    "total_loss": loss.item(),
                    }

                if "cls_logits1" in output:
                    wandb_log["loss_cls_organ"] = loss_cls1.item()
                if "cls_logits2" in output:
                    wandb_log["loss_cls_diag"] = loss_cls2.item()

                wandb.log(wandb_log, commit=True)
                step_time_m.reset()
                data_time_m.reset()
                if args.rank == 0:
                    print(
                        f"Step {local_step+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: {loss.item():.3f}"
                        )

        # Log loss to console
        if ((local_step + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {local_step+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: {loss.item():.3f}"
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
    model_state = model.state_dict()
    optim_state = optimizer.state_dict()

    if args.rank == 0:
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