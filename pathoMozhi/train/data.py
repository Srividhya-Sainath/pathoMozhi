import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathoMozhi.train.data_utils import *
from pathoMozhi.train.train_utils import get_cast_dtype

class PathDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, feature_loader, epoch=0, max_tokens=256):
        self.tokenizer = tokenizer
        self.feature_loader = feature_loader
        self.epoch = epoch
        self.max_tokens = max_tokens
        self.entries = self._load_entries(jsonl_file)
        self.instructions = [
            "Provide a report.",
            "Describe key findings.",
            "Summarize abnormalities if any precisely.",
            "Generate a detailed pathology summary.",
            "State the diagnostic observations clearly.",
            "State relevant abnormalities if observed.",
            "Summarize the microscopic findings.",
            "Report any unusual tissue patterns, if present."
            "Identify key pathological features.",
            "Provide concise diagnostic impressions.",
        ]

    def _load_entries(self, jsonl_file):
        with open(jsonl_file, "r") as f:
            return [json.loads(line) for line in f]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        report_text = entry.get("result", "")
        file_path = entry["file_path"]
        instruction = np.random.choice(self.instructions)
        processed_text = f"<image> {instruction} {report_text} <|endofchunk|>"
        num_image_tokens = processed_text.count("<image>")
        text_encoding = self.tokenizer(
            processed_text,
            max_length=self.max_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            )
        input_ids = text_encoding["input_ids"].squeeze(0)
        attention_mask = text_encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_prefix = f"<image> {instruction}"
        prefix_ids = self.tokenizer(
                prompt_prefix,
                max_length=self.max_tokens,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )["input_ids"].squeeze(0)
        prefix_len = (prefix_ids != self.tokenizer.pad_token_id).sum().item()

        labels[:prefix_len] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        labels[input_ids == self.tokenizer.convert_tokens_to_ids("<image>")] = -100

        image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        image_token_mask = (input_ids == image_token_id)

        assert self.feature_loader is not None, f"Feature loader is None for file {file_path}"
        feature_dict = self.feature_loader(file_path)
        features = feature_dict["feature"]
        if num_image_tokens > 1:
            print(f"[MULTI-IMAGE WARNING] {file_path} has {num_image_tokens} <image> tokens")
        if features.ndim == 2:
            num_repeats = num_image_tokens if num_image_tokens > 0 else 1
            features = features.unsqueeze(0).repeat(num_repeats, 1, 1)
        return {
            "file_path": file_path,
            "raw_text": processed_text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "features": features,
            "image_token_mask": image_token_mask,
        }

def collate_fn(batch, cast_dtype=None):
    images = torch.stack([sample["features"] for sample in batch])
    if cast_dtype is not None:
        images = images.to(dtype=cast_dtype)
    return {
        "file_path": [sample["file_path"] for sample in batch],
        "raw_text": [sample["raw_text"] for sample in batch],
        "input_ids": torch.stack([sample["input_ids"] for sample in batch]),
        "attention_mask": torch.stack([sample["attention_mask"] for sample in batch]),
        "labels": torch.stack([sample["labels"] for sample in batch]),
        "images": images,
        "image_token_mask": torch.stack([sample["image_token_mask"] for sample in batch]),
    }

def build_dataset(args, tokenizer, feature_loader, epoch=0, floor=False):
    shared_epoch = SharedEpoch(epoch=epoch)

    jsonl_file = args.jsonl_file
    dataset = PathDataset(
        jsonl_file=jsonl_file,
        tokenizer=tokenizer,
        feature_loader=feature_loader,
        max_tokens=args.max_tokens,
    )

    sampler = None
    if args.world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
    global_batch_size = args.batch_size * args.world_size
    num_samples = args.train_num_samples
    round_fn = math.floor if floor else math.ceil
    num_batches = round_fn(num_samples / global_batch_size)
    num_samples = num_batches * global_batch_size

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.workers,
        drop_last=True,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda batch: collate_fn(batch, cast_dtype=get_cast_dtype(args.precision)),
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, sampler=sampler, shared_epoch=shared_epoch)

def get_data(args, feature_loader, tokenizer, epoch=0):
    return build_dataset(args, tokenizer, feature_loader, epoch=epoch)
