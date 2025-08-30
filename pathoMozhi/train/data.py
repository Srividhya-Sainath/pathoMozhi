import json
import math
import torch
from torch.utils.data import DataLoader, Dataset
from data_utils import *
from train_utils import get_cast_dtype

class PathDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, feature_loader, epoch=0, max_tokens=312):
        self.tokenizer = tokenizer
        self.feature_loader = feature_loader
        self.epoch = epoch
        self.max_tokens = max_tokens

    def _load_entries(self, jsonl_file):
        with open(jsonl_file, "r") as f:
            return [json.loads(line) for line in f]
    

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        report_text = entry.get("result", "")
        file_path = entry["file_path"]
        prompt = "<image> Final Diagnosis:"
        text = f"{report_text} <|endofchunk|>"

        # Logging the file_path and raw_text
        log_entry = {
            "file_path": file_path,
            "text": text
        }
        with open(f"data_log_epoch_{self.epoch}.jsonl", "a", encoding="utf-8") as log_f:
            log_f.write(json.dumps(log_entry) + "\n")

        text_encoding = self.tokenizer(
            prompt,
            text,
            max_length=self.max_tokens,
            truncation="only_second",
            padding="max_length",
            return_tensors="pt",
            )
        input_ids = text_encoding["input_ids"].squeeze(0)
        attention_mask = text_encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        labels[input_ids == self.tokenizer.convert_tokens_to_ids("<image>")] = -100

        assert self.feature_loader is not None, f"Feature loader is None for file {file_path}"
        feature_dict = self.feature_loader(file_path)
        features = feature_dict["feature"]
        if features.ndim == 2:
            features = features.unsqueeze(0)

        return {
            "file_path": file_path,
            "raw_text": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "features": features,
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
    }

def build_dataset(args, tokenizer, feature_loader, epoch=0, floor=False):
    shared_epoch = SharedEpoch(epoch=epoch)

    jsonl_file = args.jsonl_file
    dataset = PathDataset(
        jsonl_file=jsonl_file,
        tokenizer=tokenizer,
        feature_loader=feature_loader,
        max_tokens=args.max_tokens,
        epoch=epoch,  # Pass epoch to dataset
    )
    dataset.epoch = epoch  # Ensure epoch is set for proper logging

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
