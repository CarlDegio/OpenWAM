#!/usr/bin/env python3
"""Prepare train-only stats and check real samples before launching torchrun.

    python scripts/prepare_piper_joint20.py
    python scripts/prepare_piper_joint20.py dataloader.val_episode_ids='[28,29,30]'

Only project-local preparation artifacts are written; the dataset is read-only.
This does not load the full model or run training.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from openwam.dataloader.piper_joint20 import PiperJoint20Dataset, prepare_normalization_stats
from openwam.dataloader.utils.piper_joint20 import JOINT14_IN_20
from openwam.train.utils.checkpointing import find_latest_weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_foundation(cfg):
    """Read only safetensors metadata to detect incompatible I/O widths early."""
    source = cfg.training.finetune_ckpt_path
    if source and cfg.training.resume_ckpt_path:
        raise ValueError("finetune_ckpt_path and resume_ckpt_path are mutually exclusive")
    if not source:
        if cfg.training.resume_ckpt_path:
            return {"resume": str(cfg.training.resume_ckpt_path), "weight_check": "handled by full-state resume"}
        raise ValueError("This recipe requires training.finetune_ckpt_path or training.resume_ckpt_path")
    source = Path(source)
    checkpoint_dir = source.parent if source.suffix == ".safetensors" else source
    weights = source if source.suffix == ".safetensors" else Path(find_latest_weights(str(source)))
    if not (checkpoint_dir / "config.yaml").is_file() or not (checkpoint_dir / "tokenizer").is_dir():
        raise FileNotFoundError(f"Checkpoint is missing config/tokenizer assets: {checkpoint_dir}")
    arch = cfg.model.architecture
    if arch.action_dim != 80 or arch.state_dim != 80:
        raise ValueError("Piper joint20 must keep model action_dim=80 and state_dim=80")
    with weights.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if size > 100_000_000:
            raise ValueError("Invalid safetensors header length")
        header = json.loads(handle.read(size))
    expected = {
        "action_backbone.action_encoder.weight": [int(cfg.model.action_backbone.dim), 80],
        "action_backbone.action_decoder.weight": [80, int(cfg.model.action_backbone.dim)],
        "action_backbone.action_decoder.bias": [80],
        "proprio_encoder.weight": [4096, 80],
    }
    for name, shape in expected.items():
        if header.get(name, {}).get("shape") != shape:
            raise ValueError(f"Checkpoint tensor {name} does not match shape {shape}")
    return {"weights": str(weights), "io_tensor_shapes": expected, "full_model_loaded": False}


def check_samples(dataset, count, preview_path):
    for index in np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int):
        sample = dataset[int(index)]
        for key, shape, mask_key in (("action", (32, 80), "action_mask"), ("proprio", (1, 80), "proprio_mask")):
            values, mask = sample[key], sample[mask_key]
            if tuple(values.shape) != shape or tuple(mask.shape) != shape:
                raise ValueError(f"{dataset.split} sample {index}: invalid {key}/mask shape")
            if not np.isfinite(values.numpy()).all() or bool((values[~mask] != 0).any()):
                raise ValueError(f"{dataset.split} sample {index}: nonfinite values or nonzero padding")
            if not bool((mask.sum(dim=-1) == 14).all()):
                raise ValueError(f"{dataset.split} sample {index}: expected fourteen physical channels")
        if len(sample["video"]) != 9:
            raise ValueError("Expected nine sampled video frames")
        if index == 0:
            sample["video"][0].save(preview_path)
    return {"episodes": dataset.episode_ids, "windows": len(dataset), "samples_checked": min(count, len(dataset))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="train_piper_joint20")
    parser.add_argument("--force", action="store_true", help="Explicitly regenerate normalization statistics")
    parser.add_argument("--check-samples", type=int, default=2)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides shared with the training command")
    args = parser.parse_args()
    if args.check_samples < 1:
        parser.error("--check-samples must be positive")
    os.chdir(PROJECT_ROOT)
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)
    checkpoint = check_foundation(cfg)
    print("Checkpoint input/output tensor shapes match (full model not loaded).", flush=True)
    stats_path = prepare_normalization_stats(cfg.dataloader, overwrite=args.force)
    print(f"Normalization statistics: {stats_path}", flush=True)
    report = {
        "checkpoint": checkpoint,
        "normalization_stats": str(stats_path),
        "active_joint20_indices": JOINT14_IN_20.tolist(),
    }
    for split in ("train", "val"):
        if split == "val" and not cfg.dataloader.val_episode_ids:
            continue
        dataset = PiperJoint20Dataset(cfg.dataloader, split=split)
        report[split] = check_samples(dataset, args.check_samples, stats_path.parent / f"preview_{split}.jpg")
        print(
            f"{split}: {len(dataset.episode_ids)} episodes, {len(dataset)} windows; sample/mask/JPEG checks passed.",
            flush=True,
        )
    (stats_path.parent / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n")
    OmegaConf.save(cfg, stats_path.parent / "prepared_config.yaml", resolve=True)
    print("Preparation complete. No training was started.", flush=True)


if __name__ == "__main__":
    main()
