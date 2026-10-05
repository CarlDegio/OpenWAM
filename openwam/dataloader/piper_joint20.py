"""Read embedded-image LeRobot v2.1 Piper trajectories for Alpha fine-tuning.

Actions are the recorded row-aligned joint targets, not next-frame measured
states. Both physical action14 and state14 become padded joint20, normalize
with separate train-only statistics, then scatter into the original 80-D head.
No FK, rotation conversion, gripper clipping or change of units is performed.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from PIL import Image

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.transforms.multiview import assemble_multiview_layout, format_prompt_for_inference
from openwam.dataloader.transforms.video import VideoColorJitter, color_jitter_enabled
from openwam.dataloader.utils.normalization import STAT_KEYS, apply_normalization
from openwam.dataloader.utils.piper_joint20 import (
    DIM_MASK20,
    FEATURE_NAMES,
    JOINT20_IN_80,
    PADDING20,
    REPRESENTATION,
    pack_joint14,
)
from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify, parse_unify_spec


@dataclass(frozen=True)
class Episode:
    index: int
    path: Path
    length: int
    size: int
    mtime_ns: int


def _config(config) -> dict:
    cfg = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
    required = {
        "action_mode": "joint20",
        "action_representation": REPRESENTATION,
        "unify_action": True,
        "normalize_mode": "min-max",
        "num_frames": 33,
        "video_stride": 4,
    }
    for key, value in required.items():
        if cfg.get(key) != value:
            raise ValueError(f"Piper joint20 requires {key}={value!r}, got {cfg.get(key)!r}")
    for key in ("unify_action_map", "unify_state_map"):
        spec = cfg.get(key) if cfg.get(key) is not None else cfg.get("unify_action_map")
        if spec is None or not np.array_equal(parse_unify_spec(spec, UNIFY_DIM), JOINT20_IN_80):
            raise ValueError(f"Piper joint20 requires {key} to map left/right blocks to 0-9 / 34-43")
    for key in ("action_dim_mask", "proprio_dim_mask"):
        mask = np.asarray(cfg.get(key))
        if mask.dtype != np.bool_ or not np.array_equal(mask, DIM_MASK20):
            raise ValueError(f"{key} must enable only the 14 physical channels of padded joint20")
    if int(cfg.get("window_stride", 1)) < 1:
        raise ValueError("window_stride must be positive")
    if not cfg.get("normalization_stats_path"):
        raise ValueError("normalization_stats_path is required; run scripts/prepare_piper_joint20.py first")
    cameras = cfg.get("camera_layout")
    if not cameras or len(cameras) != 3 or len(set(cameras)) != 3:
        raise ValueError("camera_layout must specify three distinct camera names")
    if cfg.get("target_camera") not in cameras:
        raise ValueError("target_camera must be in camera_layout")
    if int(cfg.get("height", 384)) % 32 or int(cfg.get("width", 320)) % 32:
        raise ValueError("height and width must be multiples of 32 for the Wan2.2 backbone")
    return cfg


def _jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _metadata(cfg):
    root = Path(cfg["dataset_dir"]).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError("piper_joint20 reads LeRobot v2.1 with embedded image bytes")
    for key in ("action", "observation.state"):
        feature = info["features"][key]
        if feature.get("shape") != [14] or feature.get("names") != list(FEATURE_NAMES):
            raise ValueError(f"{key} must contain left q1..q6/gripper then right q1..q6/gripper")
    for camera in cfg["camera_layout"]:
        key = f"observation.images.{camera}"
        if info["features"].get(key, {}).get("dtype") != "image":
            raise ValueError(f"{key} must be an embedded image feature")
    if float(info.get("fps", 0)) <= 0:
        raise ValueError("Dataset fps must be positive")
    tasks = {int(row["task_index"]): row["task"] for row in _jsonl(root / "meta/tasks.jsonl")}
    if not tasks or any(not isinstance(text, str) or not text.strip() for text in tasks.values()):
        raise ValueError("tasks.jsonl must contain nonempty task instructions")
    records = []
    for row in _jsonl(root / "meta/episodes.jsonl"):
        index, length = int(row["episode_index"]), int(row["length"])
        if length < 33:
            raise ValueError(f"Episode {index} has {length} rows; at least 33 are required")
        relative = info["data_path"].format(episode_chunk=index // int(info["chunks_size"]), episode_index=index)
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Episode path is outside the dataset: {path}")
        stat = path.stat()
        records.append(Episode(index, path, length, stat.st_size, stat.st_mtime_ns))
    records.sort(key=lambda row: row.index)
    indices = [row.index for row in records]
    if len(set(indices)) != len(indices) or len(records) != int(info["total_episodes"]):
        raise ValueError("Episode metadata has duplicate indices or a total_episodes mismatch")
    if sum(row.length for row in records) != int(info["total_frames"]):
        raise ValueError("Episode metadata does not match total_frames")
    val_ids = list(cfg.get("val_episode_ids") or [])
    if len(set(val_ids)) != len(val_ids) or not set(val_ids).issubset(indices):
        raise ValueError("val_episode_ids contains duplicates or unknown episode indices")
    train_ids = [index for index in indices if index not in val_ids]
    if not train_ids:
        raise ValueError("The split leaves no training episodes")
    meta_hash = hashlib.sha256()
    for filename in ("info.json", "episodes.jsonl", "tasks.jsonl"):
        meta_hash.update((root / "meta" / filename).read_bytes())
    provenance = {
        "schema_version": 1,
        "representation": REPRESENTATION,
        "dataset_dir": str(root),
        "metadata_sha256": meta_hash.hexdigest(),
        "train_episode_ids": train_ids,
        "val_episode_ids": sorted(val_ids),
        "files": [[row.index, row.length, row.size, row.mtime_ns] for row in records],
        "action_alignment": "recorded_action_at_observation_row",
        "units": "dataset_native",
        "gripper_convention": "dataset_native_minmax_no_clipping",
    }
    return info, tasks, records, provenance


def _validate_stats(raw, provenance):
    if raw.get("provenance") != provenance:
        raise ValueError(
            "Stale normalization provenance: dataset or episode split changed; rerun preparation with --force"
        )
    for mode in ("joint20", "joint20_state"):
        stats = raw.get(mode, {})
        if stats.get("representation") != REPRESENTATION:
            raise ValueError(f"{mode} is not a {REPRESENTATION} statistics block")
        for key in STAT_KEYS:
            values = np.asarray(stats.get(key), dtype=np.float32)
            if values.shape != (20,) or not np.isfinite(values).all():
                raise ValueError(f"{mode}.{key} must be a finite 20-D vector")
        if np.any(stats["max"] < stats["min"]) or np.any(stats["std"] <= 0):
            raise ValueError(f"Invalid range or standard deviation in {mode}")
        for key, value in {"min": -1, "max": 1, "q01": -1, "q99": 1, "mean": 0, "std": 1}.items():
            if not np.all(np.asarray(stats[key])[PADDING20] == value):
                raise ValueError(f"{mode}.{key}: padding statistics must be neutral")


def _read_joint_columns(record):
    table = pq.read_table(record.path, columns=["action", "observation.state"], use_threads=False)
    if table.num_rows != record.length:
        raise ValueError(f"{record.path}: row count differs from episodes.jsonl")
    return tuple(pack_joint14(table[key].to_pylist()) for key in ("action", "observation.state"))


def _statistics(arr):
    stats = {
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
        "mean": arr.mean(axis=0, dtype=np.float64).astype(np.float32),
        "std": np.maximum(arr.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32),
        "q01": np.quantile(arr, 0.01, axis=0).astype(np.float32),
        "q99": np.quantile(arr, 0.99, axis=0).astype(np.float32),
        "representation": REPRESENTATION,
        "gripper_convention": "dataset_native_minmax_no_clipping",
    }
    # These are structural zeros, NOT rotation dimensions. All physical joints
    # (including the former rot6d locations) keep their fitted statistics.
    for key, value in {"min": -1, "max": 1, "q01": -1, "q99": 1, "mean": 0, "std": 1}.items():
        stats[key][PADDING20] = value
    return stats


def prepare_normalization_stats(config, *, overwrite: bool = False) -> Path:
    """Fit train-only action/state statistics before distributed workers start.

    The source dataset is read-only. An existing artifact is reused only when
    its provenance and contract match; replacement requires an explicit flag.
    """
    cfg = _config(config)
    _, _, records, provenance = _metadata(cfg)
    destination = Path(cfg["normalization_stats_path"]).resolve()
    if destination.exists() and not overwrite:
        _validate_stats(np.load(destination, allow_pickle=True).item(), provenance)
        return destination
    actions, states = [], []
    for record in records:
        action, state = _read_joint_columns(record)
        if record.index in provenance["train_episode_ids"]:
            actions.append(action)
            states.append(state)
    raw = {
        "joint20": _statistics(np.concatenate(actions)),
        "joint20_state": _statistics(np.concatenate(states)),
        "num_timesteps": sum(len(arr) for arr in actions),
        "provenance": provenance,
    }
    _validate_stats(raw, provenance)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".joint20-", suffix=".npy", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, raw)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


@functools.lru_cache(maxsize=2)
def _episode_table(path, size, mtime_ns, columns):
    # Include the source signature in the cache key. Cache compressed JPEG
    # bytes, never decoded full-episode RGB arrays; bound memory per worker.
    return pq.read_table(path, columns=list(columns), memory_map=True, use_threads=False)


class PiperJoint20Dataset(BaseDataset):
    """Complete-window, row-aligned joint supervision for a single v2.1 dataset."""

    action_dim = UNIFY_DIM
    normalization_stats = None  # Normalized in the reader, never normalize again.

    def __init__(self, config, split="train"):
        self.cfg = _config(config)
        if split not in ("train", "val"):
            raise ValueError(f"Unsupported split {split!r}; expected train or val")
        self.split = split
        self.info, self.tasks, records, provenance = _metadata(self.cfg)
        self.episode_ids = provenance[f"{split}_episode_ids"]
        self.train_episode_ids = provenance["train_episode_ids"]
        self.episodes = [row for row in records if row.index in self.episode_ids]
        if not self.episodes:
            raise ValueError(f"No episodes in {split} split")
        self.normalization_stats_path = str(Path(self.cfg["normalization_stats_path"]).resolve())
        if not Path(self.normalization_stats_path).is_file():
            raise FileNotFoundError("Missing Piper joint20 statistics; run scripts/prepare_piper_joint20.py first")
        raw = np.load(self.normalization_stats_path, allow_pickle=True).item()
        _validate_stats(raw, provenance)
        self._action_stats, self._state_stats = raw["joint20"], raw["joint20_state"]
        self._windows = [
            (ep, start)
            for ep, record in enumerate(self.episodes)
            for start in range(0, record.length - 32, int(self.cfg.get("window_stride", 1)))
        ]
        self._dim_mask = np.zeros(UNIFY_DIM, dtype=bool)
        self._dim_mask[JOINT20_IN_80] = DIM_MASK20
        self._columns = (
            "action",
            "observation.state",
            "task_index",
            *(f"observation.images.{cam}" for cam in self.cfg["camera_layout"]),
        )
        jitter = self.cfg.get("color_jitter")
        self._jitter = None
        if split == "train" and color_jitter_enabled(jitter):
            self._jitter = VideoColorJitter(
                **{k: jitter[k] for k in ("brightness", "contrast", "saturation", "hue") if k in jitter}
            )

    @classmethod
    def from_config(cls, config, split="train"):
        return cls(config, split)

    def __len__(self):
        return len(self._windows)

    def _normalized_unified(self, values, stats):
        padded = pack_joint14(values)
        normalized = apply_normalization(padded, stats, "min-max").astype(np.float32)
        normalized[..., PADDING20] = 0
        return map_to_unify(normalized, JOINT20_IN_80, UNIFY_DIM)[0]

    def __getitem__(self, index):
        ep, start = self._windows[index]
        record = self.episodes[ep]
        table = _episode_table(str(record.path), record.size, record.mtime_ns, self._columns)
        if table.num_rows != record.length:
            raise ValueError(f"{record.path}: row count differs from episodes.jsonl")
        actions = self._normalized_unified(table["action"].slice(start, 32).to_pylist(), self._action_stats)
        state = self._normalized_unified(table["observation.state"].slice(start, 1).to_pylist(), self._state_stats)
        task_index = int(table["task_index"][start].as_py())
        if task_index not in self.tasks:
            raise ValueError(f"{record.path}: task_index {task_index} missing from tasks.jsonl")
        videos = []
        cameras = self.cfg["camera_layout"] if self.cfg.get("multiview", True) else [self.cfg["target_camera"]]
        for frame_index in range(start, start + 33, 4):
            frames = {}
            for camera in cameras:
                item = table[f"observation.images.{camera}"][frame_index].as_py()
                try:
                    if not item or not item.get("bytes"):
                        raise ValueError("Missing embedded image bytes")
                    with Image.open(io.BytesIO(item["bytes"])) as image:
                        frames[camera] = image.convert("RGB")
                except (OSError, ValueError, TypeError) as exc:
                    raise ValueError(f"Cannot decode {record.path}:{camera} row {frame_index}: {exc}") from exc
            height, width = int(self.cfg.get("height", 384)), int(self.cfg.get("width", 320))
            if self.cfg.get("multiview", True):
                image = assemble_multiview_layout(frames, cameras, height, width)
            else:
                image = frames[cameras[0]].resize((width, height), Image.Resampling.BILINEAR)
            videos.append(image)
        if self._jitter is not None:
            videos = self._jitter.apply({"video": videos})["video"]
        return {
            "video": videos,
            "first_frame_image": [videos[0]],
            "vace_video": None,
            "prompt": format_prompt_for_inference(self.tasks[task_index]),
            "action": torch.from_numpy(actions),
            "proprio": torch.from_numpy(state),
            "action_mask": torch.from_numpy(np.broadcast_to(self._dim_mask, (32, UNIFY_DIM)).copy()),
            "proprio_mask": torch.from_numpy(self._dim_mask[None].copy()),
            "video_mask": torch.ones(9, dtype=torch.bool),
        }
