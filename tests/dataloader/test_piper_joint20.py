"""Real LeRobot v2.1 fixtures: joint semantics, image decoding, and split isolation."""

import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from openwam.dataloader.registry import build_dataset

ACTIVE20 = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 19]
ACTIVE80 = [0, 1, 2, 3, 4, 5, 9, 34, 35, 36, 37, 38, 39, 43]
CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
NAMES = [f"{arm}_{joint}" for arm in ("left", "right") for joint in (*[f"j{i}" for i in range(1, 7)], "gripper")]


def jpeg(color):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buffer, format="JPEG", quality=100, subsampling=0)
    return buffer.getvalue()


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    features = {key: {"dtype": "float32", "shape": [14], "names": NAMES} for key in ("action", "observation.state")}
    features.update({f"observation.images.{cam}": {"dtype": "image", "shape": [3, 24, 32]} for cam in CAMERAS})
    info = {
        "codebase_version": "v2.1",
        "robot_type": "piper_bimanual",
        "total_episodes": 3,
        "total_frames": 111,
        "fps": 30,
        "chunks_size": 1000,
        "features": features,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "splits": {"train": "0:3"},
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "put the pen into the pen holder"}) + "\n"
    )
    episodes = []
    for ep in range(3):
        rows = []
        for t in range(37):
            offset = ep * 10 if ep < 2 else 1000
            row = {
                "action": [(offset + t - 5) * (d + 1) for d in range(14)],
                "observation.state": [100 + (offset + t) * (d + 2) for d in range(14)],
                "timestamp": t / 30,
                "frame_index": t,
                "episode_index": ep,
                "task_index": 0,
            }
            for axis, cam in enumerate(CAMERAS):
                color = [0, 0, 0]
                color[axis] = 100 + t
                row[f"observation.images.{cam}"] = {"bytes": jpeg(tuple(color)), "path": f"frame_{t:06d}.jpg"}
            rows.append(row)
        pq.write_table(pa.Table.from_pylist(rows), root / f"data/chunk-000/episode_{ep:06d}.parquet")
        episodes.append({"episode_index": ep, "tasks": ["put the pen into the pen holder"], "length": 37})
    (root / "meta/episodes.jsonl").write_text("\n".join(json.dumps(row) for row in episodes) + "\n")
    return root


@pytest.fixture
def cfg(source, tmp_path):
    return OmegaConf.create(
        {
            "type": "piper_joint20",
            "dataset_dir": str(source),
            "val_episode_ids": [2],
            "normalization_stats_path": str(tmp_path / "prepared/normalization_stats.npy"),
            "action_mode": "joint20",
            "action_representation": "piper_joint20_v1",
            "unify_action": True,
            "unify_action_map": ["0-9", "34-43"],
            "unify_state_map": None,
            "action_dim_mask": [i in ACTIVE20 for i in range(20)],
            "proprio_dim_mask": [i in ACTIVE20 for i in range(20)],
            "num_frames": 33,
            "video_stride": 4,
            "window_stride": 1,
            "height": 384,
            "width": 320,
            "multiview": True,
            "camera_layout": CAMERAS,
            "target_camera": "cam_high",
            "normalize_mode": "min-max",
            "color_jitter": {"enabled": False},
        }
    )


def prepare(cfg):
    from openwam.dataloader.piper_joint20 import prepare_normalization_stats

    return prepare_normalization_stats(cfg)


def test_registry_reads_joint_targets_in_source_row_order(cfg):
    prepare(cfg)
    ds = build_dataset(cfg, split="train")
    assert len(ds) == 10  # Two 37-frame episodes, five complete windows each.
    sample = ds[0]
    assert sample["action"].shape == (32, 80)
    assert sample["proprio"].shape == (1, 80)
    assert sample["action_mask"].dtype == torch.bool
    assert sample["proprio_mask"].dtype == torch.bool
    assert sample["action_mask"].sum().item() == 32 * 14
    assert sample["proprio_mask"].sum().item() == 14
    assert sample["action_mask"][0].nonzero().flatten().tolist() == ACTIVE80
    assert sample["proprio_mask"][0].nonzero().flatten().tolist() == ACTIVE80
    np.testing.assert_allclose(sample["action"][0, ACTIVE80], -1.0, atol=1e-6)
    np.testing.assert_allclose(sample["action"][1, ACTIVE80], -1 + 2 / 46, atol=1e-6)
    np.testing.assert_allclose(sample["action"][-1, ACTIVE80], -1 + 62 / 46, atol=1e-6)
    np.testing.assert_allclose(sample["proprio"][0, ACTIVE80], -1.0, atol=1e-6)
    assert torch.count_nonzero(sample["action"][~sample["action_mask"]]) == 0
    assert torch.count_nonzero(sample["proprio"][~sample["proprio_mask"]]) == 0
    assert sample["prompt"].endswith("put the pen into the pen holder")


def test_images_decode_from_embedded_bytes_and_keep_camera_and_time_order(cfg):
    prepare(cfg)
    sample = build_dataset(cfg)[0]
    assert len(sample["video"]) == 9
    assert sample["video_mask"].tolist() == [True] * 9
    assert sample["first_frame_image"][0] is sample["video"][0]
    assert sample["video"][0].size == (320, 384)
    np.testing.assert_allclose(sample["video"][0].getpixel((160, 128)), [100, 0, 0], atol=2)
    np.testing.assert_allclose(sample["video"][0].getpixel((80, 320)), [0, 100, 0], atol=2)
    np.testing.assert_allclose(sample["video"][0].getpixel((240, 320)), [0, 0, 100], atol=2)
    np.testing.assert_allclose(sample["video"][-1].getpixel((160, 128)), [132, 0, 0], atol=2)


def test_statistics_use_training_episodes_and_separate_state_from_action(cfg):
    path = prepare(cfg)
    raw = np.load(path, allow_pickle=True).item()
    np.testing.assert_allclose(raw["joint20"]["min"][0], -5)
    np.testing.assert_allclose(raw["joint20"]["max"][0], 41)
    np.testing.assert_allclose(raw["joint20_state"]["min"][0], 100)
    np.testing.assert_allclose(raw["joint20_state"]["max"][0], 192)
    # q4 occupies what used to be a rot6d slot. It must use joint statistics.
    np.testing.assert_allclose(raw["joint20"]["min"][3], -20)
    np.testing.assert_allclose(raw["joint20"]["max"][3], 164)
    val = build_dataset(cfg, split="val")
    assert len(val) == 5
    assert val.episode_ids == [2]
    assert build_dataset(cfg).episode_ids == [0, 1]
    np.testing.assert_allclose(val[0]["action"][0, ACTIVE80], 1.0)
    assert prepare(cfg) == path  # Identical preparation can be rerun safely.


def test_stale_statistics_cannot_be_reused_after_split_changes(cfg):
    prepare(cfg)
    cfg.val_episode_ids = [1]
    with pytest.raises(ValueError, match="(provenance|split|stale)"):
        build_dataset(cfg)


def test_missing_stats_requires_explicit_preparation(cfg):
    with pytest.raises(FileNotFoundError, match="prepare"):
        build_dataset(cfg)


@pytest.mark.parametrize("field,value", [("action_mode", "eef"), ("num_frames", 17), ("video_stride", 2)])
def test_reader_rejects_a_different_action_or_temporal_contract(cfg, field, value):
    cfg[field] = value
    with pytest.raises(ValueError, match=field):
        prepare(cfg)


def test_nonfinite_source_values_fail_before_training(cfg, source):
    path = source / "data/chunk-000/episode_000000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["action"][0] = float("nan")
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="finite"):
        prepare(cfg)


def test_joint_pack_unpack_preserves_gripper_sign_and_padding():
    from openwam.dataloader.utils.piper_joint20 import pack_joint14, unpack_joint20

    raw = np.array([1, 2, 3, 4, 5, 6, -0.003, 7, 8, 9, 10, 11, 12, 0.071], dtype=np.float32)
    padded = pack_joint14(raw)
    np.testing.assert_array_equal(padded[[6, 7, 8, 16, 17, 18]], 0)
    np.testing.assert_array_equal(padded[ACTIVE20], raw)
    np.testing.assert_array_equal(unpack_joint20(padded), raw)
    with pytest.raises(ValueError):
        pack_joint14(np.zeros(20))


def test_corrupt_camera_is_an_error_instead_of_a_silent_black_tile(cfg, source):
    path = source / "data/chunk-000/episode_000000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["observation.images.cam_left_wrist"]["bytes"] = b"not an image"
    pq.write_table(pa.Table.from_pylist(rows), path)
    prepare(cfg)
    with pytest.raises(ValueError, match="cam_left_wrist"):
        build_dataset(cfg)[0]


def test_deployment_and_reader_share_joint_coordinates_and_padding(cfg, tmp_path):
    from openwam.deploy.model_loader import _build_normalizer
    from openwam.model.architectures.base import BaseWAMArchitecture

    path = prepare(cfg)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "normalization_stats.npy").write_bytes(Path(path).read_bytes())
    normalizer = _build_normalizer(OmegaConf.create({"dataloader": cfg}), str(checkpoint))
    state20 = np.zeros(20, np.float32)
    state20[ACTIVE20] = 100
    # Ignore garbage in padding supplied by an external client.
    state20[[6, 7, 8, 16, 17, 18]] = 999
    np.testing.assert_allclose(normalizer.normalize(state20), build_dataset(cfg)[0]["proprio"][0], atol=1e-6)
    model_action = build_dataset(cfg)[0]["action"][0].numpy().copy()
    model_action[[6, 7, 8, 40, 41, 42]] = 0.7
    result = normalizer.unnormalize(model_action)
    assert result.shape == (20,)
    # Float32 inverse scaling over this fixture's intentionally large ranges.
    np.testing.assert_allclose(result[ACTIVE20], [-5 * (d + 1) for d in range(14)], atol=3e-5)
    np.testing.assert_array_equal(result[[6, 7, 8, 16, 17, 18]], 0)

    # Use the real inference-mask resolver without constructing the 5B backbone.
    class Architecture:
        action_dim = 80

    arch = Architecture()
    arch.normalizer = normalizer
    inactive = BaseWAMArchitecture._resolve_inactive_action_dims(arch, None, torch.device("cpu"))
    assert (~inactive).nonzero().flatten().tolist() == ACTIVE80
