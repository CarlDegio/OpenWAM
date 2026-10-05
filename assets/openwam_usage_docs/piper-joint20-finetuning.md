# Piper joint-space fine-tuning from OpenWAM-Alpha

This recipe uses the downloaded Alpha Foundation and the local LeRobot v2.1
dataset `piper_bimanual_h50`. It is a **custom joint-space adaptation**, not the
official EEF Piper policy. The original 80-D action and state tensors are kept
so all Foundation parameter shapes remain compatible; XYZ/rotation slots are
relearned as joint targets during fine-tuning.

## Data and action contract

The inspected dataset has 31 episodes, 23,610 frames at 30 Hz (about 13.1 minutes),
and the instruction `put the pen into the pen holder`. Three RGB cameras
(`cam_high`, `cam_left_wrist`, `cam_right_wrist`) are embedded as JPEG bytes in
episode Parquet files. No LeRobot package, video conversion or separate vision
encoder download is required.

Native action and state order is:

```text
[left_q1..q6, left_gripper, right_q1..q6, right_gripper]  # 14
```

The physical 20-D interface is:

```text
[left_q1..q6, 0, 0, 0, left_gripper,
 right_q1..q6, 0, 0, 0, right_gripper]                  # 20
```

The left/right 10-D blocks map into unified slices `0:10` / `34:44` (Python
half-open indexing). Only unified indices `[0,1,2,3,4,5,9,34,35,36,37,38,39,43]`
are valid for action loss. All remaining values are zero and masks are false.
The configuration explicitly records `action_mode: joint20` and
`action_representation: piper_joint20_v1`; never label these vectors as EEF.

All fourteen physical dimensions use min-max normalization, including joints
occupying former rot6d positions. **No rotation-identity pinning is applied.**
Action statistics and proprio statistics are separate. Raw negative gripper
readings are preserved, with no presumed meters/radians conversion or physical
zero clipping. Min-max normalization itself clips to `[-1,1]`, consistent with
the training framework; the extrema are empirical training-data extrema, not
calibrated hardware limits. Deployment uses the inverse action transform and
the matching state transform, and returns exact zeros in interface padding.

The reader treats `action[t]` as the recorded joint target associated with
`observation[t]`. It does not substitute `state[t+1]` or shift action rows.
Confirm this recorded-command convention against the collection code before
using a different dataset. No FK/IK conversion is included.

Each complete 33-row window emits action rows `t:t+32`, state row `t`, and video
rows `t,t+4,...,t+32`. Thus a chunk has 32 actions (about 1.07 seconds at 30 Hz)
and nine images. Incomplete tail windows are excluded. Images use the same
384×320 three-view canvas as the official recipe: head above, wrists below.
Prompts use the repository's standard robot-view instruction prefix.

## Prepare and check without training

From the project directory:

```bash
cd /mnt/workspace/liuzihao/project/OpenWAM
bash scripts/train_piper_joint20.sh --prepare-only
```

This reads the dataset without modifying it and writes:

```text
assets/piper_joint20_h50/
  normalization_stats.npy
  dataset_report.json
  prepared_config.yaml
  preview_train.jpg
  preview_val.jpg
```

Episodes 28–30 are held out; episodes 0–27 alone determine statistics. There
are 20,691 train windows and 1,927 held-out windows at stride 1. The source
file signatures, feature schema and split are recorded in statistics
provenance. A changed dataset or split requires explicit regeneration:

```bash
bash scripts/train_piper_joint20.sh --prepare-only --force
```

Existing `joint14` statistics under the source dataset are not used. Do not
regenerate statistics during a run or to resume an existing checkpoint with
different normalization coordinates.

To view the complete training configuration without starting a job:

```bash
.venv/bin/python scripts/train.py --config-name train_piper_joint20 --cfg job --resolve
```

## Single-node, eight-H20 training

```bash
cd /mnt/workspace/liuzihao/project/OpenWAM
tmux new -s piper-joint20-train
bash scripts/train_piper_joint20.sh
```

The launcher selects the existing `.venv`, sets one node/eight processes, and
uses the checkpoint at:

```text
/mnt/workspace/base_model/OpenWAM-Alpha-Pretrain-Foundation-Model
```

Preparation runs before `torchrun`. Checkpoints are saved to a run directory
under `outputs/piper_joint20/`; read the `Checkpoints will be saved to ...`
startup line for the exact location. W&B is offline by default; set
`WANDB_MODE=online` explicitly if remote logging is wanted. Hub access is also
offline because the Foundation contains its weights and tokenizer assets.

Starting hyperparameters:

| Setting | Value |
|---|---|
| Precision / optimizer sharding | BF16 / DeepSpeed ZeRO-2 |
| Batch / accumulation | 2 per GPU / 1 (global batch 16) |
| Learning rates | video 1e-5; action and other trainable modules 5e-5 |
| Epochs | 5 |
| Gradient checkpointing | enabled |
| DataLoader workers | 2 per GPU |
| Frozen modules | VAE and text encoder, as in the standard dual-system config |
| Checkpoints | every 1,000 microsteps; keep two; save full resume state |

These are conservative starting parameters, not a measured memory/throughput
or convergence guarantee. With only 31 short demonstrations, choose duration
using held-out evaluation and robot performance instead of copying the
official 30,000-step schedule. **The current trainer does not automatically
evaluate the held-out split.** This reader reserves those episodes and exposes
`build_dataset(cfg.dataloader, split="val")` for subsequent evaluation; it
does not select a "best validation" checkpoint.

A short optional training smoke run (this command really starts training):

```bash
bash scripts/train_piper_joint20.sh \
  training.max_steps=20 \
  training.batch_size=1 \
  training.save_steps=0 \
  training.save_full_states_for_resume=false \
  training.output_path=outputs/piper_joint20_smoke
```

With `save_steps=0`, the current trainer skips both periodic and final weights.
After checking finite loss, GPU memory and step time, start the full run with the earlier
command. To increase per-GPU batch size after measuring memory:

```bash
bash scripts/train_piper_joint20.sh training.batch_size=4
```

### Measured eight-H20 batch-4 smoke run (2026-09-09)

Twenty training steps completed with exit code 0 using BF16, ZeRO-2,
gradient checkpointing and accumulation 1. The launcher's CUDA detection
fallback skips DeepSpeed's optional compiler probes when `nvcc` is absent;
this recipe uses PyTorch AdamW and no compiled DeepSpeed CUDA operators.

For 20,691 training windows, Accelerate pads the final distributed batch:

| Per-GPU batch | Global batch | Steps per epoch | Steps for five epochs |
|---|---|---|---|
| 2 | 16 | 1,294 | 6,470 |
| 4 | 32 | 647 | 3,235 |

GPU memory was sampled through `nvidia-smi` every 0.5 seconds, including
model loading and optimizer updates. Observed peaks on GPUs 0–7 were
45.31, 45.33, 45.34, 45.36, 45.44, 45.34, 45.33 and 42.98 GiB respectively
(97,871 MiB available per card). Sampling may miss brief peaks. This run
did not save checkpoints or test a full epoch. All GPUs were released afterward.
Final total/video/action losses were 0.18211 / 0.15223 / 0.02988.

After warm-up, steps took approximately 1.5–1.8 seconds. Extrapolating this
short run gives about 81–97 minutes for 3,235 steps, excluding model loading
and checkpoint writes; longer-run data loading and throughput may differ.
The complete probe took 289 seconds, mostly model initialization/loading.
Logs, per-GPU samples and summary are in `outputs/piper_joint20_batch4_probe/`.

The subsequent **batch-16** probe also completed all 20 steps with exit code 0
under the same training settings. Global batch is 128, giving 162 steps per
epoch and **810 steps for five epochs**. Sampled GPU peaks were 45.19 GiB on
GPUs 0–3 and 5–7, and 47.52 GiB on GPU 4. Post-warm-up steps took about
4.8–5.0 seconds: a short-run extrapolation is 65–68 minutes for five epochs,
excluding loading and checkpoint writes. Final total/video/action losses
were 0.18722 / 0.16947 / 0.01774. This verifies execution, not convergence or
equivalent quality across batch sizes. The same sampling/full-epoch limits
apply. Records are in `outputs/piper_joint20_batch16_probe/`.

To resume a run with a saved full optimizer state, pass its actual run
directory and clear the Foundation warm-start path:

```bash
bash scripts/train_piper_joint20.sh \
  training.finetune_ckpt_path=null \
  training.resume_ckpt_path=/absolute/path/to/the/saved/run
```

## Deployment interface

The standard server expects a **physical 20-D state** and returns **physical
20-D actions** for this recipe. At the robot client boundary, use:

```python
from openwam.dataloader.utils.piper_joint20 import pack_joint14, unpack_joint20

state20 = pack_joint14(measured_state14)
# Send state20 in obs["state"], the three cameras using the configured names,
# and the same formatted instruction used in training.
# actions20 = ... receive the policy server response ...
joint_targets14 = unpack_joint20(actions20)
```

Do not send these 20-D outputs to an EEF/rot6d client. The joint helper does
not enforce mechanical limits, add a trajectory controller, or transform
units: execution must match the data collection interface. A complete
robot-specific rollout client is outside this data/finetuning change.
