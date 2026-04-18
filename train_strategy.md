# Event Training Strategy (BS-ERGB)

## Dataset Layout
- Root: `G:/bs_ergb`
- Splits:
  - `1_TEST/`
  - `2_VALIDATION/`
  - `3_TRAINING/`
- Per-scene structure:
  - `scene_xxx/images/` (RGB frames)
  - `scene_xxx/events/` (event files)

## Event Feature Construction
- `dataset_events.py` uses `event_process` utilities directly:
  - `event_process.event.load_events`
  - `event_process.event.EventSequence`
  - `event_process.representation.to_voxel_grid`
- Event files are selected by timestamp range between input frame pair (`img0`, `img1`) with index fallback.
- Voxelized event tensor is returned as `event_feat: [B, Ce, H, W]` after batching (`Ce = num_bins`, default 5).
- Training augmentations apply identical spatial transforms to RGB and event voxel to keep alignment.

## Model Integration
- Event fusion is inserted in `MotionFormer.forward` after `patch_embed` and `get_cor`.
- Event is injected into:
  - appearance tokens `x`
  - motion prior tokens `cor`
- `InterFrameAttention` logic remains unchanged.
- `event_feat=None` keeps original RGB-only path.

## Strategy B (Partial Fine-tuning)
- `freeze_backbone()` freezes:
  - `block1`, `block2`, `block3`
  - `patch_embed1`, `patch_embed2`, `patch_embed3`
  - `CrossScalePatchEmbed` (first transformer-stage patch embed)
- Keeps trainable:
  - last 1-2 transformer stages (`MotionFormerBlock`)
  - corresponding norm layers
  - `event_fusion`

## Optimization
- Start from pretrained RGB checkpoint (`ckpt/ours.pkl`).
- Optimize only trainable parameters:

```python
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()))
```

- Original loss remains unchanged (`LapLoss` + merged supervision).

## Recommended Entry Script
- Use `train_event.py` (does not overwrite original `train.py`).
- Example:

```bash
python train_event.py --data_root G:/bs_ergb --pretrained_name ours --batch_size 8
```
