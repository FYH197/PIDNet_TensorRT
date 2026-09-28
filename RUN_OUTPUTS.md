# Training run outputs

Every new training launch writes to an isolated directory:

```text
output/<dataset>/<model>_<dataset>_<init>_<seed>_<lr>_<timestamp>/
```

Example:

```text
output/indoor/pidnet-s_sun15_img_s304_lr0p01_20260824-153000/
```

The directory contains:

- `best_miou.pth`: best validation-mIoU model weights for evaluation/export.
- `last.ckpt`: latest full training state for resume.
- `final.pth`: final-epoch model weights.
- `config.yaml`: exact effective configuration.
- `metrics.json`: machine-readable run result.
- `summary.txt`: concise human-readable result.
- `logs/train.log`: training log.
- `logs/events.out.tfevents*`: TensorBoard data.

The dataset directory also contains `experiments.csv`, updated when a run
finishes.

An automatic name is used by default:

```powershell
python tools/train.py --cfg configs/yaml/pidnet_s_sunrgbd15_official_strong.yaml --seed 304
```

An optional concise prefix can be supplied:

```powershell
python tools/train.py --cfg configs/yaml/pidnet_s_sunrgbd15_official_strong.yaml --seed 304 --run-name pidnet-s_sun15_img_s304_lr0p01
```

Resume requires both `TRAIN.RESUME true` and the existing run directory:

```powershell
python tools/train.py --cfg configs/yaml/pidnet_s_sunrgbd15_official_strong.yaml --seed 304 --run-dir output/indoor/<run-id> TRAIN.RESUME true
```

Legacy evaluation remains compatible with `best.pt`. For new runs, pass the
new weight explicitly through `TEST.MODEL_FILE`.
