# Image Patch Reconstruction

A ResNet encoder, transformer with cross-attention, and U-Net inpainter that reconstruct scrambled 96x96 image patches. Reaches MAE 0.033 against a 0.18 baseline.

> **Result:** _add headline figure or metric here (keep it in the first screen)._

![result](figures/.gitkeep)

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce

```bash
python src/train.py --config configs/default.yaml
```

## Layout

See the directory tree below; notebooks in `notebooks/` give a short walkthrough.

## Notes

_Background, references, and method notes go here._
