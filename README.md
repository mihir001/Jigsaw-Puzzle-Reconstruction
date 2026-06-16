# Jigsaw Puzzle Reconstruction

Reconstructs a 96×96 image from 9 scrambled, eroded 28×28 patches — a shared CNN encodes each patch, a transformer lets patches reason about each other, a DETR-style attention layer infers where each one belongs, and a U-Net inpaints the resulting gaps.

> **Result:** Final MAE **0.0301 ± 0.0010**, an **83.5% reduction** over a 0.1824 mean-patch baseline, with **3.44M** trainable parameters.

## Problem

Each 96×96 RGB image is split into a 3×3 grid of patches, each cropped from 32×32 down to 28×28 (a 2px erosion border per side), then shuffled into a random order. The model has to recover both the correct placement of the 9 patches and the visual content removed by the erosion, without ever being told which patch came from where. Trained and evaluated on STL-10 (100k unlabeled 96×96 images).

## Approach

A shared CNN encodes each of the 9 patches independently into a 256-dim token, using the same weights for every patch. Three transformer self-attention blocks then let each token update itself based on the other eight deliberately with no positional encoding, since the input slot order carries no real information here. A small attention layer with 9 learned position queries attends over the 9 tokens to produce a soft assignment of patches to grid cells. Those assigned tiles are scattered onto a 96×96 canvas. True pixels where a patch landed, zeros in the eroded gaps and a U-Net inpaints the rest, conditioned on a binary mask marking which pixels are real. The final image keeps the exact patch pixels and uses the U-Net's output only inside the gaps.

## Training

Training runs in two phases. Phase 1 (40 epochs, lr 1e-3) optimizes reconstruction MAE jointly with a per-slot position classification loss, which gives the transformer a much more direct positioning signal than MAE alone. A second auxiliary loss — classifying the relative compass direction between each of the 36 patch pairs, following the pairwise-permutation task in Zhao & Dong (2020) — is held at zero weight in Phase 1 to avoid early gradient conflicts, then enabled in Phase 2 (20 epochs, lr 3e-5) for fine-tuning. Random horizontal flips, applied before patch extraction during training only, closed a train/validation gap caused by the model memorizing specific images rather than general positioning rules.

## Results

| | MAE | std |
|---|---|---|
| Baseline (mean patch) | 0.1824 | 0.0007 |
| Final model | 0.0301 | 0.0010 |

83.5% improvement over baseline. 3,439,572 trainable parameters.

## Run it

Open the notebook above and run top to bottom — it downloads STL-10 automatically and reproduces both training phases.

To run locally instead:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/jigsaw_final.ipynb
```

Skip the `drive.mount` cell if running outside Colab; training from scratch (no Drive access needed) reproduces the same result without the checkpoint downloads.

## Notes

- Zhao, Q., & Dong, J. (2020). *Self-supervised Representation Learning by Predicting Visual Permutations* — the pairwise relative-direction auxiliary task.
- Carion et al. (2020). *End-to-End Object Detection with Transformers (DETR)* — the learned positional-query mechanism used for patch-to-slot assignment.
