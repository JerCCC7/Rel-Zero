# Training Notes

The main training entry point is:

```bash
python tools/train_relzero.py \
  --coco-root path/to/coco/train2014 \
  --coco-ann path/to/coco/annotations/captions_train2014.json \
  --checkpoint-dir checkpoints \
  --run-name relzero_train \
  --device cuda:0
```

The script follows the training recipe described in the paper:

1. Extract ViT patch features from original images and VAE-reconstructed images.
2. Rank patch pairs by the stability of their relational distance.
3. Train the pair predictor to recover the stable top-k patch pairs from the
   original image alone.

Recommended defaults used by the release:

- image size: `224`
- patch grid: `14 x 14`
- top-k relational watermark length: `50`
- negative matching probability for TPR@0.1%FPR: `0.06`

The evaluation tools in `tools/` are checkpoint-compatible with the released
`RelZeroPipeline` state dict format.
