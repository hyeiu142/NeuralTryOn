# Model 3 Production Training

The Model 3 training implementation has been migrated from
`notebooks/02_models/model_3_sd_lora.ipynb` into importable production modules.
The notebook remains unchanged as the original experiment record.

The adapter has passed local configuration, import, shape, and unit tests. Its
status remains `production_adapter_gpu_smoke_pending` until the first Kaggle GPU
smoke test completes successfully.

## Migrated Components

```text
src/models/model_3_sd_lora/
├── settings.py       Typed resolved configuration
├── dataset.py        Identity-paired VITON-HD preprocessing
├── modules.py        Perceiver, spatial projector, and 17-channel UNet expansion
├── components.py     Stable Diffusion, CLIP, LoRA, and optimizer construction
├── objective.py      Masked diffusion, x0, and Min-SNR objective
├── checkpoints.py    Notebook-compatible checkpoint layout
└── trainer.py        Accelerate training and validation lifecycle
```

The adapter preserves the notebook's important contracts:

- `runwayml/stable-diffusion-inpainting`
- LoRA rank and alpha `16`
- Expanded `17`-channel UNet input
- Eight global garment tokens and 64 spatial tokens
- Mask-weighted noise and x0 reconstruction losses
- Min-SNR weighting, gradient accumulation, EMA, and early stopping
- `checkpoint_latest` and `checkpoint_best` artifact layout

## Kaggle Smoke Test

Clone the repository into a Kaggle GPU notebook, then run:

```bash
pip install -e .

python scripts/train.py \
  --config configs/experiments/model_3_default.yaml \
  --epochs 1 \
  --max-train-samples 8 \
  --max-validation-samples 4 \
  --output-dir /kaggle/working/model_3_smoke_test
```

This performs real forward and backward passes using only a few samples. It is
the recommended first check before starting a full run.

## Full Training

```bash
python scripts/train.py \
  --config configs/experiments/model_3_default.yaml \
  --output-dir /kaggle/working/vto_v2_production \
  --wandb
```

## Resume from a Published Checkpoint

Attach the checkpoint Kaggle Dataset and run:

```bash
python scripts/train.py \
  --config configs/experiments/model_3_default.yaml \
  --resume-dir /kaggle/input/datasets/khoaanh1234/ckpt-epoch-12-yen \
  --output-dir /kaggle/working/vto_v2_production \
  --wandb
```

The resume directory must contain `checkpoint_latest` and `loss_history.json`.

## Output Contract

```text
/kaggle/working/vto_v2_production/
├── config.yaml
├── metrics.jsonl
├── summary.json
├── checkpoint_latest/
├── checkpoint_best/
└── loss_history.json
```

Large checkpoints should be published as a Kaggle Dataset rather than committed
to Git.

## Inference Smoke Test

```bash
python scripts/infer.py \
  --config configs/experiments/model_3_default.yaml \
  --checkpoint /kaggle/input/datasets/khoaanh1234/ckpt-epoch-12-yen \
  --person-id 14144_00 \
  --cloth-id 05242_00 \
  --output /kaggle/working/model_3_smoke_result.jpg
```

## Paired Evaluation Smoke Test

```bash
python scripts/evaluate.py \
  --config configs/experiments/model_3_default.yaml \
  --checkpoint /kaggle/input/datasets/khoaanh1234/ckpt-epoch-12-yen \
  --max-samples 3 \
  --output-dir /kaggle/working/model_3_paired_smoke
```

Remove `--max-samples 3` only after the smoke test succeeds.
