# Model 3 Execution

Model 3 can be trained, resumed, inferred, and evaluated through reusable
scripts. The original notebook remains the completed experiment record.

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
  --resume-dir /kaggle/input/datasets/hypivepiu/model-3-bestbest \
  --output-dir /kaggle/working/vto_v2_production \
  --wandb
```

The resume directory must contain `checkpoint_latest` and `loss_history.json`.
The output directory contains:

```text
/kaggle/working/vto_v2_production/
├── config.yaml
├── metrics.jsonl
├── summary.json
├── checkpoint_latest/
├── checkpoint_best/
└── loss_history.json
```

## Inference Smoke Test

```bash
python scripts/infer.py \
  --config configs/experiments/model_3_default.yaml \
  --checkpoint /kaggle/input/datasets/hypivepiu/model-3-bestbest \
  --person-id 14144_00 \
  --cloth-id 05242_00 \
  --output /kaggle/working/model_3_smoke_result.jpg
```

## Paired Evaluation Smoke Test

```bash
python scripts/evaluate.py \
  --config configs/experiments/model_3_default.yaml \
  --checkpoint /kaggle/input/datasets/hypivepiu/model-3-bestbest \
  --max-samples 3 \
  --output-dir /kaggle/working/model_3_paired_smoke
```

Large checkpoints and generated galleries should be published as Kaggle
Datasets instead of committed to Git.
