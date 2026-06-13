# Virtual Try-On: Deep Learning Final Project

Đồ án xây dựng và so sánh ba pipeline Virtual Try-On trên bộ dữ liệu
[VITON-HD](https://www.kaggle.com/datasets/marquis03/high-resolution-viton-zalando-dataset).
Trọng tâm của dự án là tiền xử lý dữ liệu, huấn luyện trong điều kiện GPU hạn chế,
đánh giá định lượng và phân tích lỗi.

## Models

| Model | Pipeline |
| --- | --- |
| Model 1 | Lightweight U-Net mask → GMM warp → TOM generator |
| Model 2 | GMM → shape generation → Pix2Pix texture fusion + PatchGAN |
| Model 3 | Stable Diffusion Inpainting + LoRA + pose/cloth conditioning |

## Project Structure

```text
VTO/
├── notebooks/
│   ├── 01_data_pipeline/    # EDA, cleaning, captioning, SD validation
│   ├── 02_models/           # Three main training notebooks
│   ├── 03_evaluation/       # Quantitative and qualitative evaluation
│   └── 04_demo/             # Demo integration notes
├── evaluation/              # Complete model-specific evaluation workflows
├── src/                     # Shared metrics, visualization, reproducibility
├── results/                 # EDA, metrics, convergence, comparisons, failures
├── docs/                    # Dataset, architecture, checkpoint documentation
```

`data/` and `checkpoints/` are local-only and intentionally excluded from Git.
Local development drafts under `archive/` are also excluded.

## Dataset Splits

| Split | Usage |
| --- | --- |
| `clean_vto_dataset_train.csv` | Model training |
| `clean_vto_dataset_test.csv` | Paired reconstruction with `cloth_id = person_id`; SSIM, PSNR, LPIPS |
| `holdout_test.csv` | Unpaired try-on; qualitative comparison and error analysis |

The test split was also used as validation during training. Results on it are
therefore reported as paired reconstruction evaluation, while holdout results
demonstrate practical cross-garment try-on behavior.

## Kaggle Workflow

1. Add the VITON-HD, cleaned CSV, caption, and required checkpoint datasets.
2. Open the desired notebook under `notebooks/`.
3. Run cells from top to bottom.
4. Save generated artifacts from `/kaggle/working` as a private Kaggle Dataset.

Recommended execution order:

```text
notebooks/01_data_pipeline/01_eda_and_cleaning.ipynb
notebooks/01_data_pipeline/02_blip_captioning.ipynb
notebooks/02_models/model_1_lightweight_unet.ipynb
notebooks/02_models/model_2_pix2pix.ipynb
notebooks/02_models/model_3_sd_lora.ipynb
notebooks/03_evaluation/model_3_sd_lora_evaluation.ipynb
```

See [docs/dataset.md](docs/dataset.md), [docs/architecture.md](docs/architecture.md),
and [docs/checkpoints.md](docs/checkpoints.md) for details.

## Local Setup

```bash
python -m pip install -r requirements.txt
```

Large-scale training and inference require a CUDA GPU and are intended to run
on Kaggle.
