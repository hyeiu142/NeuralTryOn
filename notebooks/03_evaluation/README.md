# Evaluation

Quantitative evaluation uses paired reconstruction on
`clean_vto_dataset_test.csv` with `cloth_id = person_id`.

Qualitative comparison and error analysis use original unpaired pairs from
`holdout_test.csv`.

The complete SD + LoRA workflow lives under
`evaluation/model_3_sd_lora/`. Run it through
`model_3_sd_lora_evaluation.ipynb`.

The Model 1 evaluation workflow lives under
`evaluation/model_1_lightweight_unet/`. Current completed Model 1 and Model 3
results are summarized by `evaluation/common_comparison/`.
