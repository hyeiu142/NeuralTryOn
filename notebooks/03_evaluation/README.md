# Evaluation

Quantitative evaluation uses paired reconstruction on
`clean_vto_dataset_test.csv` with `cloth_id = person_id`.

Qualitative comparison and error analysis use original unpaired pairs from
`holdout_test.csv`.

The complete SD + LoRA workflow lives under
`evaluation/model_3_sd_lora/`. Run it through
`model_3_sd_lora_evaluation.ipynb`.
