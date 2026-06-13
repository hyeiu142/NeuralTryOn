# Evaluation

Quantitative evaluation uses paired reconstruction on
`clean_vto_dataset_test.csv` with `cloth_id = person_id`.

Qualitative comparison and error analysis use original unpaired pairs from
`holdout_test.csv`.

The SD + LoRA evaluation runner executes the reusable cells under
`evaluation_cells/model_3_sd_lora/`.
