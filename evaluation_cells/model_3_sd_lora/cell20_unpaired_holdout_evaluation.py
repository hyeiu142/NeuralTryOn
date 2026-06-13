import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm.auto import tqdm


UNPAIRED_SEED = 2026
UNPAIRED_DIR = OUTPUT_DIR / "unpaired_evaluation"
UNPAIRED_IMAGE_DIR = UNPAIRED_DIR / "images"
for directory in [UNPAIRED_DIR, UNPAIRED_IMAGE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

holdout_csv = CSV_ROOT / "holdout_test.csv"
holdout_df = pd.read_csv(holdout_csv)
required_columns = {"person_id", "cloth_id"}
missing_columns = required_columns - set(holdout_df.columns)
if missing_columns:
    raise ValueError(f"{holdout_csv} thieu cot: {sorted(missing_columns)}")

holdout_df["person_id"] = holdout_df["person_id"].astype(str)
holdout_df["cloth_id"] = holdout_df["cloth_id"].astype(str)
holdout_df = holdout_df.reset_index(drop=True)
holdout_df["sample_index"] = holdout_df.index
holdout_df["seed"] = UNPAIRED_SEED + holdout_df.index
manifest_path = UNPAIRED_DIR / "unpaired_manifest.csv"
holdout_df.to_csv(manifest_path, index=False)

metrics_path = UNPAIRED_DIR / "unpaired_metrics.csv"
if metrics_path.exists():
    try:
        metrics_df = pd.read_csv(metrics_path)
    except pd.errors.EmptyDataError:
        metrics_df = pd.DataFrame()
else:
    metrics_df = pd.DataFrame()

required_result_columns = {"person_id", "cloth_id", "status"}
if len(metrics_df) > 0 and required_result_columns.issubset(metrics_df.columns):
    metrics_df["person_id"] = metrics_df["person_id"].astype(str)
    metrics_df["cloth_id"] = metrics_df["cloth_id"].astype(str)
    metrics_df = metrics_df.drop_duplicates(
        subset=["person_id", "cloth_id"],
        keep="last",
    )
else:
    metrics_df = pd.DataFrame()

finished_keys = {
    (str(row.person_id), str(row.cloth_id))
    for row in metrics_df.itertuples(index=False)
    if row.status == "ok"
}
records = metrics_df.to_dict("records")


def _save_np_image(image_np, path):
    image_u8 = (np.clip(image_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(image_u8).save(path, quality=95)


def _clip_embedding_unpaired(image_np):
    image_u8 = (np.clip(image_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    pixel_values = clip_proc(
        images=Image.fromarray(image_u8),
        return_tensors="pt",
    ).pixel_values.to(DEVICE, DTYPE)
    with torch.no_grad():
        embedding = image_encoder(pixel_values=pixel_values).image_embeds.float()
    return F.normalize(embedding, dim=-1)


def _masked_on_white_unpaired(image_np, mask_np):
    mask_3 = np.clip(mask_np[..., None], 0.0, 1.0)
    return image_np * mask_3 + (1.0 - mask_3)


def _comparison_image(out):
    panels = [
        ("Person", out["person"]),
        ("Target cloth", out["cloth_proc"]),
        ("Mask", np.repeat(out["mask"][..., None], 3, axis=2)),
        ("Raw model", out["raw_image"]),
        ("Result", out["image"]),
    ]
    header_h = 30
    gap = 5
    canvas = Image.new(
        "RGB",
        (len(panels) * W + (len(panels) - 1) * gap, H + header_h),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, image_np) in enumerate(panels):
        image_u8 = (np.clip(image_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        x = index * (W + gap)
        canvas.paste(Image.fromarray(image_u8), (x, header_h))
        draw.text((x + 5, 8), title, fill="black")
    return canvas


print(f"Unpaired holdout: {len(holdout_df)} samples")
print(f"Da co ket qua: {len(finished_keys)}")
print(f"Checkpoint: {CHECKPOINT_NAME}")

for row in tqdm(holdout_df.itertuples(index=False), total=len(holdout_df), desc="Unpaired evaluation"):
    person_id = str(row.person_id)
    cloth_id = str(row.cloth_id)
    if (person_id, cloth_id) in finished_keys:
        continue

    started = time.time()
    stem = f"{int(row.sample_index):04d}_{person_id}__{cloth_id}"
    try:
        out = run_inference(
            person_id=person_id,
            cloth_id=cloth_id,
            split="test",
            seed=int(row.seed),
            return_debug=True,
            show_progress=False,
        )

        target_embedding = _clip_embedding_unpaired(out["cloth_proc"])
        result_cloth = _masked_on_white_unpaired(out["image"], out["mask"])
        result_embedding = _clip_embedding_unpaired(result_cloth)
        clip_similarity = float(
            F.cosine_similarity(target_embedding, result_embedding).item()
        )

        outside = 1.0 - out["mask"]
        outside_denom = max(float(outside.sum()) * 3.0, 1.0)
        outside_mae = float(
            (np.abs(out["image"] - out["person"]) * outside[..., None]).sum()
            / outside_denom
        )
        raw_outside_mae = float(
            (np.abs(out["raw_image"] - out["person"]) * outside[..., None]).sum()
            / outside_denom
        )

        result_path = UNPAIRED_IMAGE_DIR / f"{stem}_result.jpg"
        raw_path = UNPAIRED_IMAGE_DIR / f"{stem}_raw.jpg"
        comparison_path = UNPAIRED_IMAGE_DIR / f"{stem}_comparison.jpg"
        _save_np_image(out["image"], result_path)
        _save_np_image(out["raw_image"], raw_path)
        _comparison_image(out).save(comparison_path, quality=92)

        record = {
            "checkpoint": CHECKPOINT_NAME,
            "sample_index": int(row.sample_index),
            "person_id": person_id,
            "cloth_id": cloth_id,
            "seed": int(row.seed),
            "clip_garment_similarity": clip_similarity,
            "outside_mae": outside_mae,
            "raw_outside_mae": raw_outside_mae,
            "mask_area": float(out["mask"].mean()),
            "seconds": float(time.time() - started),
            "caption": out["caption"],
            "result_path": str(result_path),
            "raw_path": str(raw_path),
            "comparison_path": str(comparison_path),
            "status": "ok",
            "error": "",
        }
    except Exception as exc:
        record = {
            "checkpoint": CHECKPOINT_NAME,
            "sample_index": int(row.sample_index),
            "person_id": person_id,
            "cloth_id": cloth_id,
            "seed": int(row.seed),
            "seconds": float(time.time() - started),
            "status": "error",
            "error": str(exc),
        }

    records.append(record)
    pd.DataFrame(records).to_csv(metrics_path, index=False)

metrics_df = pd.DataFrame(records)
ok_df = metrics_df[metrics_df["status"] == "ok"].copy()
metric_columns = [
    "clip_garment_similarity",
    "outside_mae",
    "raw_outside_mae",
    "mask_area",
    "seconds",
]
summary = {
    "checkpoint": CHECKPOINT_NAME,
    "checkpoint_source": str(CHECKPOINT_SOURCE),
    "evaluation_source": str(holdout_csv),
    "evaluation_mode": "unpaired holdout: original person_id and cloth_id",
    "samples_requested": int(len(holdout_df)),
    "samples_ok": int(len(ok_df)),
    "samples_error": int(len(metrics_df) - len(ok_df)),
    "metrics_mean": {
        column: float(ok_df[column].mean())
        for column in metric_columns
        if column in ok_df and len(ok_df)
    },
    "metrics_std": {
        column: float(ok_df[column].std())
        for column in metric_columns
        if column in ok_df and len(ok_df) > 1
    },
}
summary_path = UNPAIRED_DIR / "unpaired_summary.json"
with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

print(json.dumps(summary, indent=2))
print(f"Metrics: {metrics_path}")
print(f"Summary: {summary_path}")
print(f"Images: {UNPAIRED_IMAGE_DIR}")
