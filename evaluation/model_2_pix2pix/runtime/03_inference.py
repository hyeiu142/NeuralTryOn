"""Reusable Model 2 preprocessing and GMM -> Stage 1 -> Pix2Pix inference."""

from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


_norm = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def _heatmaps(json_path):
    heatmaps = torch.zeros((18, HEIGHT, WIDTH))
    if not json_path.exists():
        return heatmaps
    try:
        keypoints = np.asarray(json.loads(json_path.read_text())["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)
    except Exception:
        return heatmaps
    for index, (x_orig, y_orig, confidence) in enumerate(keypoints[:18]):
        if confidence <= 0.05:
            continue
        x, y, radius = int(x_orig * WIDTH / 768), int(y_orig * HEIGHT / 1024), 4
        left, right = max(0, x - 3 * radius), min(WIDTH, x + 3 * radius + 1)
        top, bottom = max(0, y - 3 * radius), min(HEIGHT, y + 3 * radius + 1)
        yy, xx = np.ogrid[top:bottom, left:right]
        gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * radius ** 2))
        heatmaps[index, top:bottom, left:right] = torch.maximum(
            heatmaps[index, top:bottom, left:right], torch.from_numpy(gaussian).float()
        )
    return heatmaps


def _rgb(path):
    return TF.to_tensor(processor.resize_rgb(Image.open(path).convert("RGB")))


def _mask(path):
    return TF.to_tensor(processor.resize_mask(Image.open(path)))


def load_model_2_inputs(person_id, cloth_id):
    person_id, cloth_id = str(person_id).replace(".jpg", ""), str(cloth_id).replace(".jpg", "")
    person = _rgb(TEST_DIR / "image" / f"{person_id}.jpg")
    agnostic_raw = _rgb(TEST_DIR / "agnostic-v3.2" / f"{person_id}.jpg")
    cloth = _rgb(TEST_DIR / "cloth" / f"{cloth_id}.jpg")
    densepose = _rgb(TEST_DIR / "image-densepose" / f"{person_id}.jpg")
    parse_v3 = _mask(TEST_DIR / "image-parse-v3" / f"{person_id}.png")
    cloth_mask = _mask(TEST_DIR / "cloth-mask" / f"{cloth_id}.jpg")

    parse_int = (parse_v3 * 255).round()
    intact = torch.zeros_like(parse_v3)
    for label in [0, 1, 2, 4, 6, 8, 9, 12, 13, 16, 17, 18, 19]:
        intact += (parse_int == label).float()
    wipe_dilated = F.max_pool2d((1.0 - intact.clamp(0, 1)).unsqueeze(0), 5, 1, 2).squeeze(0)
    intact_eroded = 1.0 - wipe_dilated
    agnostic = intact_eroded * person + (1.0 - intact_eroded) * agnostic_raw
    hair_mask = (parse_int == 2).float()
    hair_ref = person * hair_mask

    def batch(tensor, normalize=False):
        tensor = _norm(tensor) if normalize else tensor
        return tensor.unsqueeze(0).to(DEVICE)

    return {
        "person": batch(person, True),
        "agnostic": batch(agnostic, True),
        "cloth": batch(cloth, True),
        "densepose": batch(densepose, True),
        "parse_v3": batch(parse_v3),
        "cloth_mask": batch(cloth_mask),
        "hair_mask": batch(hair_mask),
        "hair_ref": batch(hair_ref, True),
        "pose": batch((_heatmaps(TEST_DIR / "openpose_json" / f"{person_id}_keypoints.json") * 2.0) - 1.0),
    }


def _np01(tensor):
    return ((tensor.squeeze(0).detach().float().cpu() + 1.0) / 2.0).clamp(0, 1).permute(1, 2, 0).numpy()


def run_inference(person_id, cloth_id, return_debug=False):
    batch = load_model_2_inputs(person_id, cloth_id)
    autocast = torch.amp.autocast("cuda") if DEVICE.type == "cuda" else nullcontext()
    with torch.no_grad(), autocast:
        warped_cloth, warped_mask, _, _ = model_gmm(
            batch["agnostic"], batch["densepose"], batch["pose"], batch["cloth"], batch["cloth_mask"]
        )
        warped_cloth = warped_cloth * warped_mask
        stage1_mask = model_stage1(
            batch["agnostic"], batch["pose"], batch["cloth_mask"], batch["densepose"], batch["hair_mask"]
        )
        generator_input = torch.cat(
            [batch["agnostic"], warped_cloth, warped_mask, batch["parse_v3"], batch["densepose"], stage1_mask],
            dim=1,
        )
        result, blend_mask, rendered, generation_zone = model_tom(
            generator_input, warped_cloth, batch["agnostic"], batch["parse_v3"], batch["densepose"],
            stage1_mask, batch["hair_ref"], batch["hair_mask"],
        )
    output = {
        "image": _np01(result),
        "person": _np01(batch["person"]),
        "cloth": _np01(batch["cloth"]),
        "warped_cloth": _np01(warped_cloth),
        "blend_mask": blend_mask.squeeze().detach().float().cpu().numpy(),
        "generation_zone": generation_zone.squeeze().detach().float().cpu().numpy(),
    }
    return output if return_debug else output["image"]


print("run_inference(person_id, cloth_id, return_debug=False) is ready.")
