"""Reusable preprocessing and inference functions for Model 1 evaluation."""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


_normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def _with_jpg(value):
    value = str(value)
    return value if value.endswith(".jpg") else f"{value}.jpg"


def _tensor_rgb(image):
    return _normalize(TF.to_tensor(processor.resize_rgb(image)))


def _mask_tensor(mask):
    mask_u8 = (np.asarray(mask, dtype=np.float32) * 255.0).astype(np.uint8)
    return TF.to_tensor(processor.resize_mask(Image.fromarray(mask_u8)))


def load_model_1_inputs(person_id, cloth_id):
    """Load one person-cloth pair using the same preprocessing as the notebook."""
    person_file = _with_jpg(person_id)
    cloth_file = _with_jpg(cloth_id)
    person_stem = Path(person_file).stem

    person_pil = Image.open(TEST_DIR / "image" / person_file).convert("RGB")
    raw_agnostic_pil = Image.open(TEST_DIR / "agnostic-v3.2" / person_file).convert("RGB")
    cloth_pil = Image.open(TEST_DIR / "cloth" / cloth_file).convert("RGB")
    cloth_mask_pil = Image.open(TEST_DIR / "cloth-mask" / cloth_file).convert("L")
    parse_pil = Image.open(TEST_DIR / "image-parse-v3" / f"{person_stem}.png")
    pose_path = TEST_DIR / "openpose_json" / f"{person_stem}_keypoints.json"
    densepose_path = TEST_DIR / "image-densepose" / person_file
    densepose_pil = (
        Image.open(densepose_path).convert("RGB")
        if densepose_path.exists()
        else Image.new("RGB", (768, 1024), (0, 0, 0))
    )

    parse_np = np.asarray(parse_pil)
    hair_mask = np.isin(parse_np, [1, 2]).astype(np.float32)
    arm_mask = np.isin(parse_np, [14, 15]).astype(np.float32)
    hair_mask_3d = hair_mask[..., None]
    agnostic_fixed = Image.fromarray(
        (
            np.asarray(person_pil) * hair_mask_3d
            + np.asarray(raw_agnostic_pil) * (1.0 - hair_mask_3d)
        ).astype(np.uint8)
    )

    return {
        "person": _tensor_rgb(person_pil).unsqueeze(0).to(DEVICE),
        "agnostic": _tensor_rgb(agnostic_fixed).unsqueeze(0).to(DEVICE),
        "raw_agnostic": _tensor_rgb(raw_agnostic_pil).unsqueeze(0).to(DEVICE),
        "cloth": _tensor_rgb(cloth_pil).unsqueeze(0).to(DEVICE),
        "cloth_mask": TF.to_tensor(processor.resize_mask(cloth_mask_pil)).unsqueeze(0).to(DEVICE),
        "densepose": _tensor_rgb(densepose_pil).unsqueeze(0).to(DEVICE),
        "pose": ((processor.generate_pose_heatmap(pose_path) * 2.0) - 1.0).unsqueeze(0).to(DEVICE),
        "hair_mask": _mask_tensor(hair_mask).unsqueeze(0).to(DEVICE),
        "arm_mask": _mask_tensor(arm_mask).unsqueeze(0).to(DEVICE),
        "person_pil": person_pil,
        "cloth_pil": cloth_pil,
    }


def _to_numpy(tensor):
    return (
        ((tensor.detach().float().cpu().squeeze(0) + 1.0) / 2.0)
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )


def run_inference(person_id, cloth_id, mode="paired", return_debug=False):
    """Run U-Net -> GMM -> TOM and return an RGB NumPy image in [0, 1]."""
    batch = load_model_1_inputs(person_id, cloth_id)
    autocast = torch.amp.autocast("cuda") if DEVICE.type == "cuda" else nullcontext()

    with torch.no_grad(), autocast:
        if mode == "unpaired":
            unet_agnostic = (
                batch["person"] * batch["hair_mask"]
                + batch["raw_agnostic"] * (1.0 - batch["hair_mask"])
            )
            gmm_agnostic = batch["raw_agnostic"]
        else:
            unet_agnostic = batch["agnostic"]
            gmm_agnostic = batch["agnostic"]

        unet_input = torch.cat(
            [unet_agnostic, batch["pose"], batch["cloth"], batch["densepose"]],
            dim=1,
        )
        predicted_mask = (torch.sigmoid(model_unet(unet_input)) > 0.5).float()

        theta = model_gmm(batch["cloth"], gmm_agnostic, batch["pose"])
        grid = tps_generator(theta)
        warped_cloth = F.grid_sample(
            batch["cloth"], grid, padding_mode="border", align_corners=True
        )
        warped_mask = (
            F.grid_sample(
                batch["cloth_mask"], grid, padding_mode="zeros", align_corners=True
            )
            > 0.5
        ).float()

        if mode == "unpaired":
            visible_arm = torch.clamp(batch["arm_mask"] - predicted_mask, 0.0, 1.0)
            preserve = torch.clamp(batch["hair_mask"] + visible_arm, 0.0, 1.0)
            tom_agnostic = (
                batch["person"] * preserve
                + batch["raw_agnostic"] * (1.0 - preserve)
            )
            tom_mask = predicted_mask * (1.0 - visible_arm)
        else:
            tom_agnostic = batch["agnostic"]
            tom_mask = predicted_mask

        cloth_network = warped_cloth * warped_mask + (-1.0) * (1.0 - warped_mask)
        cloth_clean = warped_cloth * warped_mask + (1.0 - warped_mask)
        tom_input = torch.cat(
            [tom_agnostic, batch["pose"], cloth_network, tom_mask],
            dim=1,
        )
        rendered, composition_mask = model_tom(tom_input)
        result = composition_mask * cloth_clean + (1.0 - composition_mask) * rendered

    output = {
        "image": _to_numpy(result),
        "person": _to_numpy(batch["person"]),
        "cloth": _to_numpy(batch["cloth"]),
        "predicted_mask": predicted_mask.squeeze().detach().float().cpu().numpy(),
        "warped_mask": warped_mask.squeeze().detach().float().cpu().numpy(),
    }
    return output if return_debug else output["image"]


print("run_inference(person_id, cloth_id, mode='paired'|'unpaired') is ready.")
