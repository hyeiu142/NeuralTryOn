# ================================================================
# CELL 14: PREPROCESS + UNPAIRED HOLDOUT INFERENCE FUNCTION
# ================================================================

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor


TEST_DIR = VITON_ROOT / "test"
clip_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")


def _normalize(t):
    return t * 2.0 - 1.0


def _crop_resize(img, is_mask=False):
    w, h = img.size
    target_ratio = W / H
    if w / h > target_ratio:
        new_w = int(h * target_ratio)
        img = img.crop(((w - new_w) // 2, 0, (w - new_w) // 2 + new_w, h))
    else:
        new_h = int(w / target_ratio)
        img = img.crop((0, (h - new_h) // 2, w, (h - new_h) // 2 + new_h))
    interp = Image.NEAREST if is_mask else Image.BILINEAR
    return img.resize((W, H), interp)


def _process_cloth(cloth_img, cloth_mask_pil):
    bin_mask = (np.array(cloth_mask_pil) > 128).astype(np.uint8) * 255
    mask_pil = Image.fromarray(bin_mask).convert("L")

    white_bg = Image.new("RGB", cloth_img.size, (255, 255, 255))
    cloth_cut = Image.composite(cloth_img, white_bg, mask_pil)

    bbox = mask_pil.getbbox()
    if bbox:
        cloth_cut = cloth_cut.crop(bbox)
        mask_pil = mask_pil.crop(bbox)

    w, h = cloth_cut.size
    pw, ph = max(1, int(w * 0.1)), max(1, int(h * 0.1))
    canvas = Image.new("RGB", (w + 2 * pw, h + 2 * ph), (255, 255, 255))
    canvas.paste(cloth_cut, (pw, ph))

    cw, ch = canvas.size
    ratio = W / H
    if cw / ch > ratio:
        fh = int(cw / ratio)
        c2 = Image.new("RGB", (cw, fh), (255, 255, 255))
        c2.paste(canvas, (0, (fh - ch) // 2))
    else:
        fw = int(ch * ratio)
        c2 = Image.new("RGB", (fw, ch), (255, 255, 255))
        c2.paste(canvas, ((fw - cw) // 2, 0))

    return c2.resize((W, H), Image.BILINEAR)


def _find_file(split_dir, folder, stem, exts):
    for ext in exts:
        path = split_dir / folder / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def _load_rgb_tensor(path):
    img = Image.open(path).convert("RGB")
    img = _crop_resize(img)
    return _normalize(TF.to_tensor(img)).unsqueeze(0).to(DEVICE, DTYPE)


def _load_pose_tensor(person_id, split_dir):
    pose_path = _find_file(split_dir, "image-densepose", person_id, [".jpg", ".png"])
    if pose_path is None:
        pose_path = _find_file(split_dir, "openpose_img", f"{person_id}_rendered", [".png", ".jpg"])
    if pose_path is None:
        img = Image.fromarray(np.full((H, W, 3), 128, dtype=np.uint8))
        return _normalize(TF.to_tensor(img)).unsqueeze(0).to(DEVICE, DTYPE)
    return _load_rgb_tensor(pose_path)


def _parse_diff_mask(person_id, split_dir):
    parse_path = _find_file(split_dir, "image-parse-v3", person_id, [".png"])
    agn_parse_path = _find_file(split_dir, "image-parse-agnostic-v3.2", person_id, [".png"])

    if parse_path and agn_parse_path:
        parse_np = np.array(Image.open(parse_path))
        agn_np = np.array(Image.open(agn_parse_path))
        if parse_np.shape[:2] != (H, W):
            parse_np = np.array(Image.fromarray(parse_np).resize((W, H), Image.NEAREST))
            agn_np = np.array(Image.fromarray(agn_np).resize((W, H), Image.NEAREST))
        mask_np = (parse_np != agn_np).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        return Image.fromarray(mask_np, mode="L")

    fallback = np.zeros((H, W), dtype=np.uint8)
    fallback[:int(H * 0.6), :] = 255
    return Image.fromarray(fallback, mode="L")


def _cloth_color_words(cloth_img, cloth_mask_pil):
    img = np.array(cloth_img.convert("RGB")).astype(np.float32)
    mask = np.array(cloth_mask_pil.convert("L")) > 128
    if mask.sum() < 16:
        return []

    pixels = img[mask] / 255.0
    mean = pixels.mean(axis=0)
    r, g, b = mean.tolist()
    brightness = float(mean.mean())
    saturation = float(mean.max() - mean.min())

    if brightness > 0.82 and saturation < 0.16:
        return ["plain white"]
    if brightness < 0.22 and saturation < 0.18:
        return ["black"]
    if saturation < 0.10:
        if brightness > 0.62:
            return ["light gray"]
        if brightness < 0.42:
            return ["dark gray"]
        return ["gray"]

    colors = []
    if r > 0.55 and g < 0.45 and b < 0.45:
        colors.append("red")
    elif r > 0.65 and g > 0.42 and b < 0.45:
        colors.append("orange")
    elif r > 0.55 and b > 0.55 and g < 0.52:
        colors.append("pink")
    elif b > 0.55 and r < 0.50:
        colors.append("blue")
    elif g > 0.48 and r < 0.55 and b < 0.55:
        colors.append("green")
    elif r > 0.45 and g > 0.35 and b < 0.28:
        colors.append("brown")

    return colors[:1]


def _cloth_shape_words(cloth_img, cloth_mask_pil):
    mask = np.array(cloth_mask_pil.convert("L")) > 128
    if mask.sum() < 16:
        return []

    ys, xs = np.where(mask)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    area_ratio = float(mask.mean())
    aspect = w / max(h, 1)
    words = []

    if aspect > 0.75 and area_ratio < 0.34:
        words.extend(["short sleeve", "crew neck", "t-shirt"])
    elif area_ratio >= 0.34:
        words.append("long sleeve top")
    else:
        words.append("top")

    return words


def _enhance_caption(raw_caption, cloth_img, cloth_mask_pil):
    cap = raw_caption.strip()
    lower = cap.lower()
    additions = []

    if not any(c in lower for c in [
        "white", "black", "gray", "grey", "red", "pink", "blue",
        "green", "brown", "orange", "yellow", "purple", "beige"
    ]):
        additions.extend(_cloth_color_words(cloth_img, cloth_mask_pil))

    if not any(w in lower for w in ["short sleeve", "long sleeve", "sleeveless", "tank", "tee", "t-shirt", "shirt", "blouse", "top"]):
        additions.extend(_cloth_shape_words(cloth_img, cloth_mask_pil))
    elif ("tee" in lower or "t-shirt" in lower) and "short sleeve" not in lower and "long sleeve" not in lower:
        additions.append("short sleeve")

    if not any(w in lower for w in ["plain", "solid", "print", "printed", "stripe", "striped", "logo", "graphic"]):
        additions.append("plain")

    deduped = []
    seen = set()
    for item in additions:
        item = item.strip()
        if item and item not in seen and item not in lower:
            deduped.append(item)
            seen.add(item)

    if deduped:
        cap = f"{cap}, " + ", ".join(deduped)
    return cap


def _caption_for(cloth_id, cloth_img=None, cloth_mask_pil=None):
    roots = []
    if CAPTION_ROOT:
        cap_root = Path(CAPTION_ROOT) / "cloth-captions"
        roots.extend([cap_root / "test", cap_root / "train"])
    raw = "a photo of a garment"
    for root in roots:
        txt = root / f"{cloth_id}.txt"
        if txt.exists():
            raw = txt.read_text(encoding="utf-8").strip()
            break
    if cloth_img is not None and cloth_mask_pil is not None:
        raw = _enhance_caption(raw, cloth_img, cloth_mask_pil)
    return raw if TRIGGER in raw else f"{TRIGGER}, {raw}"


def _encode_latent(img_tensor, use_mode=False):
    dist = vae.encode(img_tensor).latent_dist
    lat = dist.mode() if use_mode else dist.sample()
    return lat * 0.18215


def _decode_latent(lat):
    img = vae.decode((lat / 0.18215).to(DTYPE)).sample
    return ((img + 1.0) / 2.0).clamp(0, 1)


def _tensor_to_np01(img_tensor):
    img = img_tensor.detach().squeeze(0).float().cpu()
    img = ((img + 1.0) / 2.0).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def _np_from_01_tensor(img_tensor):
    img = img_tensor.detach().squeeze(0).float().cpu()
    return img.permute(1, 2, 0).numpy()


def _encode_text(prompt):
    ids = tokenizer(
        [prompt],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(DEVICE)
    return text_encoder(ids)[0]


@torch.no_grad()
def run_inference(
    person_id,
    cloth_id,
    split="test",
    seed=42,
    return_debug=False,
    zero_cloth_latent=False,
    zero_cloth_tokens=False,
    show_progress=True,
):
    """
    Unpaired VTO inference on holdout row:
      person_id: người nhận áo
      cloth_id : áo target
    """
    split_dir = VITON_ROOT / split

    person = _load_rgb_tensor(split_dir / "image" / f"{person_id}.jpg")
    agnostic = _load_rgb_tensor(split_dir / "agnostic-v3.2" / f"{person_id}.jpg")
    pose = _load_pose_tensor(person_id, split_dir)

    cloth_pil = Image.open(split_dir / "cloth" / f"{cloth_id}.jpg").convert("RGB")
    cloth_mask_pil = Image.open(split_dir / "cloth-mask" / f"{cloth_id}.jpg").convert("L")
    cloth_proc = _process_cloth(cloth_pil, cloth_mask_pil)
    cloth = _normalize(TF.to_tensor(cloth_proc)).unsqueeze(0).to(DEVICE, DTYPE)
    cloth_clip = clip_proc(images=cloth_proc, return_tensors="pt").pixel_values.to(DEVICE, DTYPE)

    lat_agn = _encode_latent(agnostic, use_mode=True)
    lat_pose = _encode_latent(pose, use_mode=True)
    lat_cloth = _encode_latent(cloth, use_mode=True)
    lat_person = _encode_latent(person, use_mode=True)

    mask_pil = _parse_diff_mask(person_id, split_dir)
    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
    mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(DEVICE, DTYPE)
    lat_mask = F.interpolate(mask, size=lat_agn.shape[-2:], mode="nearest")
    lat_mask_4 = lat_mask.expand_as(lat_agn)

    caption = _caption_for(cloth_id, cloth_pil, cloth_mask_pil)
    text_cond = _encode_text(caption)
    text_uncond = _encode_text("")

    clip_patches = image_encoder(pixel_values=cloth_clip, output_hidden_states=True).last_hidden_state
    cloth_global = perceiver(clip_patches.float()).to(DTYPE)
    cloth_spatial = cloth_spatial_proj(lat_cloth.float()).to(DTYPE)

    if zero_cloth_tokens:
        cloth_global = torch.zeros_like(cloth_global)
        cloth_spatial = torch.zeros_like(cloth_spatial)

    lat_cloth_input = torch.zeros_like(lat_cloth) if zero_cloth_latent else lat_cloth

    enc_cond = torch.cat([text_cond, cloth_global, cloth_spatial], dim=1)
    enc_uncond = torch.cat(
        [text_uncond, torch.zeros_like(cloth_global), torch.zeros_like(cloth_spatial)],
        dim=1,
    )

    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    init_noise = torch.randn(
        lat_agn.shape,
        generator=generator,
        device=DEVICE,
        dtype=DTYPE,
    )
    latents = init_noise * scheduler.init_noise_sigma

    # Start outside-mask from noisy original person latent to preserve identity.
    t_start = scheduler.timesteps[0].reshape(1).to(DEVICE)
    orig_noisy = scheduler.add_noise(lat_person, init_noise, t_start)
    latents = lat_mask_4 * latents + (1.0 - lat_mask_4) * orig_noisy

    for i, t in enumerate(tqdm(
        scheduler.timesteps,
        desc=f"Denoising {person_id}->{cloth_id}",
        leave=False,
        disable=not show_progress,
    )):
        model_input = torch.cat([latents, lat_mask, lat_agn, lat_pose, lat_cloth_input], dim=1)
        model_input_2x = torch.cat([model_input, model_input], dim=0)
        enc_2x = torch.cat([enc_uncond, enc_cond], dim=0)

        noise = unet(model_input_2x, t, encoder_hidden_states=enc_2x).sample
        noise_uncond, noise_cond = noise.chunk(2)
        noise_pred = noise_uncond + CFG_SCALE * (noise_cond - noise_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if i < len(scheduler.timesteps) - 1:
            t_prev = scheduler.timesteps[i + 1].reshape(1).to(DEVICE)
            orig_noisy = scheduler.add_noise(lat_person, init_noise, t_prev)
            latents = lat_mask_4 * latents + (1.0 - lat_mask_4) * orig_noisy

    latents = lat_mask_4 * latents + (1.0 - lat_mask_4) * lat_person
    raw = _decode_latent(latents)
    raw_np = _np_from_01_tensor(raw)

    # Pixel composite outside mask to reduce face/background distortion.
    orig_np = _tensor_to_np01(person)
    comp_mask = cv2.GaussianBlur(mask_np.astype(np.float32), (13, 13), 0)
    comp_mask = np.clip(comp_mask[..., None], 0.0, 1.0)
    result_np = comp_mask * raw_np + (1.0 - comp_mask) * orig_np

    if return_debug:
        return {
            "image": result_np,
            "raw_image": raw_np,
            "mask": mask_np,
            "caption": caption,
            "person_id": person_id,
            "cloth_id": cloth_id,
            "cloth_proc": np.array(cloth_proc).astype(np.float32) / 255.0,
            "person": orig_np,
            "agnostic": _tensor_to_np01(agnostic),
            "zero_cloth_latent": zero_cloth_latent,
            "zero_cloth_tokens": zero_cloth_tokens,
        }
    return result_np


print("Inference function ready.")
print("Call: out = run_inference(person_id, cloth_id, split='test')")
