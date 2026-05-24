# File: app/app.py
import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
from diffusers import DDIMScheduler
import os

from src.models.builder import build_models
from src.utils.common import seed_everything

# 1. Khởi tạo môi trường & Model (Chạy 1 lần khi bật web)
print("Đang khởi tạo Web App...")
cfg = OmegaConf.load("configs/train.yaml")
seed_everything(cfg.training.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

vae, text_encoder, unet, _, image_encoder, image_proj = build_models(cfg)
scheduler = DDIMScheduler.from_pretrained(cfg.model.model_id, subfolder="scheduler")

CKPT_DIR = os.path.join(cfg.training.output_dir, "ckpt_epoch_03")
if os.path.exists(CKPT_DIR):
    unet.load_adapter(os.path.join(CKPT_DIR, "unet_lora"), adapter_name="default")
    image_proj.load_state_dict(torch.load(os.path.join(CKPT_DIR, "image_proj.pt"), map_location=device))
    print("Đã load Model Weights thành công!")
else:
    print("Chạy ở chế độ DEMO (Chưa có Weights).")

unet.eval()
image_proj.eval()

# Khởi tạo mô hình tự động nhận diện áo (Auto-Segmentation)
print("Đang khởi tạo Auto-Segmentation (Bóc tách áo tự động)...")
from transformers import pipeline
segmenter = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes", device=device)

# 2. Hàm Xử lý ảnh (Inference)
def try_on_clothes(person_img, cloth_img):
    if person_img is None or cloth_img is None:
        return None
    
    import numpy as np
    from PIL import Image
    import cv2

    bg_img = person_img.convert("RGB")
    
    # 1. Chạy AI nhận diện áo cũ
    seg_outputs = segmenter(bg_img)
    mask_np = np.zeros((bg_img.size[1], bg_img.size[0]), dtype=np.uint8)
    
    # Lọc ra vùng áo (Upper-clothes hoặc Dress)
    for out in seg_outputs:
        if out["label"] in ["Upper-clothes", "Dress"]:
            layer_np = np.array(out["mask"].convert("L"))
            mask_np = np.maximum(mask_np, layer_np)
            
    # 2. Làm phồng mask một xíu (dilate) để che kín hẳn viền áo cũ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
    
    # Làm mờ viền mask cho tự nhiên
    mask_np = cv2.GaussianBlur(mask_np, (7, 7), 0)
    mask_img = Image.fromarray(mask_np, mode="L")

    # 3. Tạo ảnh agnostic bằng cách bôi xám vùng áo cũ
    bg_np = np.array(bg_img)
    mask_bool = mask_np > 128
    agnostic_np = bg_np.copy()
    agnostic_np[mask_bool] = 128  # 128 = màu xám
    agnostic_img = Image.fromarray(agnostic_np)

    # Helper: Crop center để ảnh không bị méo (giống notebook)
    def sync_center_crop_and_resize(img, is_mask=False):
        target_h, target_w = cfg.dataset.target_size
        target_ratio = target_w / target_h
        w, h = img.size
        current_ratio = w / h

        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            img = img.crop((left, 0, left + new_w, h))
        else:
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            img = img.crop((0, top, w, top + new_h))

        interp = Image.NEAREST if is_mask else Image.BILINEAR
        return img.resize((target_w, target_h), interp)

    # Áp dụng crop cho toàn bộ ảnh để giữ đúng tỷ lệ body/áo
    agnostic_img = sync_center_crop_and_resize(agnostic_img, is_mask=False)
    mask_img = sync_center_crop_and_resize(mask_img, is_mask=True)
    cloth_img = sync_center_crop_and_resize(cloth_img, is_mask=False)

    # Preprocess
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    clip_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Convert PIL to Tensor
    agnostic = transform(agnostic_img).unsqueeze(0).to(device, dtype=dtype)
    cloth = transform(cloth_img.convert("RGB")).unsqueeze(0).to(device, dtype=dtype)
    clip_cloth = clip_transform(cloth_img.convert("RGB")).unsqueeze(0).to(device, dtype=dtype)
    
    mask = mask_transform(mask_img)
    mask = (mask > 0.5).float().unsqueeze(0).to(device, dtype=dtype)

    # Lấy text ngẫu nhiên hoặc rỗng (do dùng IP-Adapter là chính)
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(cfg.model.model_id, subfolder="tokenizer")
    input_ids = tokenizer(
        cfg.dataset.trigger_word + ", a clothing item",
        max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids.to(device)

    # Chạy AI
    with torch.no_grad():
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            text_embeds = text_encoder(input_ids)[0]
            cloth_embeds = image_encoder(pixel_values=clip_cloth).image_embeds
            cloth_tokens = image_proj(cloth_embeds.float()).to(dtype)
            encoder_hidden_states = torch.cat([text_embeds, cloth_tokens], dim=1)

            masked_latents = vae.encode(agnostic).latent_dist.sample() * cfg.model.vae_scale
            mask_down = F.interpolate(mask, size=masked_latents.shape[-2:], mode="nearest")

            latents = torch.randn_like(masked_latents)
            scheduler.set_timesteps(30)

            for t in scheduler.timesteps:
                latent_model_input = torch.cat([latents, mask_down, masked_latents], dim=1)
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            latents = latents / cfg.model.vae_scale
            generated_image = vae.decode(latents).sample
            generated_image = torch.clamp(generated_image, -1.0, 1.0)
            
            # Convert back to PIL
            gen_img = (generated_image + 1.0) / 2.0
            gen_img = gen_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            gen_img = (gen_img * 255).astype("uint8")
            return Image.fromarray(gen_img)

# 3. Giao diện Web (UI)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👕 NeuralTryOn - Trải Nghiệm Thử Đồ Bằng AI (Model 4)")
    gr.Markdown("Thử nghiệm sức mạnh của Stable Diffusion Inpainting + LoRA + IP-Adapter")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Đầu vào (Inputs)")
            gr.Markdown("Upload ảnh của bạn. AI sẽ tự động phân tích và tìm vùng áo cũ để thay thế.")
            in_person = gr.Image(type="pil", label="1. Ảnh người mẫu (AI sẽ tự bóc tách áo)")
            
            gr.Markdown("Upload áo mới.")
            in_cloth = gr.Image(type="pil", label="2. Ảnh Áo mới")
            btn_run = gr.Button("✨ Mặc Thử Áo ✨", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### Kết quả (Output)")
            out_img = gr.Image(type="pil", label="Ảnh Kết Quả Tự Sinh")

    btn_run.click(fn=try_on_clothes, inputs=[in_person, in_cloth], outputs=out_img)

if __name__ == "__main__":
    demo.launch(share=True)
