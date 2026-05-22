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
    unet.load_attn_projs(os.path.join(CKPT_DIR, "unet_lora"))
    image_proj.load_state_dict(torch.load(os.path.join(CKPT_DIR, "image_proj.pt"), map_location=device))
    print("Đã load Model Weights thành công!")
else:
    print("Chạy ở chế độ DEMO (Chưa có Weights).")

unet.eval()
image_proj.eval()

# 2. Hàm Xử lý ảnh (Inference)
def try_on_clothes(agnostic_img, mask_img, cloth_img):
    if agnostic_img is None or mask_img is None or cloth_img is None:
        return None
    
    # Preprocess giống y hệt dataset.py
    transform = transforms.Compose([
        transforms.Resize(cfg.dataset.target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(cfg.dataset.target_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    clip_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Convert Pll to Tensor
    agnostic = transform(agnostic_img.convert("RGB")).unsqueeze(0).to(device, dtype=dtype)
    cloth = transform(cloth_img.convert("RGB")).unsqueeze(0).to(device, dtype=dtype)
    clip_cloth = clip_transform(cloth_img.convert("RGB")).unsqueeze(0).to(device, dtype=dtype)
    
    mask = mask_transform(mask_img.convert("L"))
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
            in_agnostic = gr.Image(type="pil", label="1. Ảnh người (Đã che vùng áo)")
            in_mask = gr.Image(type="pil", label="2. Ảnh Mask (Đen trắng)")
            in_cloth = gr.Image(type="pil", label="3. Ảnh Áo mới")
            btn_run = gr.Button("✨ Mặc Thử Áo ✨", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### Kết quả (Output)")
            out_img = gr.Image(type="pil", label="Ảnh Kết Quả Tự Sinh")

    btn_run.click(fn=try_on_clothes, inputs=[in_agnostic, in_mask, in_cloth], outputs=out_img)

if __name__ == "__main__":
    demo.launch(share=True)
