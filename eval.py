import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
from diffusers import DDIMScheduler
from src.utils.common import seed_everything
from src.data.dataset import StableDiffusionInpaintPreprocess
from src.models.builder import build_models
from src.utils.metrics import VTONMetrics

def main():
    cfg = OmegaConf.load('configs/train.yaml')
    seed_everything(cfg.training.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32

    CKPT_DIR = os.path.join(cfg.training.output_dir, "ckpt_epoch_03")

    print('Loading model and checkpoint')
    vae, text_encoder, unet, _, image_encoder, image_proj = build_models(cfg)

    scheduler = DDIMScheduler.from_pretrained(cfg.model.model_id, subfolder='scheduler')

    if os.path.exists(CKPT_DIR):
        print(f"Loading weights from {CKPT_DIR}")
        unet.load_attn_projs(os.path.join(CKPT_DIR, 'unet_lora'))
        image_proj.load_state_dict(torch.load(os.path.join(KDPT_DIR, 'image_proj.pt'), map_location=device))

    else: 
        print('No checkpoint found, using pre-trained weights')
        
    unet.eval()
    image_proj.eval()

    metrics_calc = VTONMetrics(device=device)

    print("\nLoading data from test set")
    val_dataset = StableDiffusionInpaintPreprocess(
        csv_path=cfg.dataset.val_csv,     
        root_dir=cfg.dataset.val_root,
        caption_dir=cfg.dataset.caption_val_dir,
        tokenizer_path=cfg.model.model_id,
        target_size=cfg.dataset.target_size,
        trigger_word=cfg.dataset.trigger_word,
        test_flag=True
    )

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    print("\nStarting inference on test set...")

    total_ssim = 0.0
    total_psnr = 0.0
    total_lpip = 0.0
    num_samples = 0

    VAE_SCALE = cfg.model.vae_scale
    num_inference_steps = 30

    for step, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        pixel_values = batch['image'].to(device, dtype=dtype)
        agnostic = batch["agnostic"].to(device, dtype=dtype)
        inpaint_mask = batch["inpaint_mask"].to(device, dtype=dtype)
        clip_cloth = batch["clip_cloth"].to(device, dtype=dtype)
        input_ids = batch["input_ids"].to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                # embeddings
                text_embeds = text_encoder(input_ids)[0]
                cloth_embeds = image_encoder(pixel_values=clip_cloth).image_embeds
                cloth_tokens = image_proj(cloth_embeds.float()).to(dtype)
                encoder_hidden_states = torch.cat([text_embeds, cloth_tokens], dim=1)
                # Encode agnostic
                masked_latents = vae.encode(agnostic).latent_dist.sample() * VAE_SCALE
                mask_down = F.interpolate(inpaint_mask, size=masked_latents.shape[-2:], mode="nearest")

                # Noise
                latents = torch.randn_like(masked_latents)
                scheduler.set_timesteps(num_inference_steps)
                
                for t in scheduler.timesteps:
                    latent_model_input = torch.cat([latents, mask_down, masked_latents], dim=1)
                    
                    noise_pred = unet(
                        latent_model_input, t, 
                        encoder_hidden_states=encoder_hidden_states
                    ).sample
                    
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
                # Decode result
                latents = latents / VAE_SCALE
                generated_image = vae.decode(latents).sample
                
                # Clamp pixel in range [-1, 1]
                generated_image = torch.clamp(generated_image, -1.0, 1.0)
                # Metrics
                gen_01 = (generated_image + 1.0) / 2.0
                target_01 = (pixel_values + 1.0) / 2.0
                
                metrics = metrics_calc.calculate(gen_01, target_01)
                
                total_ssim += metrics["SSIM"]
                total_psnr += metrics["PSNR"]
                total_lpips += metrics["LPIPS"]
                num_samples += 1
                if step == 9: 
                    break
    # Report    
    print("\n" + "="*40)
    print(" EVALUATION RESULTS (MODEL 4 - SD+LoRA)")
    print("="*40)
    print(f"Total test images : {num_samples}")
    print(f"Average SSIM  : {total_ssim / num_samples:.4f} (Higher is better)")
    print(f"Average PSNR  : {total_psnr / num_samples:.2f} dB (Higher is better)")
    print(f"Average LPIPS : {total_lpips / num_samples:.4f} (Lower is better)")
    print("="*40)
if __name__ == "__main__":
    main()


    


    

    

    