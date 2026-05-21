import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
import os

from src.utils.common import seed_everything
from src.data.dataset import StableDiffusionInpaintPreprocess
from src.models.builder import build_models

def main():
    # configs
    cfg = OmegaConf.load("configs/train.yaml")
    
    # setup
    seed_everything(cfg.training.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    os.makedirs(cfg.training.output_dir, exist_ok=True)

    # load models
    print("\n--- Khởi tạo Models ---")
    vae, text_encoder, unet, noise_scheduler, image_encoder, image_proj = build_models(cfg)
    
    # LoRA UNet + ImageProj
    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters())) + \
                       list(image_proj.parameters())
    
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.training.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # load data
    print("\n--- Khởi tạo Dữ liệu ---")
    train_dataset = StableDiffusionInpaintPreprocess(
        csv_path=cfg.dataset.train_csv,
        root_dir=cfg.dataset.train_root,
        caption_dir=cfg.dataset.caption_train_dir,
        tokenizer_path=cfg.model.model_id,
        target_size=cfg.dataset.target_size,
        trigger_word=cfg.dataset.trigger_word
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=4, 
        drop_last=True
    )

    # training loop
    print("\n--- Start Training ---")
    global_step = 0
    VAE_SCALE = cfg.model.vae_scale
    
    for epoch in range(cfg.training.num_epochs):
        print(f"\n===== Epoch {epoch + 1}/{cfg.training.num_epochs} =====")
        unet.train()
        image_proj.train()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad(set_to_none=True)
        
        for step, batch in pbar:
            # move data to device
            pixel_values = batch["image"].to(device, dtype=dtype)
            agnostic = batch["agnostic"].to(device, dtype=dtype)
            inpaint_mask = batch["inpaint_mask"].to(device, dtype=dtype)
            clip_cloth = batch["clip_cloth"].to(device, dtype=dtype)
            input_ids = batch["input_ids"].to(device)

            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                # Conditioning
                with torch.no_grad():
                    text_embeds = text_encoder(input_ids)[0]
                    cloth_embeds = image_encoder(pixel_values=clip_cloth).image_embeds
                
                cloth_tokens = image_proj(cloth_embeds.float()).to(dtype)
                encoder_hidden_states = torch.cat([text_embeds, cloth_tokens], dim=1)

                # Encode images
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * VAE_SCALE
                    masked_latents = vae.encode(agnostic).latent_dist.sample() * VAE_SCALE

                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 9-channels input for UNet Inpainting
                mask_down = F.interpolate(inpaint_mask, size=latents.shape[-2:], mode="nearest")
                latent_model_input = torch.cat([noisy_latents, mask_down, masked_latents], dim=1)

                # predict noise
                noise_pred = unet(
                    latent_model_input, timesteps, 
                    encoder_hidden_states=encoder_hidden_states
                ).sample

                # Loss (Mask-weighted MSE)
                loss_map = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                mask_weight = 1.0 + cfg.training.mask_loss_weight * mask_down.float()
                loss = (loss_map * mask_weight).mean() / cfg.training.grad_accum

            # Update parameters
            scaler.scale(loss).backward()

            if (step + 1) % cfg.training.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            pbar.set_description(f"Epoch {epoch+1} | Step {step+1} | Loss: {loss.item() * cfg.training.grad_accum:.4f}")

        # save checkpoint after each epoch 
        ckpt_path = os.path.join(cfg.training.output_dir, f"epoch_{epoch+1}")
        os.makedirs(ckpt_path, exist_ok=True)
        unet.save_pretrained(os.path.join(ckpt_path, "unet_lora"))
        torch.save(image_proj.state_dict(), os.path.join(ckpt_path, "image_proj.pt"))
        print(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
