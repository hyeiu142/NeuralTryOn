> Note: this is a reported-results overview, not a strict ranking. The completed models currently use different paired manifests and LPIPS backbones.

| Model | Paired samples | SSIM | PSNR (dB) | LPIPS | LPIPS backbone | Holdout samples |
| --- | --- | --- | --- | --- | --- | --- |
| Lightweight U-Net + GMM + TOM | 2032 | 0.8932 | 21.39 | 0.1455 | VGG | 665 |
| GMM + Shape Generation + Pix2Pix | 2032 | 0.8951 | 21.49 | 0.1145 | VGG | 13 |
| Stable Diffusion Inpainting + LoRA | 996 | 0.8733 | 21.32 | 0.1055 | AlexNet | 665 |
