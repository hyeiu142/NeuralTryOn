> Note: the report defines one common 996-sample paired protocol. Model 1 and Model 2 values require confirmation on the common manifest, and LPIPS backbones differ.

| Model | Paired samples | SSIM | PSNR (dB) | LPIPS | LPIPS backbone | Holdout samples |
| --- | --- | --- | --- | --- | --- | --- |
| Lightweight U-Net + GMM + TOM | 996 | 0.8932 | 21.39 | 0.1455 | VGG | 665 |
| GMM + Shape Generation + Pix2Pix | 996 | 0.8951 | 21.49 | 0.1145 | VGG | 13 |
| Stable Diffusion Inpainting + LoRA | 996 | 0.8733 | 21.32 | 0.1055 | AlexNet | 665 |
