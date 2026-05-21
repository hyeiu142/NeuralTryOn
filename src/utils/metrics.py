import torch 
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class VTONMetrics: 
    def __init__(self, device=None):
        if device is None: 
            self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        else: 
            self.device = device
        print(f"VTONMetrics on: {self.device}")

        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity().to(self.device)
        
    def calculate(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.to(self.device)
        targets = targets.to(self.device)

        ssim_val = self.ssim_metric(preds, targets)
        psnr_val = self.psnr_metric(preds, targets)
        preds_lpips = preds * 2.0 - 1.0
        targets_lpips = targets *2.0 -1.0
        lpips_val = self.lpips_metric(preds_lpips, targets_lpips)
        
        return {
            'SSIM': ssim_val.item(),
            'PSNR': psnr_val.item(),
            'LPIPS': lpips_val.item()
        }      
