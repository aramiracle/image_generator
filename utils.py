import torch
import math
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

def normalize(x):
    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    return x

def lnl1_metric(prediction, target):
    max = torch.maximum(prediction, target) + 1e-3 * torch.ones_like(prediction)
    norm1 = torch.absolute(prediction - target)
    normalize_norm1 = torch.mean(torch.mul(norm1, max.pow(-1)))
    lnl1_value = -20*torch.log10(normalize_norm1)

    return lnl1_value

def PSNR_SSIM_LNL1_loss(prediction, target):
    psnr = PeakSignalNoiseRatio()
    psnr_value = psnr(prediction, target)
    psnr_loss = -psnr_value

    # Calculate Structural Similarity Index (SSIM)
    ssim = StructuralSimilarityIndexMeasure(data_range=1)
    ssim_value = ssim(prediction, target)

    # Calculate a function which maps [0,1] to (inf, 0]
    ssim_loss = torch.tan(math.pi / 2 * (1 - ssim_value))

    lnl1_loss = -lnl1_metric(prediction, target)

    return psnr_loss + 20 * ssim_loss + lnl1_loss