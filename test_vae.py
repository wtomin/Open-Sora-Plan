import math
import random
import argparse
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
import torch
import os
import sys
sys.path.append(".")

from opensora.models.ae import getae_wrapper
from opensora.dataset.transform import CenterCropVideo, resize
from opensora.models.ae.videobase import CausalVAEModel


def main():
    device = "cuda"
    # vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir='cache_dir', **kwarg).to(device)
    vae = getae_wrapper('CausalVAEModel_4x8x8')("LanguageBind/Open-Sora-Plan-v1.1.0/vae").to(device, dtype=torch.float32)
    vae.vae.enable_tiling()
    vae.vae.tile_overlap_factor = 0.25
    vae.eval()
    vae = vae.to(device)
    # vae = vae.half()  #use vae fp32
    torch_folder = "save_torch_npy"
    if not os.path.exists(torch_folder):
        os.makedirs(torch_folder, exist_ok=True)
    
    with torch.no_grad():
        x_vae = np.load(os.path.join(torch_folder, "x_vae.npy"))
        x_vae = torch.tensor(x_vae)
        x_vae = x_vae.to(device, dtype=torch.float32)  # b c t h w
        latents = vae.encode(x_vae)
        np.save(os.path.join(torch_folder, "latents_torch.npy"),
                latents.cpu().float().numpy())
        video_recon = vae.decode(latents)  # b t c h w
        np.save(os.path.join(torch_folder, "video_recon_torch.npy"),
                video_recon.cpu().float().numpy())

if __name__ == "__main__":
    main()