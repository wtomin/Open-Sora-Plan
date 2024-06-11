import math
import os
import torch
import argparse
import numpy as np


from torchvision.utils import save_image
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer

import os, sys

from opensora.models.ae import ae_stride_config, getae, getae_wrapper
from opensora.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper
from opensora.models.diffusion.latte.modeling_latte import LatteT2V
from opensora.models.text_encoder import get_text_enc
from opensora.utils.utils import save_video_grid

sys.path.append(os.path.split(sys.path[0])[0])
from pipeline_videogen import VideoGenPipeline

import imageio


def main(args):
    # torch.manual_seed(args.seed) 
    torch.set_grad_enabled(False)
    device = "cuda"

    vae = getae_wrapper('CausalVAEModel_4x8x8')("LanguageBind/Open-Sora-Plan-v1.1.0/vae").to(device, dtype=torch.float32)
    vae.vae.enable_tiling()
    vae.vae.tile_overlap_factor = 0.25
    vae.vae_scale_factor = ae_stride_config[args.ae]
    # Load model:
    transformer_model = LatteT2V.from_pretrained(args.model_path, subfolder="65x512x512",torch_dtype=torch.float32).to(device)
    latent_model_input = np.load("latents.npy")
    current_timestep = torch.tensor([894], dtype=torch.int32)
    
    prompt_embeds = np.load('prompt_embeds_torch.npy')
    prompt_embeds = torch.tensor(prompt_embeds).to(device, dtype=torch.float32)
    prompt_embeds_mask = np.load('prompt_embeds_mask_torch.npy')
    prompt_embeds_mask = torch.tensor(prompt_embeds_mask).to(device, dtype=torch.float32)
    if prompt_embeds.ndim == 3:
        prompt_embeds = prompt_embeds.unsqueeze(1)  # b l d -> b 1 l d
    with torch.no_grad():
        noise_pred = transformer_model(
                        latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=current_timestep,
                        added_cond_kwargs={},
                        enable_temporal_attentions=True,
                        return_dict=False,
                        encoder_attention_mask=prompt_embeds_mask,
                    )[0]
    
    torch_folder = "save_torch_npy"
    if not os.path.exists(torch_folder):
        os.makedirs(torch_folder, exist_ok=True)
    np.save(os.path.join(torch_folder, "noise_pred_torch.npy"), noise_pred.cpu().detach().numpy())