from diffusers import DiffusionPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
import torch
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
import torch
import gc


from null import *
from local import *

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")

model = "stabilityai/stable-diffusion-xl-base-1.0"

ldm_stable = DiffusionPipeline.from_pretrained(
        model,
        scheduler=scheduler,
        torch_dtype=torch.float16,
    ).to(device)

# "CompVis/stable-diffusion-v1-4",
ldm_stable.enable_model_cpu_offload()

tokenizer = ldm_stable.tokenizer
ldm_stable.disable_xformers_memory_efficient_attention()

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')



null_inversion = NullInversion(ldm_stable)
image_path = "/data/hyunwook/walk/null-text-inversion-colab/tay.jpg"
prompt = "a woman with blonde hair and a blue scarf"
(image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True)

print("Modify or remove offsets according to your image!")
torch.cuda.empty_cache()
gc.collect()