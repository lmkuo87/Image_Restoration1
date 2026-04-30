import os
import torch
import torch.utils.checkpoint
import torch.cuda
import random
import numpy as np
import argparse
from typing import List
from PIL import Image

from FaithDiff.create_FaithDiff_model import FaithDiff_pipeline
from CKPT_PTH import SDXL_PATH, FAITHDIFF_PATH, VAE_FP16_PATH
from utils.color_fix import wavelet_color_fix, adain_color_fix
from utils.image_process import check_image_size, create_hdr_effect
from utils.system import torch_gc

MAX_SEED = np.iinfo(np.int32).max

def parse_args():
    parser = argparse.ArgumentParser(description="FaithDiff Local Batch Processing (No LLaVA)")
    
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input images directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output images directory")
    parser.add_argument("--prompt", type=str, default="", help="Global prompt for all images (Optional)")

    parser.add_argument("--cpu_offload", action='store_true', default=False)
    parser.add_argument("--use_fp8", action='store_true', default=False)
    
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--scale_factor", type=float, default=2.0)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_overlap", type=float, default=0.5)
    parser.add_argument("--color_fix", type=str, choices=["wavelet", "adain", "nofix"], default="adain")
    parser.add_argument("--start_point", type=str, choices=["lr", "noise"], default="lr")
    parser.add_argument("--hdr", type=float, default=0.0)
    
    return parser.parse_args()

@torch.no_grad()
def process(
    pipe,
    image: Image.Image,
    user_prompt: str,
    args,
    Diffusion_device
) -> np.ndarray:

    w, h = image.size
    w = int(w * args.scale_factor)
    h = int(h * args.scale_factor)
    image = image.resize((w, h), Image.LANCZOS)
    input_image, width_init, height_init, width_now, height_now = check_image_size(image)
    
    negative_prompt_init = ""
    generator = torch.Generator(device=Diffusion_device).manual_seed(args.seed)    
    input_image = create_hdr_effect(input_image, args.hdr)

    gen_image = pipe(
        lr_img=input_image, 
        prompt=user_prompt, 
        negative_prompt=negative_prompt_init, 
        num_inference_steps=args.num_inference_steps, 
        guidance_scale=args.guidance_scale, 
        generator=generator, 
        start_point=args.start_point, 
        height=height_now, 
        width=width_now,  
        overlap=args.latent_tiled_overlap, 
        target_size=(args.latent_tiled_size, args.latent_tiled_size)
    ).images[0]
    
    torch_gc()
    cropped_image = gen_image.crop((0, 0, width_init, height_init))
    
    if args.color_fix == 'nofix':
        out_image = cropped_image
    elif args.color_fix == 'wavelet':
        out_image = wavelet_color_fix(cropped_image, image)
    elif args.color_fix == 'adain':
        out_image = adain_color_fix(cropped_image, image)

    return np.array(out_image)

def main():
    args = parse_args()
    
    if torch.cuda.is_available():
        Diffusion_device = 'cuda:0'
    else:
        raise ValueError('Currently support CUDA only.')
        
    # 載入 FaithDiff 模型
    print("Loading FaithDiff pipeline...")
    pipe = FaithDiff_pipeline(sdxl_path=SDXL_PATH, VAE_FP16_path=VAE_FP16_PATH, FaithDiff_path=FAITHDIFF_PATH, use_fp8=args.use_fp8)
    pipe = pipe.to(Diffusion_device)

    pipe.set_encoder_tile_settings()
    pipe.enable_vae_tiling()

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()

    os.makedirs(args.output_dir, exist_ok=True)
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"No valid images found in {args.input_dir}")
        return

    print(f"Found {len(image_files)} images. Starting processing...")

    for img_name in image_files:
        img_path = os.path.join(args.input_dir, img_name)
        out_path = os.path.join(args.output_dir, img_name)
        
        print(f"Processing: {img_name}...")
        try:
            image = Image.open(img_path).convert("RGB")
            
            # 直接使用 args.prompt，若無則預設為空字串
            current_prompt = args.prompt
                
            # image process
            result_np = process(
                pipe=pipe,
                image=image,
                user_prompt=current_prompt,
                args=args,
                Diffusion_device=Diffusion_device
            )
            
            # save image
            result_img = Image.fromarray(result_np)
            result_img.save(out_path)
            print(f"Saved result to: {out_path}")
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    print("All images processed successfully!")

if __name__ == "__main__":
    main()
