import argparse
import os
import torch
import cv2
import numpy as np
from lightx2v import LightX2VPipeline
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(video_path1, video_path2):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    
    ssim_scores = []
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
            
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        score = ssim(gray1, gray2)
        ssim_scores.append(score)
        
    cap1.release()
    cap2.release()
    
    if not ssim_scores:
        return 0.0
        
    return np.mean(ssim_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="2.1", choices=["2.1", "2.2"])
    parser.add_argument("--task", type=str, default="t2v", choices=["t2v", "i2v"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--trt_engine_path", type=str, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    prompt = "A cinematic drone shot of a futuristic city at sunset, neon lights glowing."
    neg_prompt = "low quality, bad tokens, blurry"
    
    def run_pipeline(use_trt, suffix):
        print(f"--- Running {'TRT' if use_trt else 'PyTorch'} Pipeline ---")
        
        # Determine model class name based on version/task
        # Example: model_cls="wan2.1" or "wan2.2" (if registered)
        model_cls = f"wan{args.version}" 
        
        pipe = LightX2VPipeline(
            model_path=args.model_path,
            model_cls=model_cls,
            task=args.task,
        )
        
        # Config Overrides
        overrides = {
            "infer_steps": 20, # Short run for verify
            "height": 480,
            "width": 832,
            "num_frames": 33 if args.version=="2.1" else 17, # Short
        }
        
        if use_trt:
            overrides["vae_type"] = "tensorrt"
            overrides["trt_engine_path"] = args.trt_engine_path
            overrides["trt_vae_version"] = args.version
            
        pipe.create_generator(**overrides)
        
        save_path = os.path.join(args.output_dir, f"output_{suffix}.mp4")
        
        pipe.generate(
            seed=42,
            prompt=prompt,
            negative_prompt=neg_prompt,
            save_result_path=save_path
        )
        return save_path

    if args.verify:
        # Run PyTorch
        torch_out = run_pipeline(use_trt=False, suffix="torch")
        
        # Run TRT
        trt_out = run_pipeline(use_trt=True, suffix="trt")
        
        print("Calculating SSIM...")
        score = calculate_ssim(torch_out, trt_out)
        print(f"SSIM Score: {score:.4f}")
        
        if score > 0.95:
            print("SUCCESS: TRT VAE matches PyTorch VAE.")
        else:
            print("WARNING: Low SSIM match.")
    else:
        # Just Run TRT
        run_pipeline(use_trt=True, suffix="trt")

if __name__ == "__main__":
    main()
