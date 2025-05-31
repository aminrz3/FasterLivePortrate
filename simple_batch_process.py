#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Simple batch processing for FasterLivePortrait
# Process multiple source images with a single driving video

import os
import argparse
import datetime
import glob
import cv2
import time
import numpy as np
import concurrent.futures
import threading
import pickle
import shutil
import subprocess
from tqdm import tqdm
from omegaconf import OmegaConf
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.utils.utils import video_has_audio
from src.utils import logger
import platform

# Set up constants
if platform.system().lower() == 'windows':
    FFMPEG = "third_party/ffmpeg-7.0.1-full_build/bin/ffmpeg.exe"
else:
    FFMPEG = "ffmpeg"

# Setup directories
project_dir = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = os.environ.get("FLIP_CHECKPOINT_DIR", os.path.join(project_dir, "checkpoints"))
log_dir = os.path.join(project_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
result_dir = os.path.join(project_dir, "results")
os.makedirs(result_dir, exist_ok=True)

# Setup logger
logger_f = logger.get_logger("batch_process", log_file=os.path.join(log_dir, "log_batch.log"))


def process_single_image(image_path, driving_video_path, save_dir, args):
    """Process a single source image with a driving video"""
    # Get base filenames
    image_basename = os.path.basename(image_path)
    image_name = os.path.splitext(image_basename)[0]
    image_ext = os.path.splitext(image_basename)[1]
    
    # Get custom output settings if provided
    output_fps = args.output_fps
    
    # Get original source image dimensions
    src_img = cv2.imread(image_path)
    src_h, src_w = src_img.shape[:2]
    logger_f.info(f"Source image dimensions: {src_w}x{src_h}")
    
    # Use source image width as default if not specified
    output_width = args.output_width if args.output_width is not None else src_w
    
    # Get configuration
    infer_cfg = OmegaConf.load(args.cfg)
    infer_cfg.infer_params.flag_pasteback = True  # Always use pasteback

    # Initialize the pipeline
    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=args.animal)
    
    # Prepare the source image
    ret = pipe.prepare_source(image_path, realtime=False)
    if not ret:
        logger_f.warning(f"No face detected in {image_path}! Skipping!")
        return None
    
    # Set up temporary MP4 output path (will be converted to GIF later)
    tmp_video_path = os.path.join(save_dir, f"{image_name}_tmp.mp4")
    
    # Final GIF output path - using the exact same name as the image
    gif_output_path = os.path.join(save_dir, f"{image_name}.gif")
    
    # Process the video
    logger_f.info(f"Processing {image_path} with {driving_video_path}")
    
    try:
        # Open the video
        vcap = cv2.VideoCapture(driving_video_path)
        fps = int(vcap.get(cv2.CAP_PROP_FPS))
        h, w = pipe.src_imgs[0].shape[:2]
        
        # Setup video writer for original size only
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vout_org = cv2.VideoWriter(tmp_video_path, fourcc, fps, (w, h))
        
        infer_times = []
        frame_ind = 0
        while vcap.isOpened():
            ret, frame = vcap.read()
            if not ret:
                break
            t0 = time.time()
            first_frame = frame_ind == 0
            dri_crop, out_crop, out_org, dri_motion_info = pipe.run(frame, pipe.src_imgs[0], pipe.src_infos[0],
                                                                first_frame=first_frame)
            frame_ind += 1
            if out_crop is None:
                logger_f.warning(f"No face in driving frame:{frame_ind}")
                continue
            
            infer_times.append(time.time() - t0)
            
            # Write original size output only
            if isinstance(out_org, np.ndarray):
                out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
            else:  # If it's a tensor
                out_org = cv2.cvtColor(out_org.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
                
            vout_org.write(out_org)
        
        vcap.release()
        vout_org.release()
        
        logger_f.info(f"Converting {tmp_video_path} to GIF: {gif_output_path}")
        
        # Convert MP4 to GIF using FFMPEG
        # Using a palette for better GIF quality
        palette_path = os.path.join(save_dir, f"{image_name}_palette.png")
        
        # Use custom FPS if provided, otherwise use source fps
        gif_fps = output_fps if output_fps is not None else fps
        logger_f.info(f"Using frame rate: {gif_fps} fps for output GIF")
        
        # Calculate output height based on output width, maintaining aspect ratio of the output video
        output_height = int(h * output_width / w)
        
        # If output dimensions match the processed video, no need to scale
        if output_width == w and output_height == h:
            scale_filter = ""  # No scaling
            logger_f.info(f"Keeping pipeline output size {w}x{h}")
        else:
            # Resize to match desired dimensions
            scale_filter = f"scale={output_width}:{output_height}:flags=lanczos"
            logger_f.info(f"Resizing output to {output_width}x{output_height} (source image width)")
            
        # Combine filters
        if scale_filter:
            palette_filter = f"fps={gif_fps},{scale_filter},palettegen"
            gif_filter = f"fps={gif_fps},{scale_filter}[x];[x][1:v]paletteuse=dither=sierra2_4a"
        else:
            palette_filter = f"fps={gif_fps},palettegen"
            gif_filter = f"fps={gif_fps}[x];[x][1:v]paletteuse=dither=sierra2_4a"
        
        # Step 1: Generate a palette from the video
        subprocess.call([
            FFMPEG, "-i", tmp_video_path, 
            "-vf", palette_filter, 
            palette_path, "-y"
        ])
        
        # Step 2: Use the palette to create the GIF
        subprocess.call([
            FFMPEG, "-i", tmp_video_path, "-i", palette_path,
            "-filter_complex", gif_filter,
            gif_output_path, "-y"
        ])
        
        # Remove temporary files
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)
        if os.path.exists(palette_path):
            os.remove(palette_path)
        
        logger_f.info(f"Processed {image_path}, inference time: median={np.median(infer_times) * 1000:.2f}ms/frame")
        
        return {
            "source_image": image_path,
            "output_gif": gif_output_path,
            "frame_count": frame_ind
        }
        
    except Exception as e:
        logger_f.error(f"Error processing {image_path}: {str(e)}")
        import traceback
        logger_f.error(traceback.format_exc())
        return None


def batch_process(source_dir, driving_video_path, save_dir, args):
    """Process multiple source images with a single driving video"""
    
    # Get all image files from the source directory
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
        image_files.extend(glob.glob(os.path.join(source_dir, f"*.{ext}")))
    
    if not image_files:
        logger_f.error(f"No image files found in {source_dir}")
        return False
    
    # Remove duplicates if any
    image_files = list(set(image_files))
    
    if args.batch_size:
        image_files = image_files[:args.batch_size]
    
    logger_f.info(f"Found {len(image_files)} images to process")
    
    # Create a timestamp-based save directory
    if save_dir is None:
        save_dir = os.path.join(result_dir, f"batch-{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}")
    
    os.makedirs(save_dir, exist_ok=True)
    logger_f.info(f"Saving results to {save_dir}")
    
    # Create a threadpool for parallel processing - each thread will have its own pipeline instance
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit tasks for each image
        future_to_image = {
            executor.submit(process_single_image, image_path, driving_video_path, save_dir, args): image_path
            for image_path in image_files
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_image), total=len(image_files)):
            image_path = future_to_image[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    logger_f.info(f"Successfully processed {image_path}")
                else:
                    logger_f.warning(f"Failed to process {image_path}")
            except Exception as e:
                logger_f.error(f"Error processing {image_path}: {str(e)}")
                # Print the full traceback for debugging
                import traceback
                logger_f.error(traceback.format_exc())
    
    # Create a summary of processed images
    summary_path = os.path.join(save_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Batch processing summary\n")
        f.write(f"Driving video: {driving_video_path}\n")
        f.write(f"Total images processed: {len(results)}/{len(image_files)}\n\n")
        
        for i, result in enumerate(results):
            f.write(f"{i+1}. Source: {os.path.basename(result['source_image'])}\n")
            f.write(f"   Output GIF: {os.path.basename(result['output_gif'])}\n\n")
    
    logger_f.info(f"Batch processing complete. Processed {len(results)}/{len(image_files)} images")
    logger_f.info(f"Summary saved to {summary_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch process multiple source images with a single driving video")
    parser.add_argument("--source_dir", required=True, help="Directory containing source images")
    parser.add_argument("--driving_video", required=True, help="Path to the driving video")
    parser.add_argument("--save_dir", default=None, help="Directory to save results (default: auto-generated)")
    parser.add_argument("--cfg", default="configs/trt_infer.yaml", help="Path to config file")
    parser.add_argument("--animal", action="store_true", help="Use animal model")
    parser.add_argument("--max_workers", type=int, default=2, help="Maximum number of worker threads")
    parser.add_argument("--batch_size", type=int, default=None, help="Limit the number of images to process")
    parser.add_argument("--output_fps", type=int, default=None, help="Output GIF frame rate (default: same as driving video)")
    parser.add_argument("--output_width", type=int, default=None, help="Output GIF width in pixels (default: same as source image width)")
    
    args = parser.parse_args()
    
    batch_process(
        args.source_dir,
        args.driving_video,
        args.save_dir,
        args
    )


if __name__ == "__main__":
    main()
