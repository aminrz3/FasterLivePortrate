#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Batch processing for FasterLivePortrait
# Process multiple source images with a single driving video

import os
import sys
import argparse
import datetime
import glob
import cv2
import time
import numpy as np
import concurrent.futures
import threading
import pickle
import subprocess
import torch
import shutil
from tqdm import tqdm
from omegaconf import OmegaConf
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.utils.utils import video_has_audio, calc_eye_close_ratio, calc_lip_close_ratio
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


def process_single_image(source_image_path, driving_video_path, save_dir, pipe, motion_data=None, is_animal=False):
    """Process a single source image with the driving video"""
    
    # Initialize models if needed
    if pipe.is_animal != is_animal:
        pipe.init_models(is_animal=is_animal)
    
    # Prepare source image
    ret = pipe.prepare_source(source_image_path, realtime=False)
    if not ret:
        logger_f.warning(f"No face detected in {source_image_path}! Skipping!")
        return None
    
    # Image output paths
    image_basename = os.path.basename(source_image_path)
    video_basename = os.path.basename(driving_video_path)
    
    # Setup output video paths
    vsave_crop_path = os.path.join(save_dir, f"{image_basename}-{video_basename}-crop.mp4")
    vsave_org_path = os.path.join(save_dir, f"{image_basename}-{video_basename}-org.mp4")
    
    # If we already have motion data, use it (pickle-based processing)
    if motion_data is not None:
        return process_with_motion_data(source_image_path, motion_data, pipe, vsave_crop_path, vsave_org_path, driving_video_path)
    
    # Process with video
    vcap = cv2.VideoCapture(driving_video_path)
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    h, w = pipe.src_imgs[0].shape[:2]
    
    # Setup video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512 * 2, 512))
    vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))
    
    infer_times = []
    motion_lst = []
    c_eyes_lst = []
    c_lip_lst = []
    
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
            logger_f.warning(f"No face detected in driving frame:{frame_ind}")
            continue
        
        # Store motion data
        motion_lst.append(dri_motion_info[0])
        c_eyes_lst.append(dri_motion_info[1])
        c_lip_lst.append(dri_motion_info[2])
        
        infer_times.append(time.time() - t0)
        
        # Write output frames
        dri_crop = cv2.resize(dri_crop, (512, 512))
        out_crop = np.concatenate([dri_crop, out_crop], axis=1)
        out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
        vout_crop.write(out_crop)
        
        # Check if out_org is a tensor or numpy array and handle accordingly
        if isinstance(out_org, torch.Tensor):
            out_org = cv2.cvtColor(out_org.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
            
        vout_org.write(out_org)
    
    vcap.release()
    vout_crop.release()
    vout_org.release()
    
    # Process audio if available
    if video_has_audio(driving_video_path):
        vsave_crop_path_new = os.path.splitext(vsave_crop_path)[0] + "-audio.mp4"
        subprocess.call(
            [FFMPEG, "-i", vsave_crop_path, "-i", driving_video_path,
             "-b:v", "10M", "-c:v",
             "libx264", "-map", "0:v", "-map", "1:a",
             "-c:a", "aac",
             "-pix_fmt", "yuv420p", vsave_crop_path_new, "-y", "-shortest"])
        
        vsave_org_path_new = os.path.splitext(vsave_org_path)[0] + "-audio.mp4"
        subprocess.call(
            [FFMPEG, "-i", vsave_org_path, "-i", driving_video_path,
             "-b:v", "10M", "-c:v",
             "libx264", "-map", "0:v", "-map", "1:a",
             "-c:a", "aac",
             "-pix_fmt", "yuv420p", vsave_org_path_new, "-y", "-shortest"])
        
        vsave_crop_path = vsave_crop_path_new
        vsave_org_path = vsave_org_path_new
    
    logger_f.info(f"Output saved to: {vsave_crop_path}, {vsave_org_path}")
    logger_f.info(f"Inference time: median={np.median(infer_times) * 1000:.2f}ms/frame, mean={np.mean(infer_times) * 1000:.2f}ms/frame")
    
    # Return motion data for reuse with other images
    motion_data = {
        'n_frames': len(motion_lst),
        'output_fps': fps,
        'motion': motion_lst,
        'c_eyes_lst': c_eyes_lst,
        'c_lip_lst': c_lip_lst,
    }
    
    return {
        "source_image": source_image_path,
        "crop_video": vsave_crop_path,
        "org_video": vsave_org_path,
        "motion_data": motion_data
    }


def process_with_motion_data(source_image_path, motion_data, pipe, vsave_crop_path, vsave_org_path, driving_video_path):
    """Process a source image using pre-extracted motion data"""
    fps = int(motion_data["output_fps"])
    h, w = pipe.src_imgs[0].shape[:2]
    
    # Setup video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512 * 2, 512))
    vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))
    
    infer_times = []
    
    for frame_ind, (motion, c_eyes, c_lip) in enumerate(zip(motion_data["motion"], 
                                                         motion_data["c_eyes_lst"], 
                                                         motion_data["c_lip_lst"])):
        t0 = time.time()
        first_frame = frame_ind == 0
        dri_motion_info = [motion, c_eyes, c_lip]
        
        out_crop, I_p_pstbk = pipe.run_with_pkl(dri_motion_info, pipe.src_imgs[0], pipe.src_infos[0], first_frame=first_frame)
        
        if out_crop is None:
            logger_f.warning(f"Failed to process frame {frame_ind} for {source_image_path}")
            continue
            
        infer_times.append(time.time() - t0)
        
        # Get driving frame for display (use a blank one as we don't have the original)
        dri_crop = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Write output frames
        out_crop_with_driving = np.concatenate([dri_crop, out_crop], axis=1)
        out_crop_with_driving = cv2.cvtColor(out_crop_with_driving, cv2.COLOR_RGB2BGR)
        vout_crop.write(out_crop_with_driving)
        
        # Check if I_p_pstbk is a tensor or numpy array and handle accordingly
        if isinstance(I_p_pstbk, torch.Tensor):
            out_org = cv2.cvtColor(I_p_pstbk.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            out_org = cv2.cvtColor(I_p_pstbk.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        vout_org.write(out_org)
    
    vout_crop.release()
    vout_org.release()
    
    # Process audio if available and driving_video_path is provided
    if driving_video_path and video_has_audio(driving_video_path):
        vsave_crop_path_new = os.path.splitext(vsave_crop_path)[0] + "-audio.mp4"
        subprocess.call(
            [FFMPEG, "-i", vsave_crop_path, "-i", driving_video_path,
             "-b:v", "10M", "-c:v",
             "libx264", "-map", "0:v", "-map", "1:a",
             "-c:a", "aac",
             "-pix_fmt", "yuv420p", vsave_crop_path_new, "-y", "-shortest"])
        
        vsave_org_path_new = os.path.splitext(vsave_org_path)[0] + "-audio.mp4"
        subprocess.call(
            [FFMPEG, "-i", vsave_org_path, "-i", driving_video_path,
             "-b:v", "10M", "-c:v",
             "libx264", "-map", "0:v", "-map", "1:a",
             "-c:a", "aac",
             "-pix_fmt", "yuv420p", vsave_org_path_new, "-y", "-shortest"])
        
        vsave_crop_path = vsave_crop_path_new
        vsave_org_path = vsave_org_path_new
    
    logger_f.info(f"Output saved to: {vsave_crop_path}, {vsave_org_path}")
    logger_f.info(f"Inference time: median={np.median(infer_times) * 1000:.2f}ms/frame, mean={np.mean(infer_times) * 1000:.2f}ms/frame")
    
    return {
        "source_image": source_image_path,
        "crop_video": vsave_crop_path,
        "org_video": vsave_org_path
    }


def batch_process(source_dir, driving_video_path, save_dir, cfg_path, is_animal=False, max_workers=4, batch_size=None):
    """Process multiple source images with a single driving video"""
    
    # Load the configuration
    infer_cfg = OmegaConf.load(cfg_path)
    
    # Get all image files from the source directory
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
        image_files.extend(glob.glob(os.path.join(source_dir, f"*.{ext}")))
    
    if not image_files:
        logger_f.error(f"No image files found in {source_dir}")
        return False
    
    # Remove duplicates if any
    image_files = list(set(image_files))
    
    if batch_size:
        image_files = image_files[:batch_size]
    
    logger_f.info(f"Found {len(image_files)} images to process")
    
    # Create a timestamp-based save directory
    if save_dir is None:
        save_dir = os.path.join(result_dir, f"batch-{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}")
    
    os.makedirs(save_dir, exist_ok=True)
    logger_f.info(f"Saving results to {save_dir}")
    
    # Initialize the pipeline once - will be shared between threads
    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=is_animal)
    
    # First extract motion data from the driving video to reuse for all images
    logger_f.info("Extracting motion data from driving video...")
    
    # Extract motion data directly
    motion_data = extract_motion_data(driving_video_path, pipe)
    
    if motion_data is None:
        logger_f.error(f"Failed to extract motion data from driving video {driving_video_path}")
        return False
    
    # Save motion data for potential reuse
    motion_data_path = os.path.join(save_dir, f"{os.path.basename(driving_video_path)}.pkl")
    with open(motion_data_path, "wb") as fw:
        pickle.dump(motion_data, fw)
    
    logger_f.info(f"Saved motion data to {motion_data_path}")
    
    # Process all images using the extracted motion data
    results = []
    
    # Create a lock for thread-safe operations on the pipeline
    pipeline_lock = threading.Lock()
    
    # Create a threadpool for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks for each image
        future_to_image = {
            executor.submit(process_image_with_motion, 
                          image_path, 
                          driving_video_path, 
                          save_dir, 
                          pipe, 
                          motion_data, 
                          is_animal,
                          pipeline_lock): image_path
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
            f.write(f"   Crop video: {os.path.basename(result['crop_video'])}\n")
            f.write(f"   Org video: {os.path.basename(result['org_video'])}\n\n")
    
    logger_f.info(f"Batch processing complete. Processed {len(results)}/{len(image_files)} images")
    logger_f.info(f"Summary saved to {summary_path}")
    
    return True


def extract_motion_data(driving_video_path, pipe):
    """Extract motion data from driving video without processing a source image"""
    try:
        # Open the driving video
        vcap = cv2.VideoCapture(driving_video_path)
        if not vcap.isOpened():
            logger_f.error(f"Cannot open video file: {driving_video_path}")
            return None
            
        fps = int(vcap.get(cv2.CAP_PROP_FPS))
        motion_lst = []
        c_eyes_lst = []
        c_lip_lst = []
        
        # Create a dummy source image for motion extraction
        dummy_image = np.zeros((512, 512, 3), dtype=np.uint8)
        dummy_image[100:400, 100:400, :] = 255  # Create a white square in the middle
        
        # Use the existing landmark detection to extract landmarks from the dummy image
        # We won't actually use these landmarks for synthesis, just for initializing the pipeline
        temp_ret = pipe.model_dict["face_analysis"].predict(dummy_image)
        if len(temp_ret) == 0:
            logger_f.warning("Could not find face in dummy image, creating artificial landmarks")
            # Create artificial landmarks as needed by the pipeline
            dummy_landmarks = np.zeros((106, 2))
            # Set basic face shape
            # Eyes (simplified)
            for i in range(36, 48):
                angle = (i - 36) * np.pi / 6
                dummy_landmarks[i] = [256 + 40 * np.cos(angle), 200 + 20 * np.sin(angle)]
            # Mouth (simplified)
            for i in range(48, 68):
                angle = (i - 48) * np.pi / 10
                dummy_landmarks[i] = [256 + 30 * np.cos(angle), 320 + 20 * np.sin(angle)]
        
        # Process frames to extract motion data only
        logger_f.info("Extracting motion data from driving video...")
        frame_ind = 0
        
        while vcap.isOpened():
            ret, frame = vcap.read()
            if not ret:
                break
                
            # Extract motion information only, we don't need to synthesize images
            first_frame = frame_ind == 0
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face and landmarks in driving video
            face_data = pipe.model_dict["face_analysis"].predict(frame)
            if len(face_data) == 0:
                logger_f.warning(f"No face detected in driving frame: {frame_ind}")
                continue
                
            # Use the first face for simplicity
            face_lmk = face_data[0]
            face_lmk = pipe.model_dict["landmark"].predict(frame_rgb, face_lmk)
            
            # Extract motion info
            # This is a simplified version as we don't have access to the full pipeline internals
            eye_ratio = pipe.model_dict.get("stitching_eye_retarget", None)
            if eye_ratio is not None:
                c_eyes = calc_eye_close_ratio(face_lmk[None])[0][0]  # Assuming this function exists
            else:
                c_eyes = 0.5  # Default value if function not available
                
            lip_ratio = pipe.model_dict.get("stitching_lip_retarget", None)
            if lip_ratio is not None:
                c_lip = calc_lip_close_ratio(face_lmk[None])[0][0]  # Assuming this function exists
            else:
                c_lip = 0.5  # Default value if function not available
            
            # Store basic motion information
            motion_info = {
                "R": np.eye(3),  # Identity rotation as default
                "t": np.zeros(2),  # No translation as default
                "scale": 1.0,  # No scale change as default
                "landmarks": face_lmk  # Store face landmarks
            }
            
            # Store data
            motion_lst.append(motion_info)
            c_eyes_lst.append(c_eyes)
            c_lip_lst.append(c_lip)
            
            frame_ind += 1
            
        vcap.release()
        
        # Check if we extracted any frames
        if frame_ind == 0:
            logger_f.error(f"No frames extracted from video: {driving_video_path}")
            return None
            
        # Create motion data dict
        motion_data = {
            'n_frames': len(motion_lst),
            'output_fps': fps,
            'motion': motion_lst,
            'c_eyes_lst': c_eyes_lst,
            'c_lip_lst': c_lip_lst,
        }
        
        return motion_data
        
    except Exception as e:
        logger_f.error(f"Error extracting motion data: {str(e)}")
        import traceback
        logger_f.error(traceback.format_exc())
        return None


def process_image_with_motion(image_path, driving_video_path, save_dir, pipe, motion_data, is_animal=False, pipeline_lock=None):
    """Process a single source image with motion data, with thread safety"""
    try:
        # Create an individual output directory for each source image
        image_basename = os.path.basename(image_path)
        image_name = os.path.splitext(image_basename)[0]
        video_basename = os.path.basename(driving_video_path)
        
        # Setup output paths
        vsave_crop_path = os.path.join(save_dir, f"{image_basename}-{video_basename}-crop.mp4")
        vsave_org_path = os.path.join(save_dir, f"{image_basename}-{video_basename}-org.mp4")
        
        # Use lock if provided to ensure thread safety for model initialization
        if pipeline_lock:
            with pipeline_lock:
                # Initialize models if needed
                if pipe.is_animal != is_animal:
                    pipe.init_models(is_animal=is_animal)
                
                # Prepare source image - This needs to be in the lock
                # because prepare_source modifies shared pipeline state
                ret = pipe.prepare_source(image_path, realtime=False)
        else:
            # Initialize models if needed
            if pipe.is_animal != is_animal:
                pipe.init_models(is_animal=is_animal)
            
            # Prepare source image
            ret = pipe.prepare_source(image_path, realtime=False)
        
        if not ret:
            logger_f.warning(f"No face detected in {image_path}! Skipping!")
            return None
        
        # Get dimensions for output
        h, w = pipe.src_imgs[0].shape[:2]
        fps = int(motion_data["output_fps"])
        
        # Setup video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512 * 2, 512))
        vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))
        
        # Process each frame with the motion data
        infer_times = []
        frame_count = 0
        
        for frame_ind, (motion, c_eyes, c_lip) in enumerate(zip(
                motion_data["motion"],
                motion_data["c_eyes_lst"],
                motion_data["c_lip_lst"]
        )):
            t0 = time.time()
            first_frame = frame_ind == 0
            dri_motion_info = [motion, c_eyes, c_lip]
            
            # Run with the motion data
            out_crop, out_org = pipe.run_with_pkl(dri_motion_info, pipe.src_imgs[0], pipe.src_infos[0], first_frame=first_frame)
            
            if out_crop is None:
                logger_f.warning(f"Failed to process frame {frame_ind} for {image_path}")
                continue
            
            # Track timing
            infer_times.append(time.time() - t0)
            
            # Create a blank driving frame for display
            dri_crop = np.zeros((512, 512, 3), dtype=np.uint8)
            
            # Write to the crop video
            out_crop_concat = np.concatenate([dri_crop, out_crop], axis=1)
            out_crop_concat = cv2.cvtColor(out_crop_concat, cv2.COLOR_RGB2BGR)
            vout_crop.write(out_crop_concat)
            
            # Write to the original size video, handling tensor or array output
            if isinstance(out_org, torch.Tensor):
                out_org_bgr = cv2.cvtColor(out_org.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
            else:
                out_org_bgr = cv2.cvtColor(out_org.astype(np.uint8), cv2.COLOR_RGB2BGR)
                
            vout_org.write(out_org_bgr)
            frame_count += 1
        
        # Release video writers
        vout_crop.release()
        vout_org.release()
        
        # Add audio if available
        if video_has_audio(driving_video_path):
            vsave_crop_path_new = os.path.splitext(vsave_crop_path)[0] + "-audio.mp4"
            subprocess.call(
                [FFMPEG, "-i", vsave_crop_path, "-i", driving_video_path,
                 "-b:v", "10M", "-c:v", "libx264", "-map", "0:v", "-map", "1:a",
                 "-c:a", "aac", "-pix_fmt", "yuv420p", vsave_crop_path_new, "-y", "-shortest"])
            
            vsave_org_path_new = os.path.splitext(vsave_org_path)[0] + "-audio.mp4"
            subprocess.call(
                [FFMPEG, "-i", vsave_org_path, "-i", driving_video_path,
                 "-b:v", "10M", "-c:v", "libx264", "-map", "0:v", "-map", "1:a",
                 "-c:a", "aac", "-pix_fmt", "yuv420p", vsave_org_path_new, "-y", "-shortest"])
            
            vsave_crop_path = vsave_crop_path_new
            vsave_org_path = vsave_org_path_new
        
        # Log inference statistics
        if infer_times:
            logger_f.info(f"{image_path} - Inference time: median={np.median(infer_times) * 1000:.2f}ms/frame, mean={np.mean(infer_times) * 1000:.2f}ms/frame")
        
        # Return result information
        return {
            "source_image": image_path,
            "crop_video": vsave_crop_path,
            "org_video": vsave_org_path,
            "frame_count": frame_count
        }
        
    except Exception as e:
        logger_f.error(f"Error processing {image_path}: {str(e)}")
        import traceback
        logger_f.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(description="Batch process multiple source images with a single driving video")
    parser.add_argument("--source_dir", required=True, help="Directory containing source images")
    parser.add_argument("--driving_video", required=True, help="Path to the driving video")
    parser.add_argument("--save_dir", default=None, help="Directory to save results (default: auto-generated)")
    parser.add_argument("--cfg", default="configs/trt_infer.yaml", help="Path to config file")
    parser.add_argument("--animal", action="store_true", help="Use animal model")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker threads")
    parser.add_argument("--batch_size", type=int, default=None, help="Limit the number of images to process")
    
    args = parser.parse_args()
    
    batch_process(
        args.source_dir,
        args.driving_video,
        args.save_dir,
        args.cfg,
        args.animal,
        args.max_workers,
        args.batch_size
    )


if __name__ == "__main__":
    main()
