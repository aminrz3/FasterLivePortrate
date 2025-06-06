#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Modal deployment for FasterLivePortrait batch processing

from dataclasses import dataclass
import sys
import glob
import ctypes
import cv2
import numpy as np
import base64
import modal
import logging
from io import BytesIO
from pathlib import Path
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig

# Configure logger
logger = logging.getLogger(__name__)

import os
import time
from enum import Enum
from typing import Optional, Tuple

from modal import App, build, Image, enter, Secret, asgi_app, gpu, method, web_endpoint
from pydantic import BaseModel, Field
from fastapi import File, Form, UploadFile, HTTPException

server_timeout = 1200  # 20 minutes
modal_gpu = "A10"
DIR = "/root/"
CHECKPOINTS_DIR = "/root/checkpoints"
TMP_OUTPUT_DIR = "/tmp/liveportrait_outputs"

# Required commands
clone_cmd = (
    f"git clone https://github.com/aminrz3/FasterLivePortrate.git /tmp/repo && "
    f"mv /tmp/repo/* /root/ && "
    f"rm -rf /tmp/repo"
)
# torch_install = "pip install torch==2.2.0 torchvision==0.17.0 torchaudio --index-url https://download.pytorch.org/whl/cu121"
requirements_cmd = f"cd {DIR} && pip install -r requirements.txt"
huggingface_download = f"cd {DIR} && huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints"
huggingface_download_trt = f"cd {DIR} && huggingface-cli download Aminrz3/FasterLivePortrate-trt-linux --local-dir ./checkpoints/liveportrait_onnx"

# Convert models to TRT format using the same TensorRT version as installed (8.6.1)
convert_to_trt = f"cd {DIR} && sh -x scripts/all_onnx2trt.sh"
# Define the Modal stub and image
model_image = (
   Image.debian_slim(python_version="3.10").pip_install(
        "huggingface_hub",
        "Pillow",
        "Requests",
        "transformers",
        "peft",
        "onnxruntime-gpu",
    ).apt_install([
        "ffmpeg",
        "git",
        "wget",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "build-essential",
    ]).run_commands([
        # Original commands
        clone_cmd,
        # torch_install,
        requirements_cmd,
        huggingface_download,
        huggingface_download_trt,
        f"sed -i '/import ctypes/d' {DIR}src/models/predictor.py",
        f"sed -i '1iimport ctypes' {DIR}src/models/predictor.py",
    ])
)
app = App(
    "faster-live-portrait-batch",
    image = model_image
)



# Volume to store model checkpoints and outputs and persist data between runs
model_volume = modal.Volume.from_name("liveportrait-outputs", create_if_missing=True)
MODEL_NAME = "warmshao/FasterLivePortrait"
OUTPUT_DIR = f"{DIR}results"


with model_image.imports():
    import time
    import torch
    import transformers
    from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf

# Mapping of image name to output path for tracking outputs

@app.cls(concurrency_limit=1,
    cpu=4,
    gpu=modal_gpu,
    memory=128, 
    volumes={OUTPUT_DIR: model_volume}, 
    timeout=server_timeout,
    container_idle_timeout=1000,
    allow_concurrent_inputs=1000,
)
class FasterLivePortrate:
  @classmethod
  def build(cls):
      # Create outputs directory if it doesn't exist
      if not os.path.exists(OUTPUT_DIR):
          os.makedirs(OUTPUT_DIR, exist_ok=True)
      logger.info(f"Created outputs directory: {OUTPUT_DIR}")
      
      

  def __init__(self):
      self.cfg = OmegaConf.load(os.path.join(DIR, "configs/onnx_infer.yaml"))
      self.pipe = None
  
  def load(self):
    self.pipe = FasterLivePortraitPipeline(cfg=self.cfg, is_animal=False)
  def process(self, source_image, driving_video, output_name=None, output_fps=None, output_size=None):
        """
        Process a single source image with a driving video
        
        Args:
            source_image (str): Path to source image
            driving_video (str): Path to driving video
            output_name (str, optional): Custom name for the output directory
            output_fps (float, optional): Custom output FPS for the generated GIF
            output_size (tuple, optional): Custom output size (width, height)
        
        Returns:
            dict: Result information including paths to output files
        """
        import os
        import cv2
        import time
        import numpy as np
        import subprocess
        
        # Use the volume output directory
        save_dir = OUTPUT_DIR
        
        # Create a unique output name if not provided
        if output_name is None:
            import uuid
            output_name = f"output_{uuid.uuid4().hex[:8]}"
            
        # Create a dedicated directory for this output
        output_subdir = os.path.join(save_dir, output_name)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Get base filenames
        image_basename = os.path.basename(source_image)
        image_name = os.path.splitext(image_basename)[0]
        
        # Get FFMPEG path based on platform
        import platform
        if platform.system().lower() == 'windows':
            FFMPEG = "third_party/ffmpeg-7.0.1-full_build/bin/ffmpeg.exe"
        else:
            FFMPEG = "ffmpeg"
        
        # We already have the pipeline initialized in __init__
        pipe = self.pipe
        
        # Prepare the source image
        ret = pipe.prepare_source(source_image, realtime=False)
        if not ret:
            print(f"No face detected in {source_image}! Skipping!")
            return None
        
        # Set up temporary MP4 output path (will be converted to GIF later)
        tmp_video_path = os.path.join(output_subdir, f"{image_name}_tmp.mp4")
        
        # Final GIF output path
        gif_output_path = os.path.join(output_subdir, f"{image_name}.gif")
        
        # Process the video
        print(f"Processing {source_image} with {driving_video}")
        
        try:
            # Open the video
            vcap = cv2.VideoCapture(driving_video)
            video_fps = vcap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                video_fps = 30.0  # Default to 30 fps if we can't get valid fps
            
            # Use output_fps if provided, otherwise use video fps
            fps = float(output_fps) if output_fps is not None else float(video_fps)
            
            # Use output_size if provided, otherwise use source image size
            if output_size is not None:
                w, h = output_size
            else:
                h, w = pipe.src_imgs[0].shape[:2]
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vout_org = cv2.VideoWriter(tmp_video_path, fourcc, fps, (int(w), int(h)))
            
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
                    print(f"No face in driving frame:{frame_ind}")
                    continue
                
                infer_times.append(time.time() - t0)
                
                # Write original size output
                if isinstance(out_org, np.ndarray):
                    out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
                else:  # If it's a tensor
                    out_org = cv2.cvtColor(out_org.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
                    
                vout_org.write(out_org)
            
            vcap.release()
            vout_org.release()
            
            avg_infer_time = np.mean(infer_times) if infer_times else 0
            print(f"Average inference time: {avg_infer_time:.4f}s")
            
            # Check if the driving video has audio
            from src.utils.utils import video_has_audio
            has_audio = video_has_audio(driving_video)
            
            # Convert temp MP4 to GIF
            if os.path.exists(tmp_video_path):
                # Use specified output_fps for GIF conversion
                
                # Convert MP4 to GIF using ffmpeg
                print(f"Converting to GIF: {gif_output_path}")
                
                # Build ffmpeg command
                cmd = [
                    FFMPEG,
                    "-y",  # Overwrite output file if it exists
                    "-i", tmp_video_path,  # Input file
                    "-vf", f"fps={output_fps},split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",  # Filter for GIF
                    "-loop", "0",  # Loop indefinitely
                    gif_output_path  # Output file
                ]
                
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Clean up temporary MP4 file
                os.remove(tmp_video_path)
                
                print(f"Generated GIF: {gif_output_path}")
                
                # Return result information
                return {
                    "source_image": source_image,
                    "driving_video": driving_video,
                    "output_gif": gif_output_path,
                    "frames": frame_ind,
                    "fps": fps,
                    "avg_infer_time": avg_infer_time
                }
                
        except Exception as e:
            print(f"Error processing {source_image}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
            


class Model:
    def __init__(self):
        self.model = FasterLivePortrate()

    @build()
    def on_build(self):
        self.model.build()

    @enter()
    def enter(self):
        self.model.load()

    @method()
    def process(self, source_image, driving_video, output_name=None, output_fps=None, output_size=None):
        return self.model.process(source_image, driving_video, output_name=output_name, output_fps=output_fps, output_size=output_size)    
 



@dataclass
class Params:
        source_image: str
        driving_video: str
        output_name: Optional[str] = None
        output_fps: Optional[float] = None
        output_size: Optional[Tuple[int, int]] = None

@dataclass
class DownloadParams:
        output_name: str



@app.function(
    concurrency_limit=1,
    cpu=4,
    gpu=modal_gpu,
    memory=128, 
    volumes={OUTPUT_DIR: model_volume}, 
    timeout=server_timeout,
    container_idle_timeout=1000,
    allow_concurrent_inputs=1000,
)
@web_endpoint(method="POST")
async def endpoint(
    source_image: UploadFile = File(...),
    driving_video: UploadFile = File(...),
    output_name: Optional[str] = Form(None),
    output_fps: Optional[float] = Form(None),
    output_size: Optional[str] = Form(None)
):   
    """
    Process a single source image with a driving video using multipart form data
    
    Args:
        source_image: Source image file upload
        driving_video: Driving video file upload
        output_name: Custom name for the output directory
        output_fps: Custom output FPS for the generated GIF
        output_size: Custom output size as "width,height" string
    """
    # Create temp directory for uploads if it doesn't exist
    os.makedirs(TMP_OUTPUT_DIR, exist_ok=True)
    
    # Process output_size if provided
    size_tuple = None
    if output_size:
        try:
            width, height = output_size.split(',')
            size_tuple = (int(width), int(height))
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400, 
                detail="Invalid output_size format. Expected 'width,height' (e.g. '512,512')"
            )
    
    # Save uploaded files to temporary locations
    source_image_path = os.path.join(TMP_OUTPUT_DIR, source_image.filename)
    driving_video_path = os.path.join(TMP_OUTPUT_DIR, driving_video.filename)
    
    # Save source image
    with open(source_image_path, "wb") as f:
        f.write(await source_image.read())
    
    # Save driving video
    with open(driving_video_path, "wb") as f:
        f.write(await driving_video.read())
    
    # Process with model
    model = Model()
    # Explicitly call build and enter to initialize the model
    model.on_build()
    model.enter()
    
    result = model.process(
        source_image=source_image_path,
        driving_video=driving_video_path,
        output_name=output_name,
        output_fps=output_fps,
        output_size=size_tuple
    )
    
    # Clean up temporary files
    try:
        os.remove(source_image_path)
        os.remove(driving_video_path)
    except Exception as e:
        logger.warning(f"Failed to clean up temporary files: {str(e)}")
    
    return result

@app.function()
@web_endpoint(method="GET")
def download_output(params:DownloadParams):
    """Download a processed output file from the volume"""
    import os
    from fastapi.responses import FileResponse
    from fastapi import HTTPException
    
    output_name = params.output_name
    output_dir = os.path.join(OUTPUT_DIR, output_name)
    
    # List all gif files in the output directory
    gif_files = [f for f in os.listdir(output_dir) if f.endswith('.gif')]
    
    if not gif_files:
        raise HTTPException(status_code=404, detail=f"No output found for {output_name}")
    
    # Return the first gif file found
    file_path = os.path.join(output_dir, gif_files[0])
    return FileResponse(file_path, media_type="image/gif", filename=gif_files[0])

@app.function()
@web_endpoint(method="GET")
def list_outputs():
    """List all available outputs in the volume"""
    import os
    
    if not os.path.exists(OUTPUT_DIR):
        return {"outputs": []}
    
    # List all subdirectories in the output directory
    outputs = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    
    # For each output, find the gif files
    output_files = []
    for output_name in outputs:
        output_dir = os.path.join(OUTPUT_DIR, output_name)
        gif_files = [f for f in os.listdir(output_dir) if f.endswith('.gif')]
        
        if gif_files:
            for gif in gif_files:
                output_files.append({
                    "output_name": output_name,
                    "filename": gif,
                    "download_url": f"/download_output?output_name={output_name}"
                })
    
    return {"outputs": output_files}


# def get_cmd_arg(cmd_list, flag):
#     try:
#         idx = cmd_list.index(flag)
#         return cmd_list[idx + 1]
#     except (ValueError, IndexError):
#         return None


# @app.function(
#     gpu="A10", 
#     image=model_image, 
#     timeout=3600,
#     volumes={VOLUME_MOUNT: model_volume}
# )
# def convertTRT():
#     import torch
#     import os
#     import subprocess
#     import sys
#     from pathlib import Path
    
#     # Verify CUDA is available
#     has_cuda = torch.cuda.is_available()
#     cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else 'Unknown'
#     print(f"CUDA available: {has_cuda}, CUDA version: {cuda_version}")
#     if not has_cuda:
#         return {"error": "CUDA is not available, cannot convert models to TensorRT"}
        
#     # Verify TensorRT is available
#     try:
#         import tensorrt as trt
#         trt_version = trt.__version__
#         print(f"TensorRT available, version: {trt_version}")
#     except ImportError as e:
#         return {"error": f"TensorRT is not available: {str(e)}"}
    
#     # Create checkpoints directory if it doesn't exist
#     checkpoints_dir = os.path.join(DIR, "checkpoints")
#     onnx_dir = os.path.join(checkpoints_dir, "liveportrait_onnx")
#     os.makedirs(onnx_dir, exist_ok=True)
    
#     # Verify the ONNX models exist - if not, we need to download them first
#     required_onnx_files = [
#         "warping_spade-fix.onnx",
#         "landmark.onnx",
#         "motion_extractor.onnx",
#         "retinaface_det_static.onnx",
#         "face_2dpose_106_static.onnx",
#         "appearance_feature_extractor.onnx",
#         "stitching.onnx",
#         "stitching_eye.onnx",
#         "stitching_lip.onnx"
#     ]
    
#     # Check if all ONNX files are present
#     missing_files = []
#     for onnx_file in required_onnx_files:
#         if not os.path.exists(os.path.join(onnx_dir, onnx_file)):
#             missing_files.append(onnx_file)
    
#     # If files are missing, run the huggingface download command
#     if missing_files:
#         print(f"The following ONNX files are missing: {missing_files}")
#         print("Downloading models from HuggingFace...")
#         try:
#             download_cmd = f"cd {DIR} && huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints"
#             process = subprocess.run(
#                 download_cmd,
#                 shell=True,
#                 capture_output=True,
#                 text=True,
#                 check=False
#             )
#             if process.returncode != 0:
#                 return {"error": f"Failed to download models: {process.stderr}"}
#         except Exception as e:
#             return {"error": f"Exception during model download: {str(e)}"}
    
#     # Make sure script exists
#     onnx2trt_path = os.path.join(DIR, "scripts", "onnx2trt.py")
#     if not os.path.exists(onnx2trt_path):
#         return {"error": f"Conversion script not found at {onnx2trt_path}"}
    
#     # List of conversion commands with proper input and output paths
#     conversion_tasks = [
#         # Warping model
#         [sys.executable, onnx2trt_path, "-o", os.path.join(onnx_dir, "warping_spade-fix.onnx"), "-e", os.path.join(onnx_dir, "warping_spade-fix.trt")],
#         # Landmark model
#         [sys.executable, onnx2trt_path, "-o", os.path.join(onnx_dir, "landmark.onnx"), "-e", os.path.join(onnx_dir, "landmark.trt")],
#         # Motion extractor model (fp32)
#         [sys.executable, onnx2trt_path, "-o", os.path.join(onnx_dir, "motion_extractor.onnx"), "-p", "fp32", "-e", os.path.join(onnx_dir, "motion_extractor.trt")],
#         # Face analysis models
#        [sys.executable, onnx2trt_path, "-o", os.path.join(onnx_dir, "retinaface_det_static.onnx"), "-e", os.path.join(onnx_dir, "retinaface_det_static.trt")],
#         [sys.executable, onnx2trt_path, "-o", os.path.join(onnx_dir, "face_2dpose_106_static.onnx"), "-e", os.path.join(onnx_dir, "face_2dpose_106_static.trt")],
#         # Appearance extractor model
#         [sys.executable, onnx2trt_path, "-o", os.path.join(onnx_dir, "appearance_feature_extractor.onnx"), "-e", os.path.join(onnx_dir, "appearance_feature_extractor.trt")],
#         # Stitching models
#         [sys.executable, onnx2trt_path, "-o", os.path.join(onnx_dir, "stitching.onnx"), "-e", os.path.join(onnx_dir, "stitching.trt")],
#         [sys.executable, onnx2trt_path, "-o", os.path.join(onnx_dir, "stitching_eye.onnx"), "-e", os.path.join(onnx_dir, "stitching_eye.trt")],
#         [sys.executable, onnx2trt_path, "-o", os.path.join(onnx_dir, "stitching_lip.onnx"), "-e", os.path.join(onnx_dir, "stitching_lip.trt")],
#     ]
    
#     results = []
#     for i, cmd in enumerate(conversion_tasks):
#         try:
#             print(f"Running conversion {i+1}/{len(conversion_tasks)}: {' '.join(cmd)}")
#             # Set current working directory to project root
#             process = subprocess.run(
#                 cmd,
#                 cwd=DIR,
#                 capture_output=True,
#                 text=True,
#                 check=False
#             )
#             if process.returncode == 0:
#                 model_name = Path(get_cmd_arg(cmd, "-o")).name
#                 # Get the engine path from the command (now at index 7 with -e param back)
#                 engine_path = get_cmd_arg(cmd, "-e")
                
#                 # Check if the engine file was actually created
#                 if os.path.exists(engine_path):
#                     file_size = os.path.getsize(engine_path)
#                     print(f"✅ Successfully converted {model_name} to TensorRT. File created at {engine_path} with size {file_size} bytes")
#                 else:
#                     print(f"⚠️ Conversion reported success but engine file not found at {engine_path}. Attempting manual creation...")
                    
#                     # Since TensorRT conversion succeeded but file wasn't created, let's try to do it manually
#                     try:
#                         # Get the corresponding onnx file path (at index 3)
#                         onnx_path = get_cmd_arg(cmd, "-o")
                        
#                         # Create a simplified TensorRT conversion function
#                         import tensorrt as trt
#                         def create_trt_engine(onnx_path, engine_path, precision="fp16"):
#                             logger = trt.Logger(trt.Logger.INFO)
#                             builder = trt.Builder(logger)
#                             network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
#                             parser = trt.OnnxParser(network, logger)
                            
#                             # Parse the ONNX model
#                             with open(onnx_path, 'rb') as f:
#                                 if not parser.parse(f.read()):
#                                     print(f"Failed to parse ONNX file: {onnx_path}")
#                                     for error in range(parser.num_errors):
#                                         print(f"ONNX Parser Error: {parser.get_error(error)}")
#                                     return False
                            
#                             # Create the config with appropriate settings
#                             config = builder.create_builder_config()
#                             config.max_workspace_size = 12 * (2 ** 30)  # 12 GB
                            
#                             if precision == "fp16":
#                                 config.set_flag(trt.BuilderFlag.FP16)
                                
#                             # Build and serialize the engine
#                             engine = builder.build_engine(network, config)
#                             if engine is None:
#                                 print("Failed to build engine")
#                                 return False
                                
#                             # Ensure the output directory exists
#                             os.makedirs(os.path.dirname(engine_path), exist_ok=True)
                            
#                             # Serialize the engine to a file
#                             with open(engine_path, 'wb') as f:
#                                 f.write(engine.serialize())
#                                 print(f"Engine serialized to {engine_path}")
                            
#                             return os.path.exists(engine_path)
                        
#                         # Get precision (fp16 is default)
#                         precision = get_cmd_arg(cmd, "-p")
#                         if precision is None:
#                             precision = "fp16"
                        
#                         # Try direct creation
#                         success = create_trt_engine(onnx_path, engine_path, precision)
#                         if success:
#                             print(f"✅ Successfully created engine manually at {engine_path}")
#                         else:
#                             print(f"❌ Failed to create engine manually")
                    
#                     except Exception as e:
#                         import traceback
#                         print(f"❌ Error during manual engine creation: {str(e)}")
#                         print(traceback.format_exc())
                
#                 results.append({
#                     "model": model_name,
#                     "success": True,
#                     "engine_path": engine_path
#                 })
#             else:
#                 model_name = Path(get_cmd_arg(cmd, "-o")).name
#                 results.append({
#                     "model": model_name,
#                     "success": False,
#                     "error": process.stderr
#                 })
#                 print(f"❌ Failed to convert {model_name}: {process.stderr}")
#         except Exception as e:
#             results.append({
#                 "model": f"Task {i+1}",
#                 "success": False,
#                 "error": str(e)
#             })
#             print(f"❌ Exception during conversion: {e}")
    
#     # Check how many models were successfully converted
#     success_count = sum(1 for r in results if r.get("success", False))
    
#     # Create a backup directory in the volume for the converted models
#     backup_dir = os.path.join(VOLUME_MOUNT)
#     os.makedirs(backup_dir, exist_ok=True)
    
#     # Copy all successfully converted engine files to the volume backup
#     backup_results = []
#     for result in results:
#         if result.get("success", False) and "engine_path" in result:
#             try:
#                 engine_path = result["engine_path"]
#                 engine_filename = os.path.basename(engine_path)
#                 backup_path = os.path.join(backup_dir, engine_filename)
                
#                 # Copy the file if it exists
#                 if os.path.exists(engine_path):
#                     import shutil
#                     shutil.copy2(engine_path, backup_path)
#                     backup_results.append({
#                         "model": result["model"],
#                         "backup_path": backup_path,
#                         "success": True
#                     })
#                     print(f"✅ Backed up {engine_filename} to volume at {backup_path}")
#                 else:
#                     backup_results.append({
#                         "model": result["model"],
#                         "success": False,
#                         "error": f"Engine file not found at {engine_path}"
#                     })
#                     print(f"❌ Failed to backup {engine_filename}: file not found")
#             except Exception as e:
#                 backup_results.append({
#                     "model": result["model"],
#                     "success": False,
#                     "error": str(e)
#                 })
#                 print(f"❌ Exception during backup: {e}")
    
#     # Copy the entire onnx directory to the volume as a complete backup
#     try:
#         full_backup_dir = os.path.join(VOLUME_MOUNT, "model_backup")
#         import shutil
#         if os.path.exists(onnx_dir):
#             print(f"Creating full backup of {onnx_dir} to {full_backup_dir}")
#             # Using rsync-like approach to copy the directory
#             if not os.path.exists(full_backup_dir):
#                 os.makedirs(full_backup_dir, exist_ok=True)
#             shutil.copytree(onnx_dir, os.path.join(full_backup_dir, "liveportrait_onnx"), dirs_exist_ok=True)
            
#             # Explicitly commit changes to ensure they're persisted
#             model_volume.commit()
#             print(f"✅ Full backup of onnx directory completed successfully and changes committed")
#     except Exception as e:
#         print(f"❌ Failed to create full backup of onnx directory: {e}")
    
#     return {
#         "has_cuda": has_cuda,
#         "conversion_results": results,
#         "backup_results": backup_results,
#         "summary": f"Successfully converted {success_count}/{len(conversion_tasks)} models"
#     }


# @app.local_entrypoint()
# def main():
#     # For Modal functions, we need to use .remote() to run them on Modal
#     # or directly execute their logic locally
#     #convertTRT.remote()
#     copy_trt_models_to_checkpoints.remote()
