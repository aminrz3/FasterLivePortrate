import modal
from modal import App, build, Image, enter, Secret, asgi_app, gpu, method, fastapi_endpoint
from fastapi import File, Form, UploadFile, HTTPException

server_timeout = 1200  # 20 minutes
modal_gpu = "A10"
DIR = "/root/"
torch_install = "pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118"
clone_cmd = (
    f"git clone https://github.com/aminrz3/FasterLivePortrate.git /tmp/repo && "
    f"mv /tmp/repo/* /root/ && "
    f"rm -rf /tmp/repo"
)

requirements_cmd = f"cd {DIR} && pip install -r requirements.txt"

huggingface_download = f"cd {DIR} && huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints"

cuda_version = "11.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
model_image = (
   Image.from_registry(f"nvcr.io/nvidia/tensorrt:23.11-py3", add_python="3.10").pip_install(
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
        "clang",
    ]).pip_install("tensorrt").run_commands([
        # Original commands
        clone_cmd,
        torch_install,
        requirements_cmd,
        f"sed -i '/import ctypes/d' {DIR}src/models/predictor.py",
        f"sed -i '1iimport ctypes' {DIR}src/models/predictor.py",
        "mkdir /root/checkpoints"
    ])
)

app = App(
    "faster-live-portrait",
    image = model_image
)



# Volume to store model checkpoints and outputs and persist data between runs
model_volume = modal.Volume.from_name("liveportrait-outputs", create_if_missing=True)
model_volume_checkpoints = modal.Volume.from_name("liveportrait-checkpoints", create_if_missing=True)
OUTPUT_DIR = f"{DIR}results"
CHECKPOINTS_DIR = f"{DIR}checkpoints"
TMP_UPLOAD_DIR = "/tmp/liveportrait_upload"

@app.function(
    concurrency_limit=1,
    cpu=8,
    gpu=modal_gpu, 
    memory=16384,
    volumes={OUTPUT_DIR: model_volume, CHECKPOINTS_DIR: model_volume_checkpoints}, 
    timeout=server_timeout,
    container_idle_timeout=1000,
    allow_concurrent_inputs=1000,
)
@fastapi_endpoint(method="POST")
async def run(
    source_image: UploadFile = File(...),
    driving_video: UploadFile = File(...),
):

    import os
    import subprocess
    
    # Create output directory if it doesn't exist
    os.makedirs(TMP_UPLOAD_DIR, exist_ok=True)

    source_image_path = os.path.join(TMP_UPLOAD_DIR, source_image.filename)
    driving_video_path = os.path.join(TMP_UPLOAD_DIR, driving_video.filename)
    
    # Save source image
    with open(source_image_path, "wb") as f:
        f.write(await source_image.read())
    
    # Save and validate driving video
    with open(driving_video_path, "wb") as f:
        f.write(await driving_video.read())


    # Run with more debugging info and added error handling
    try:
        cmd = f"cd {DIR} && python run.py --src_image  {source_image_path} --dri_video {driving_video_path} --cfg configs/trt_infer.yaml"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
    except Exception as e:
        print(f"Exception during processing: {str(e)}")
        result = {"error": str(e), "success": False}
    finally:
        # Clean up temporary files
        try:
            os.remove(source_image_path)
            os.remove(driving_video_path)
        except Exception as e:
            print(f"Failed to clean up temporary files: {str(e)}")
    
    return result

@app.function(
    gpu=modal_gpu, 
    image=model_image, 
    timeout=3600,
    volumes={CHECKPOINTS_DIR: model_volume_checkpoints}
)
def convertTRT():
    import subprocess
    import os
    
    # Use absolute path for checkpoints
    checkpoint_path = os.path.join(DIR, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    print(f"Starting model download to {checkpoint_path}")
    try:
        # Use huggingface-cli with absolute path
        cmd = f"cd {DIR} && huggingface-cli download warmshao/FasterLivePortrait --local-dir {checkpoint_path}"
        print(f"Running command: {cmd}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"Download stdout: {result.stdout}")
        print(f"Download stderr: {result.stderr}")
        
        if result.returncode != 0:
            print(f"Warning: Download exited with code {result.returncode}")
            return {"error": result.stderr, "success": False, "returncode": result.returncode}
            
        print(f"Model download completed successfully")
        
        # Uncommenting the TensorRT conversion
        print("Starting TensorRT conversion")
        cmd = f"cd {DIR} && sh scripts/all_onnx2trt.sh"
        conversion_result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"Conversion stdout: {conversion_result.stdout}")
        print(f"Conversion stderr: {conversion_result.stderr}")
        
        if conversion_result.returncode != 0:
            print(f"Warning: TensorRT conversion exited with code {conversion_result.returncode}")
        
        # List the files in the checkpoints directory to debug
        try:
            files = os.listdir(checkpoint_path)
            print(f"Files in {checkpoint_path}: {files}")
        except Exception as e:
            print(f"Error listing checkpoint files: {str(e)}")
        
        return {"download_success": True,"conversion_success": conversion_result.returncode == 0}
        
    except Exception as e:
        print(f"Exception during processing: {str(e)}")
        return {"error": str(e), "success": False}


@app.local_entrypoint()
def main():
    convert_result = convertTRT.remote()
    print(f"Conversion result: {convert_result}")