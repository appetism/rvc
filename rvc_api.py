from typing import Union
import os
import sys
import fastapi
import uvicorn
import asyncio
import uuid
import boto3
import huggingface_hub
import json
import shutil
from botocore.client import Config as BotoConfig

from fastapi import FastAPI, HTTPException, UploadFile, BackgroundTasks, File, Form, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from io import BytesIO
from dotenv import load_dotenv
from scipy.io import wavfile
from configs.config import Config
from infer.modules.vc.modules import VC
from concurrent.futures import ThreadPoolExecutor

# don't like settings paths like this at all but due bad code its necessary
now_dir = os.getcwd()
sys.path.append(now_dir)
print(now_dir)

# load_dotenv is also very bad practice but necessary due bad code
load_dotenv()

app = FastAPI()

tags = [
    {
        "name": "voice2voice",
        "description": "Voice2Voice conversion using the pretrained model"
    },
    {
        "name": "models",
        "description": "Model management endpoints for downloading from Hugging Face"
    }
]

class ModelCache:
    """
    This class is used to cache the models so that they don't need to be loaded every time
    """
    def __init__(self):
        self.models = {}

    def load_model(self, model_name: str, device: str = None, is_half: bool = True):
        if model_name not in self.models:
            config = Config() # config_file_folder="A:/projects/Retrieval-based-Voice-Conversion-WebUI/configs/")
            config.device = device if device else config.device
            config.is_half = is_half if is_half else config.is_half
            vc = VC(config)
            vc.get_vc(f"{model_name}.pth")
            self.models[model_name] = vc
        return self.models[model_name]

executor = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4))  # Adjust based on your server's capability


def infer(
        input: Union[str, bytes], # filepath, URL or raw bytes
        model_name: str,
        index_path: str = None,
        f0up_key: int = 0,
        f0method: str = "crepe",
        index_rate: float = 0.66,
        device: str = None,
        is_half: bool = False,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 1,
        protect: float = 0.33,
        **kwargs
):
    model_name = model_name.replace(".pth", "")

    if index_path is None:
        index_path = os.path.join("logs", model_name, f"added_IVF1254_Flat_nprobe_1_{model_name}_v2.index")
        if not os.path.exists(index_path):
            raise ValueError(f"autinferred index_path {index_path} does not exist. Please provide a valid index_path")

    vc = model_cache.load_model(model_name, device=device, is_half=is_half)

    # If input is bytes, save to a temporary file first
    temp_file = None
    input_path = input

    try:
        # Check if input is a URL
        if isinstance(input, str) and (input.startswith('http://') or input.startswith('https://')):
            import requests
            import tempfile
            # Download the file from the URL
            response = requests.get(input, stream=True)
            response.raise_for_status()  # Raise an exception for bad responses

            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            # Write the content to the file
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.close()
            input_path = temp_file.name

        # Check if input is bytes
        elif isinstance(input, bytes):
            # Create a temporary file to save the audio bytes
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.write(input)
            temp_file.close()
            input_path = temp_file.name

        # Process the audio
        _, wav_opt = vc.vc_single(
            sid=0,
            input_audio_path=input_path,
            f0_up_key=f0up_key,
            f0_file=None,
            f0_method=f0method,
            file_index=index_path,
            file_index2=None,
            index_rate=index_rate,
            filter_radius=filter_radius,
            resample_sr=resample_sr,
            rms_mix_rate=rms_mix_rate,
            protect=protect
        )
    finally:
        # Clean up the temporary file if it was created
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except:
                pass

    # using virtual file to be able to return it as response
    wf = BytesIO()
    wavfile.write(wf, wav_opt[0], wav_opt[1])
    wf.seek(0)
    return wf


@app.post("/voice2voice", tags=["voice2voice"])
async def voice2voice(
        background_tasks: BackgroundTasks,
        input_file: UploadFile = File(...),
        model_name: str = Form(...),
        index_path: str = Form(None),
        transpose: int = Form(0),  # Renamed from f0up_key
        pitch_extraction_algorithm: str = Form("rmvpe"),  # Renamed from f0method
        search_feature_ratio: float = Form(0.66),  # Renamed from index_rate
        device: str = Form(None),
        is_half: bool = Form(False),
        filter_radius: int = Form(3),
        resample_output: int = Form(0),  # Renamed from resample_sr
        volume_envelope: float = Form(1),  # Renamed from rms_mix_rate
        voiceless_protection: float = Form(0.33)  # Renamed from protect
):
    """
    Endpoint to convert voices from one type to another using a specified model.

    Parameters:
    - input_file: the .wav file to be converted
    - model_name: the name of the model as found in the logs directory
    - index_path: optional path to an index file of the trained model
    - transpose: frequency key shifting, 0 (no shift) or any integer value
    - pitch_extraction_algorithm: method for fundamental frequency extraction (harvest, pm, dio, crepe, crepe-tiny, rmvpe)
    - search_feature_ratio: the rate at which to use the indexing file (between 0 and 1)
    - device: computation device (cuda, cpu, cuda:0, cuda:1, etc.)
    - is_half: whether to use half precision for the model
    - filter_radius: radius of the filter used in processing (between 0 and 7)
    - resample_output: resample rate, if needed (0 for no resampling)
    - volume_envelope: rate to mix in RMS normalization (between 0 and 1)
    - voiceless_protection: protection factor to prevent clipping (between 0 and 1)
    """
    audio_bytes = await input_file.read()

    # Ensure input file is closed after reading
    await input_file.close()

    # Call the infer function
    try:
        # Create a new BytesIO object for each request
        wf = await asyncio.get_event_loop().run_in_executor(
            executor, infer, audio_bytes, model_name, index_path, transpose, pitch_extraction_algorithm,
            search_feature_ratio, device, is_half, filter_radius, resample_output, volume_envelope,
            voiceless_protection
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Schedule the close operation for after the response is sent
    background_tasks.add_task(wf.close)

    # Return the response
    return StreamingResponse(wf, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=rvc.wav"})

@app.post("/voice2voice_url", tags=["voice2voice"])
async def voice2voice_url(
        background_tasks: BackgroundTasks,
        input_url: str = Form(...),
        model_name: str = Form(...),
        index_path: str = Form(None),
        transpose: int = Form(0),  # Renamed from f0up_key
        pitch_extraction_algorithm: str = Form("rmvpe"),  # Renamed from f0method
        search_feature_ratio: float = Form(0.66),  # Renamed from index_rate
        device: str = Form(None),
        is_half: bool = Form(False),
        filter_radius: int = Form(3),
        resample_output: int = Form(0),  # Renamed from resample_sr
        volume_envelope: float = Form(1),  # Renamed from rms_mix_rate
        voiceless_protection: float = Form(0.33)  # Renamed from protect
):
    """
    Endpoint to convert voices from a URL audio file using a specified model.

    Parameters:
    - input_url: URL to the .wav file to be converted
    - model_name: the name of the model as found in the logs directory
    - index_path: optional path to an index file of the trained model
    - transpose: frequency key shifting, 0 (no shift) or any integer value
    - pitch_extraction_algorithm: method for fundamental frequency extraction (harvest, pm, dio, crepe, crepe-tiny, rmvpe)
    - search_feature_ratio: the rate at which to use the indexing file (between 0 and 1)
    - device: computation device (cuda, cpu, cuda:0, cuda:1, etc.)
    - is_half: whether to use half precision for the model
    - filter_radius: radius of the filter used in processing (between 0 and 7)
    - resample_output: resample rate, if needed (0 for no resampling)
    - volume_envelope: rate to mix in RMS normalization (between 0 and 1)
    - voiceless_protection: protection factor to prevent clipping (between 0 and 1)
    """
    # Validate URL
    if not input_url.startswith('http://') and not input_url.startswith('https://'):
        raise HTTPException(status_code=400, detail="Invalid URL. Must start with http:// or https://")

    # Call the infer function
    try:
        wf = await asyncio.get_event_loop().run_in_executor(
            executor, infer, input_url, model_name, index_path, transpose, pitch_extraction_algorithm,
            search_feature_ratio, device, is_half, filter_radius, resample_output, volume_envelope,
            voiceless_protection
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Schedule the close operation for after the response is sent
    background_tasks.add_task(wf.close)

    # Return the response
    return StreamingResponse(wf, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=rvc.wav"})

@app.post("/voice2voice_url_s3", tags=["voice2voice"])
async def voice2voice_url_s3(
        background_tasks: BackgroundTasks,
        input_url: str = Form(...),
        model_name: str = Form(...), # This will be updated if it's an HF ID
        index_path: str = Form(None), # This will be updated if HF model and index_path was None
        transpose: int = Form(0),  # Renamed from f0up_key
        pitch_extraction_algorithm: str = Form("rmvpe"),  # Renamed from f0method
        search_feature_ratio: float = Form(0.66),  # Renamed from index_rate
        device: str = Form(None),
        is_half: bool = Form(False),
        filter_radius: int = Form(3),
        resample_output: int = Form(0),  # Renamed from resample_sr
        volume_envelope: float = Form(1),  # Renamed from rms_mix_rate
        voiceless_protection: float = Form(0.33)  # Renamed from protect
):
    """
    Endpoint to convert voices from a URL audio file using a specified model,
    then upload the result to S3 and return the S3 URL.

    Parameters:
    - input_url: URL to the .wav file to be converted
    - model_name: the name of the model as found in the logs directory, or a Hugging Face repo ID (e.g., username/model-name)
    - index_path: optional path to an index file of the trained model
    - transpose: frequency key shifting, 0 (no shift) or any integer value
    - pitch_extraction_algorithm: method for fundamental frequency extraction (harvest, pm, dio, crepe, crepe-tiny, rmvpe)
    - search_feature_ratio: the rate at which to use the indexing file (between 0 and 1)
    - device: computation device (cuda, cpu, cuda:0, cuda:1, etc.)
    - is_half: whether to use half precision for the model
    - filter_radius: radius of the filter used in processing (between 0 and 7)
    - resample_output: resample rate, if needed (0 for no resampling)
    - volume_envelope: rate to mix in RMS normalization (between 0 and 1)
    - voiceless_protection: protection factor to prevent clipping (between 0 and 1)
    """
    # Store original index_path to check if it was user-provided
    original_index_path_param = index_path

    if "/" in model_name:  # Check if model_name is a Hugging Face model ID
        repo_id = model_name  # Original model_name is the repo_id
        print(f"Hugging Face model ID detected: {repo_id}")

        # Create standardized filenames from repo_id
        sanitized_repo_id = repo_id.replace("/", "_").replace("-", "_")
        pth_filename = f"{sanitized_repo_id}.pth"
        index_filename = f"{sanitized_repo_id}.index"

        # Define expected local paths
        pth_path = os.path.join(now_dir, "assets", "weights", pth_filename)
        index_path_local = os.path.join(now_dir, "logs", index_filename)

        # Check if model already exists locally
        model_exists_locally = os.path.exists(pth_path) and os.path.exists(index_path_local)

        if model_exists_locally:
            print(f"Model {repo_id} already exists locally. Skipping download.")
        else:
            print(f"Downloading model {repo_id} from Hugging Face")
            try:
                # Download model from Hugging Face
                download_result = await asyncio.get_event_loop().run_in_executor(
                    executor, hf_model_manager.get_model, repo_id
                )

                if download_result["statusCode"] != 200:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to retrieve model {repo_id} from Hugging Face: {download_result['body']}"
                    )

                model_info = download_result["body"]

                # Ensure target directories exist
                os.makedirs(os.path.join(now_dir, "assets", "weights"), exist_ok=True)
                os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)

                # Copy model files if needed
                if os.path.exists(model_info["pth_path"]) and (
                   not os.path.exists(pth_path) or
                   os.path.getmtime(model_info["pth_path"]) > os.path.getmtime(pth_path)):
                    print(f"Copying model weights to {pth_path}")
                    shutil.copy2(model_info["pth_path"], pth_path)

                if os.path.exists(model_info["index_path"]) and (
                   not os.path.exists(index_path_local) or
                   os.path.getmtime(model_info["index_path"]) > os.path.getmtime(index_path_local)):
                    print(f"Copying model index to {index_path_local}")
                    shutil.copy2(model_info["index_path"], index_path_local)

            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error downloading model {repo_id}: {str(e)}"
                )

        # Update model_name and index_path for inference
        model_name = sanitized_repo_id

        # Only use local index path if user didn't provide one
        if original_index_path_param is None:
            index_path = index_path_local
            print(f"Using index path: {index_path}")

    # Check if S3 is configured and client is available
    if not S3_ENABLED or not s3_client:
        raise HTTPException(status_code=500, detail="S3 upload functionality is not enabled or configured correctly. Check server logs and S3_ENABLED, BUCKET_ENDPOINT_URL, etc. environment variables.")

    # Validate URL
    if not input_url.startswith('http://') and not input_url.startswith('https://'):
        raise HTTPException(status_code=400, detail="Invalid URL. Must start with http:// or https://")

    # Generate a unique filename for S3
    file_extension = 'wav'
    file_uuid = str(uuid.uuid4())
    file_name = f"{file_uuid}.{file_extension}"
    s3_key = f"{S3_KEY_PREFIX}/{file_name}" if S3_KEY_PREFIX else file_name

    # Call the infer function with the renamed parameters
    try:
        wf = await asyncio.get_event_loop().run_in_executor(
            executor, infer, input_url, model_name, index_path, transpose, pitch_extraction_algorithm,
            search_feature_ratio, device, is_half, filter_radius, resample_output, volume_envelope,
            voiceless_protection
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        # Upload the processed file to S3
        s3_client.upload_fileobj(wf, BUCKET_NAME, s3_key)

        # Generate a presigned URL for the uploaded file
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': BUCKET_NAME,
                'Key': s3_key
            },
            ExpiresIn=3600  # URL expires in 1 hour
        )

        # Return the S3 URL in JSON response
        return JSONResponse(
            content={
                "status": "success",
                "message": "Audio processed and uploaded successfully",
                "audio_url": presigned_url
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading to S3: {str(e)}")
    finally:
        # Clean up the BytesIO object
        background_tasks.add_task(wf.close)

@app.get("/status")
def status():
    return {"status": "ok"}

# Load S3 configuration from environment variables
S3_ENABLED = os.environ.get("S3_ENABLED", "false").lower() == 'true' # Treat as boolean
s3_client = None
BUCKET_NAME = None # Will be sourced from env var if S3 is enabled

if S3_ENABLED:
    BUCKET_AREA = os.environ.get("BUCKET_AREA", None)
    BUCKET_ENDPOINT_URL = os.environ.get("BUCKET_ENDPOINT_URL", None)
    BUCKET_ACCESS_KEY_ID = os.environ.get("BUCKET_ACCESS_KEY_ID", None)
    BUCKET_SECRET_ACCESS_KEY = os.environ.get("BUCKET_SECRET_ACCESS_KEY", None)
    BUCKET_NAME = os.environ.get("BUCKET_NAME", None) # Get BUCKET_NAME directly
    S3_KEY_PREFIX = os.environ.get("S3_KEY_PREFIX", "").strip('/')

    missing_configs = []
    if not BUCKET_ENDPOINT_URL:
        missing_configs.append("BUCKET_ENDPOINT_URL")
    if not BUCKET_ACCESS_KEY_ID:
        missing_configs.append("BUCKET_ACCESS_KEY_ID")
    if not BUCKET_SECRET_ACCESS_KEY:
        missing_configs.append("BUCKET_SECRET_ACCESS_KEY")
    if not BUCKET_AREA:
        missing_configs.append("BUCKET_AREA")
    if not BUCKET_NAME: # Add BUCKET_NAME to the check
        missing_configs.append("BUCKET_NAME")

    if missing_configs:
        print(f"Error: The following S3 environment variables are not set, but S3_ENABLED is 'true': {', '.join(missing_configs)}. S3 functionality will be disabled.")
        s3_client = None # Ensure client is None if config is incomplete
    else:
        try:
            # Initialize S3 client
            s3_client = boto3.client(
                's3',
                endpoint_url=BUCKET_ENDPOINT_URL,
                aws_access_key_id=BUCKET_ACCESS_KEY_ID,
                aws_secret_access_key=BUCKET_SECRET_ACCESS_KEY,
                config=BotoConfig(
                    signature_version='s3v4',
                    request_checksum_calculation='when_required',   # REQUIRED WITH GCS
                    response_checksum_validation='when_required',  # REQUIRED WITH GCS
                ),
                region_name=BUCKET_AREA
            )

            print("S3 Configuration loaded successfully.")
            print(f"  S3_ENABLED: {S3_ENABLED}")
            print(f"  BUCKET_NAME: {BUCKET_NAME}") # Log the directly sourced BUCKET_NAME

            print(f"  BUCKET_AREA: {BUCKET_AREA}")
            print(f"  BUCKET_ENDPOINT_URL: {BUCKET_ENDPOINT_URL}")
            print(f"  BUCKET_ACCESS_KEY_ID: {BUCKET_ACCESS_KEY_ID[:4]}...{BUCKET_ACCESS_KEY_ID[-4:] if BUCKET_ACCESS_KEY_ID and len(BUCKET_ACCESS_KEY_ID) > 8 else '****'}")
            print(f"  BUCKET_SECRET_ACCESS_KEY: {BUCKET_SECRET_ACCESS_KEY[:4]}...{BUCKET_SECRET_ACCESS_KEY[-4:] if BUCKET_SECRET_ACCESS_KEY and len(BUCKET_SECRET_ACCESS_KEY) > 8 else '****'}")
            print(f"  S3_KEY_PREFIX: {S3_KEY_PREFIX if S3_KEY_PREFIX else 'None'}")
            # BUCKET_SECRET_ACCESS_KEY is sensitive, so not printing even partial

        except Exception as e:
            print(f"Error initializing S3 client: {e}. S3 functionality will be disabled.")
            s3_client = None # Ensure client is None on error
else:
    print("S3 upload functionality is not enabled (S3_ENABLED environment variable is not set to 'true').")

# create model cache
model_cache = ModelCache()

class HuggingFaceModelManager:
    """
    Model manager for downloading and verifying models from Hugging Face Hub.
    """
    def __init__(self):
        hf_token = os.environ.get("HF_TOKEN", None)
        if hf_token is not None:
            huggingface_hub.login(token=hf_token)
            print("Logged in to Hugging Face Hub successfully")
        else:
            print("Warning: HF_TOKEN not found in environment variables. Anonymous access will be used, which may have rate limits.")

    def verify_config(self, config):
        """
        Verify that the model config meets the required format.
        """
        # Check existence of required fields
        if 'arch_type' not in config:
            return {"statusCode": 400, "body": "arch_type not found"}

        if 'arch_version' not in config:
            return {"statusCode": 400, "body": "arch_version not found"}

        if 'components' not in config:
            return {"statusCode": 400, "body": "components not found"}

        # Check field types
        if not isinstance(config['arch_type'], str):
            return {"statusCode": 400, "body": "arch_type must be str"}

        if not isinstance(config['arch_version'], str):
            return {"statusCode": 400, "body": "arch_version must be str"}

        if not isinstance(config['components'], dict):
            return {"statusCode": 400, "body": "components must be dict"}

        # Validate architecture type
        if config['arch_type'] != "rvc":
            return {"statusCode": 400, "body": "arch_type must be 'rvc'"}

        # Check required components
        if "pth" not in config['components']:
            return {"statusCode": 400, "body": "components['pth'] not found"}

        if "index" not in config['components']:
            return {"statusCode": 400, "body": "components['index'] not found"}

        return None

    def get_model(self, model_name):
        """
        Download a model and its components from Hugging Face Hub.

        Args:
            model_name (str): The Hugging Face repository ID (e.g., username/model-name)

        Returns:
            dict: Response with status code and body containing model information
        """
        try:
            print(f"Downloading config for model: {model_name}")
            config_path = huggingface_hub.hf_hub_download(
                repo_id=model_name,
                filename="config.json",
                repo_type="model")

            with open(config_path, "r") as f:
                config = json.load(f)

            check = self.verify_config(config)
            if check is not None:
                return check

            print(f"Downloading model file: {config['components']['pth']}")
            pth_path = huggingface_hub.hf_hub_download(
                repo_id=model_name,
                filename=config['components']['pth'],
                repo_type="model")

            print(f"Downloading index file: {config['components']['index']}")
            index_path = huggingface_hub.hf_hub_download(
                repo_id=model_name,
                filename=config['components']['index'],
                repo_type="model")

            print("Model download complete")
            return {
                "statusCode": 200,
                "body": {
                    "config": config,
                    "pth_path": pth_path,
                    "index_path": index_path
                }
            }
        except Exception as e:
            error_message = f"Error downloading model: {str(e)}"
            print(error_message)
            return {"statusCode": 400, "body": error_message}

# Initialize HF Model Manager
hf_model_manager = HuggingFaceModelManager()

@app.get("/list_models", tags=["models"])
async def list_models():
    """
    List all available models in the weights directory.

    Returns:
    - JSON with list of model names
    """
    try:
        weights_dir = os.path.join(now_dir, "assets", "weights")

        if not os.path.exists(weights_dir):
            return JSONResponse(
                content={
                    "status": "success",
                    "models": []
                }
            )

        models = [
            os.path.splitext(file)[0]
            for file in os.listdir(weights_dir)
            if file.endswith(".pth")
        ]

        return JSONResponse(
            content={
                "status": "success",
                "models": models
            }
        )
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "message": f"Error listing models: {str(e)}"
            },
            status_code=500
        )

# start uvicorn server
uvicorn.run(app, host="0.0.0.0", port=7866)
