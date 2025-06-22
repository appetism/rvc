import os
import uuid
import boto3
import asyncio
import tempfile
import requests
import shutil
from typing import Union
from io import BytesIO
from botocore.client import Config as BotoConfig # type: ignore
from fastapi import BackgroundTasks
from concurrent.futures import ThreadPoolExecutor
from scipy.io import wavfile # type: ignore
from .model_manager_service import HuggingFaceModelManager

# Create executor for async operations
executor = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4))

# Load S3 configuration from environment variables
S3_ENABLED = os.environ.get("S3_ENABLED", "false").lower() == 'true'
s3_client = None
BUCKET_NAME = None

if S3_ENABLED:
    BUCKET_AREA = os.environ.get("BUCKET_AREA", None)
    BUCKET_ENDPOINT_URL = os.environ.get("BUCKET_ENDPOINT_URL", None)
    BUCKET_ACCESS_KEY_ID = os.environ.get("BUCKET_ACCESS_KEY_ID", None)
    BUCKET_SECRET_ACCESS_KEY = os.environ.get("BUCKET_SECRET_ACCESS_KEY", None)
    BUCKET_NAME = os.environ.get("BUCKET_NAME", None)
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
    if not BUCKET_NAME:
        missing_configs.append("BUCKET_NAME")

    if missing_configs:
        print(f"Error: The following S3 environment variables are not set, but S3_ENABLED is 'true': {', '.join(missing_configs)}. S3 functionality will be disabled.")
        s3_client = None
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
            print(f"  BUCKET_NAME: {BUCKET_NAME}")
            print(f"  BUCKET_AREA: {BUCKET_AREA}")
            print(f"  BUCKET_ENDPOINT_URL: {BUCKET_ENDPOINT_URL}")
            print(f"  BUCKET_ACCESS_KEY_ID: {BUCKET_ACCESS_KEY_ID[:4]}...{BUCKET_ACCESS_KEY_ID[-4:] if BUCKET_ACCESS_KEY_ID and len(BUCKET_ACCESS_KEY_ID) > 8 else '****'}")
            print(f"  S3_KEY_PREFIX: {S3_KEY_PREFIX if S3_KEY_PREFIX else 'None'}")

        except Exception as e:
            print(f"Error initializing S3 client: {e}. S3 functionality will be disabled.")
            s3_client = None
else:
    print("S3 upload functionality is not enabled (S3_ENABLED environment variable is not set to 'true').")
    S3_KEY_PREFIX = ""

async def process_voice_to_s3(
    background_tasks: BackgroundTasks, # type: ignore
    input_url: str,
    model_name: str,
    index_path: str,
    transpose: int,
    pitch_extraction_algorithm: str,
    search_feature_ratio: float,
    device: str,
    is_half: bool,
    filter_radius: int,
    resample_output: int,
    volume_envelope: float,
    voiceless_protection: float,
    infer_function,
    now_dir
):
    """
    Process voice conversion and upload the result to S3.

    Args:
        background_tasks: FastAPI BackgroundTasks object
        input_url: URL to the audio file to be converted
        model_name: Model name or Hugging Face repo ID
        index_path: Path to the index file
        transpose: Frequency key shifting value
        pitch_extraction_algorithm: Method for fundamental frequency extraction
        search_feature_ratio: Rate for indexing file usage
        device: Computation device
        is_half: Whether to use half precision
        filter_radius: Radius of the filter used in processing
        resample_output: Resample rate
        volume_envelope: Rate to mix in RMS normalization
        voiceless_protection: Protection factor to prevent clipping
        infer_function: The infer function to use for voice conversion
        hf_model_manager: Hugging Face model manager instance
        now_dir: Current working directory

    Returns:
        JSONResponse with the S3 URL
    """

    print("Processing voice conversion and uploading to S3...")
    # Check if S3 is configured and client is available
    if not S3_ENABLED or not s3_client:
        # Raise ValueError instead of HTTPException
        raise ValueError("S3 upload functionality is not enabled or configured correctly. Check server logs and S3_ENABLED, BUCKET_ENDPOINT_URL, etc. environment variables.")

    # Validate URL
    if not input_url.startswith('http://') and not input_url.startswith('https://'):
        # Raise ValueError instead of HTTPException
        raise ValueError("Invalid URL. Must start with http:// or https://")

    # Initialize HF Model Manager
    hf_model_manager = HuggingFaceModelManager()

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
                    # Raise ValueError instead of HTTPException
                    raise ValueError(
                        f"Failed to retrieve model {repo_id} from Hugging Face: {download_result['body']}"
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
                # Raise RuntimeError instead of HTTPException
                raise RuntimeError(f"Error downloading model {repo_id}: {str(e)}")

        # Update model_name and index_path for inference
        model_name = sanitized_repo_id

        # Only use local index path if user didn't provide one
        if original_index_path_param is None:
            index_path = index_path_local
            print(f"Using index path: {index_path}")

    # Generate a unique filename for S3
    file_extension = 'wav'
    file_uuid = str(uuid.uuid4())
    file_name = f"{file_uuid}.{file_extension}"
    s3_key = f"{S3_KEY_PREFIX}/{file_name}" if S3_KEY_PREFIX else file_name

    # Call the infer function with the renamed parameters
    try:
        wf = await asyncio.get_event_loop().run_in_executor(
            executor, infer_function, input_url, model_name, index_path, transpose, pitch_extraction_algorithm,
            search_feature_ratio, device, is_half, filter_radius, resample_output, volume_envelope,
            voiceless_protection
        )
    except Exception as e:
        # Re-raise the exception to be caught by rvc_api.py
        raise RuntimeError(str(e))

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
        return {
            "status": "success",
            "message": "Audio processed and uploaded successfully",
            "audio_url": presigned_url
        }
    except Exception as e:
        # Raise RuntimeError instead of HTTPException
        raise RuntimeError(f"Error uploading to S3: {str(e)}")
    finally:
        # Clean up the BytesIO object
        background_tasks.add_task(wf.close)

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
    """
    Perform voice conversion inference.

    Args:
        input: filepath, URL or raw bytes of audio input
        model_name: name of the model to use
        index_path: path to the index file
        f0up_key: frequency key shifting value
        f0method: method for fundamental frequency extraction
        index_rate: rate for indexing file usage
        device: computation device (cuda, cpu, etc.)
        is_half: whether to use half precision
        filter_radius: radius of the filter used in processing
        resample_sr: resample rate
        rms_mix_rate: rate to mix in RMS normalization
        protect: protection factor to prevent clipping

    Returns:
        BytesIO object containing the processed audio
    """
    from configs.config import Config
    from infer.modules.vc.modules import VC

    # Create model cache if it doesn't exist already
    if not hasattr(infer, 'model_cache'):
        infer.model_cache = {}

    model_name = model_name.replace(".pth", "")

    if index_path is None:
        index_path = os.path.join("logs", model_name, f"added_IVF1254_Flat_nprobe_1_{model_name}_v2.index")
        if not os.path.exists(index_path):
            raise ValueError(f"autinferred index_path {index_path} does not exist. Please provide a valid index_path")

    # Load or get cached model
    if model_name not in infer.model_cache:
        config = Config()
        config.device = device if device else config.device
        config.is_half = is_half if is_half else config.is_half
        vc = VC(config)
        vc.get_vc(f"{model_name}.pth")
        infer.model_cache[model_name] = vc

    vc = infer.model_cache[model_name]

    # If input is bytes, save to a temporary file first
    temp_file = None
    input_path = input

    try:
        # Check if input is a URL
        if isinstance(input, str) and (input.startswith('http://') or input.startswith('https://')):
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

async def process_voice_to_voice(
    background_tasks: BackgroundTasks, # type: ignore
    audio_bytes: bytes,
    model_name: str,
    index_path: str,
    transpose: int,
    pitch_extraction_algorithm: str,
    search_feature_ratio: float,
    device: str,
    is_half: bool,
    filter_radius: int,
    resample_output: int,
    volume_envelope: float,
    voiceless_protection: float
):
    """
    Process voice conversion from uploaded audio bytes.

    Args:
        background_tasks: FastAPI BackgroundTasks object
        audio_bytes: The raw audio bytes from the uploaded file
        model_name: Model name to use for voice conversion
        index_path: Path to the index file
        transpose: Frequency key shifting value
        pitch_extraction_algorithm: Method for fundamental frequency extraction
        search_feature_ratio: Rate for indexing file usage
        device: Computation device
        is_half: Whether to use half precision
        filter_radius: Radius of the filter used in processing
        resample_output: Resample rate
        volume_envelope: Rate to mix in RMS normalization
        voiceless_protection: Protection factor to prevent clipping

    Returns:
        BytesIO object containing the processed audio
    """
    try:
        # Call the infer function
        wf = await asyncio.get_event_loop().run_in_executor(
            executor, infer, audio_bytes, model_name, index_path, transpose, pitch_extraction_algorithm,
            search_feature_ratio, device, is_half, filter_radius, resample_output, volume_envelope,
            voiceless_protection
        )

        # Schedule the close operation for after the response is sent
        background_tasks.add_task(wf.close)

        return wf
    except Exception as e:
        # Re-raise the exception to be caught by rvc_api.py
        raise RuntimeError(str(e))

async def process_voice_url_to_voice(
    background_tasks: BackgroundTasks, # type: ignore
    input_url: str,
    model_name: str,
    index_path: str,
    transpose: int,
    pitch_extraction_algorithm: str,
    search_feature_ratio: float,
    device: str,
    is_half: bool,
    filter_radius: int,
    resample_output: int,
    volume_envelope: float,
    voiceless_protection: float
):
    """
    Process voice conversion from a URL to an audio file.

    Args:
        background_tasks: FastAPI BackgroundTasks object
        input_url: URL to the audio file to be converted
        model_name: Model name to use for voice conversion
        index_path: Path to the index file
        transpose: Frequency key shifting value
        pitch_extraction_algorithm: Method for fundamental frequency extraction
        search_feature_ratio: Rate for indexing file usage
        device: Computation device
        is_half: Whether to use half precision
        filter_radius: Radius of the filter used in processing
        resample_output: Resample rate
        volume_envelope: Rate to mix in RMS normalization
        voiceless_protection: Protection factor to prevent clipping

    Returns:
        BytesIO object containing the processed audio
    """
    # Validate URL
    if not input_url.startswith('http://') and not input_url.startswith('https://'):
        # Raise ValueError instead of HTTPException
        raise ValueError("Invalid URL. Must start with http:// or https://")

    try:
        # Call the infer function
        wf = await asyncio.get_event_loop().run_in_executor(
            executor, infer, input_url, model_name, index_path, transpose, pitch_extraction_algorithm,
            search_feature_ratio, device, is_half, filter_radius, resample_output, volume_envelope,
            voiceless_protection
        )

        # Schedule the close operation for after the response is sent
        background_tasks.add_task(wf.close)

        return wf
    except Exception as e:
        # Re-raise the exception to be caught by rvc_api.py
        raise RuntimeError(str(e))
