import os
import runpod
import json
import gc
import torch
import sys
import signal

from services.voice_conversion_service import process_voice_to_s3, infer
from fastapi import BackgroundTasks

def ensure_directory(path):
    """Create directory if it doesn't exist, preserving symlinks."""
    if os.path.islink(path):
        target = os.readlink(path)
        if not os.path.exists(target):
            print(f"Creating target directory for symlink {path} -> {target}")
            os.makedirs(target, exist_ok=True)
        else:
            print(f"Target directory for symlink {path} -> {target} already exists.")
            print(f"Contents of target directory {target}: {os.listdir(target)}")
    elif not os.path.exists(path):
        print(f"Creating directory: {path}")
        os.makedirs(path, exist_ok=True)
    else:
        print(f"Directory already exists: {path}")
        print(f"Contents of {path}: {os.listdir(path)}")


def error_response(message, status_code=400):
    return {
        "statusCode": status_code,
        "body": message
    }

def success_response(msg):
    return {
        "statusCode": 200,
        "body": msg
    }

# --- Argument Validation ---
def validate_job_input(job_input):
    """
    Validates the input parameters for voice conversion processing.
    Returns a dictionary of validated parameters or an error string.
    """
    required_params = ["audio_url", "model_name"]
    for param in required_params:
        if param not in job_input:
            return f"Missing required parameter: {param}"

    # Type and value checks
    if not isinstance(job_input["audio_url"], str) or not (job_input["audio_url"].startswith('http://') or job_input["audio_url"].startswith('https://')):
        return "Invalid 'audio_url': Must be a valid HTTP/HTTPS URL string."
    if not isinstance(job_input["model_name"], str):
        return "Invalid 'model_name': Must be a string."

    # Optional parameters with defaults or specific validation
    params = {
        "audio_url": job_input["audio_url"],
        "model_name": job_input["model_name"],
        "index_path": job_input.get("index_path"), # Optional
        "transpose": job_input.get("transpose", 0),
        "pitch_extraction_algorithm": job_input.get("pitch_extraction_algorithm", "rmvpe"),
        "search_feature_ratio": job_input.get("search_feature_ratio", 0.66),
        "device": job_input.get("device"), # Optional
        "is_half": job_input.get("is_half", False),
        "filter_radius": job_input.get("filter_radius", 3),
        "resample_output": job_input.get("resample_output", 0),
        "volume_envelope": job_input.get("volume_envelope", 1),
        "voiceless_protection": job_input.get("voiceless_protection", 0.33)
    }

    # Example of more specific validation if needed:
    if not isinstance(params["transpose"], int):
        return "Invalid 'transpose': Must be an integer."
    # ... add more validations as per rvc_api.py for other fields ...

    # Filter out None values for optional fields if the API expects them to be absent not null
    return {k: v for k, v in params.items() if v is not None}

# --- RunPod Handler ---
async def handler(job):  # MODIFIED: Made handler async
    """
    RunPod serverless handler.
    Receives a job, validates input, calls process_voice_to_s3 directly,
    and returns the S3 URL of the processed audio.
    """
    job_input = job.get('input')
    if not job_input:
        return error_response("No input provided.")

    print(f"Received job input: {json.dumps(job_input, indent=2)}")

    validated_params = validate_job_input(job_input)
    if isinstance(validated_params, str): # Validation failed
        return error_response(validated_params)

    print(f"Processing voice conversion with parameters: {json.dumps(validated_params, indent=2)}")

    try:
        # Check disk space before processing
        if not check_disk_space():
            return error_response("Insufficient disk space. Worker will be terminated.", status_code=507)

        # Clear CUDA memory before processing
        clear_cuda_memory()

        # Create a BackgroundTasks instance for cleanup
        background_tasks = BackgroundTasks()

        # Get the current working directory
        now_dir = os.getcwd()

        # Call process_voice_to_s3 directly
        response = await process_voice_to_s3(  # MODIFIED: Awaited the async function
            background_tasks=background_tasks,
            input_url=validated_params["audio_url"],
            model_name=validated_params["model_name"],
            index_path=validated_params.get("index_path"),
            transpose=validated_params["transpose"],
            pitch_extraction_algorithm=validated_params["pitch_extraction_algorithm"],
            search_feature_ratio=validated_params["search_feature_ratio"],
            device=validated_params.get("device"),
            is_half=validated_params["is_half"],
            filter_radius=validated_params["filter_radius"],
            resample_output=validated_params["resample_output"],
            volume_envelope=validated_params["volume_envelope"],
            voiceless_protection=validated_params["voiceless_protection"],
            infer_function=infer,
            now_dir=now_dir
        )

        # The response from process_voice_to_s3 is assumed to be a dictionary directly
        print(f"Voice conversion completed: {response}")

        # Execute any background tasks for cleanup
        await background_tasks()

        if response.get("status") == "success" and "audio_url" in response:
            # Clear memory after successful processing
            clear_cuda_memory()
            return success_response({"audio_url": response["audio_url"], "message": response.get("message")})
        else:
            error_msg = response.get("detail") or response.get("message") or "Unknown error from voice conversion"

            # Check if the error is fatal
            if is_fatal_error(error_msg):
                # Return error response first, then terminate worker
                error_resp = error_response(f"Voice conversion failed: {error_msg}. Worker terminating.", status_code=500)
                terminate_worker(error_msg)
                return error_resp

            return error_response(f"Voice conversion failed: {error_msg}", status_code=500)

    except Exception as e:
        error_str = str(e)
        print(f"An error occurred during voice conversion: {error_str}")

        # Check if this is a fatal error that requires worker termination
        if is_fatal_error(error_str):
            # Return error response first, then terminate worker
            error_resp = error_response(f"Fatal error occurred: {error_str}. Worker terminating.", status_code=500)
            terminate_worker(error_str)
            return error_resp

        # For non-fatal errors, clear memory and return normal error
        clear_cuda_memory()
        return error_response(f"An error occurred: {error_str}")

def clear_cuda_memory():
    """Clear CUDA memory to free up GPU resources."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            print("CUDA memory cleared successfully")
    except Exception as e:
        print(f"Error clearing CUDA memory: {e}")

def check_disk_space():
    """Check available disk space."""
    try:
        import shutil
        _, _, free = shutil.disk_usage("/")
        free_gb = free // (1024**3)
        print(f"Available disk space: {free_gb} GB")
        return free_gb > 1  # Return True if more than 1GB free
    except Exception as e:
        print(f"Error checking disk space: {e}")
        return False

def is_fatal_error(error_str):
    """Check if the error is fatal and requires worker termination."""
    error_str = str(error_str).lower()

    # CUDA out of memory errors
    if "cuda out of memory" in error_str or "out of memory" in error_str:
        return True

    # Disk space errors
    if "no space left on device" in error_str or "errno 28" in error_str:
        return True

    # Memory allocation errors
    if "cannot allocate memory" in error_str or "memory error" in error_str:
        return True

    return False

def terminate_worker(reason):
    """Terminate the current worker gracefully."""
    print(f"FATAL ERROR DETECTED: {reason}")
    print("Terminating worker to allow RunPod to spawn a new one...")

    # Clear any remaining CUDA memory
    clear_cuda_memory()

    # Send SIGTERM to self to trigger graceful shutdown
    os.kill(os.getpid(), signal.SIGTERM)

    # If SIGTERM doesn't work, force exit
    sys.exit(1)

# --- Main Execution ---
if __name__ == "__main__":
    ensure_directory("/runpod-volume/assets/weights")
    ensure_directory("/runpod-volume/logs")
    ensure_directory("/runpod-volume/opt")
    print("Directory check completed")
    print("Starting RunPod handler for RVC voice conversion...")
    runpod.serverless.start({"handler": handler})
