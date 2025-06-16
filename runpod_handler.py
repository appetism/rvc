import os
import runpod
import json

from services.voice_conversion_service import process_voice_to_s3, infer
from fastapi import BackgroundTasks

def ensure_directory(path):
    """Create directory if it doesn't exist, preserving symlinks."""
    if os.path.islink(path):
        target = os.readlink(path)
        if not os.path.exists(target):
            print(f"Creating target directory for symlink {path} -> {target}")
            os.makedirs(target, exist_ok=True)
    elif not os.path.exists(path):
        print(f"Creating directory: {path}")
        os.makedirs(path, exist_ok=True)
    else:
        print(f"Directory already exists: {path}")


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
    required_params = ["input_url", "model_name"]
    for param in required_params:
        if param not in job_input:
            return f"Missing required parameter: {param}"

    # Type and value checks
    if not isinstance(job_input["input_url"], str) or not (job_input["input_url"].startswith('http://') or job_input["input_url"].startswith('https://')):
        return "Invalid 'input_url': Must be a valid HTTP/HTTPS URL string."
    if not isinstance(job_input["model_name"], str):
        return "Invalid 'model_name': Must be a string."

    # Optional parameters with defaults or specific validation
    params = {
        "input_url": job_input["input_url"],
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
        # Create a BackgroundTasks instance for cleanup
        background_tasks = BackgroundTasks()

        # Get the current working directory
        now_dir = os.getcwd()

        # Call process_voice_to_s3 directly
        response = await process_voice_to_s3(  # MODIFIED: Awaited the async function
            background_tasks=background_tasks,
            input_url=validated_params["input_url"],
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
            return success_response({"s3_url": response["audio_url"], "message": response.get("message")})
        else:
            error_msg = response.get("detail") or response.get("message") or "Unknown error from voice conversion"
            return error_response(f"Voice conversion failed: {error_msg}", status_code=500)

    except Exception as e:
        print(f"An error occurred during voice conversion: {e}")
        return error_response(f"An error occurred: {str(e)}")

# --- Main Execution ---
if __name__ == "__main__":
    ensure_directory("/runpod-volume/assets/weights")
    ensure_directory("/runpod-volume/opt")
    print("Directory check completed")
    print("Starting RunPod handler for RVC voice conversion...")
    runpod.serverless.start({"handler": handler})
