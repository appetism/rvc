import runpod
import os
import time
import requests
import subprocess
import json

# Configuration
RVC_API_URL = "http://127.0.0.1:7866"
VOICE2VOICE_S3_ENDPOINT = f"{RVC_API_URL}/voice2voice_url_s3"
STATUS_ENDPOINT = f"{RVC_API_URL}/status"
RVC_API_SCRIPT = "rvc_api.py" # Assuming rvc_api.py is in the same directory or accessible in PATH

# --- Helper Functions ---
def error_response(message, status_code=400):
    return {
        "error": {
            "message": str(message),
            "status_code": status_code
        }
    }

def success_response(data):
    return {"output": data}

def start_rvc_api_server():
    """Starts the rvc_api.py FastAPI server as a background process."""
    print(f"Starting {RVC_API_SCRIPT} server...")
    # Ensure Python executable is used, especially in environments where 'python' might not be python3
    # Using sys.executable is generally safer if this script itself is run by a specific python interpreter
    # For now, assuming 'python' or 'python3' is in PATH and points to the correct interpreter for rvc_api.py
    try:
        process = subprocess.Popen(["python3", RVC_API_SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"{RVC_API_SCRIPT} server started with PID: {process.pid}")
        return process
    except FileNotFoundError:
        print(f"Error: Could not find 'python' executable or '{RVC_API_SCRIPT}'. Please ensure Python is installed and {RVC_API_SCRIPT} is in the correct path.")
        return None
    except Exception as e:
        print(f"Error starting {RVC_API_SCRIPT}: {e}")
        return None

def wait_for_api_ready(timeout=120):
    """Waits for the RVC API to be ready."""
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            print("Timeout waiting for RVC API to become ready.")
            return False
        try:
            response = requests.get(STATUS_ENDPOINT, timeout=5)
            if response.status_code == 200 and response.json().get("status") == "ok":
                print("RVC API is ready.")
                return True
        except requests.ConnectionError:
            print("RVC API not ready yet, waiting...")
        except Exception as e:
            print(f"Error checking API status: {e}")
        time.sleep(5)

# --- Argument Validation ---
def validate_job_input(job_input):
    """
    Validates the input parameters for the voice2voice_url_s3 endpoint.
    Returns a dictionary of validated parameters or an error string.
    """
    required_params = ["input_url", "model_name"]
    for param in required_params:
        if param not in job_input:
            return f"Missing required parameter: {param}"

    # Type and value checks (can be expanded based on rvc_api.py's Form(...) definitions)
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
def handler(job):
    """
    RunPod serverless handler.
    Receives a job, validates input, calls the voice2voice_url_s3 endpoint,
    and returns the S3 URL of the processed audio.
    """
    job_input = job.get('input')
    if not job_input:
        return error_response("No input provided.")

    print(f"Received job input: {json.dumps(job_input, indent=2)}")

    validated_params = validate_job_input(job_input)
    if isinstance(validated_params, str): # Validation failed
        return error_response(validated_params)

    print(f"Calling {VOICE2VOICE_S3_ENDPOINT} with parameters: {json.dumps(validated_params, indent=2)}")

    try:
        response = requests.post(VOICE2VOICE_S3_ENDPOINT, data=validated_params, timeout=180) # 3 minutes timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        api_response_data = response.json()
        print(f"Response from RVC API: {json.dumps(api_response_data, indent=2)}")

        if api_response_data.get("status") == "success" and "audio_url" in api_response_data:
            return success_response({"s3_url": api_response_data["audio_url"], "message": api_response_data.get("message")})
        else:
            error_msg = api_response_data.get("detail") or api_response_data.get("message") or "Unknown error from RVC API"
            return error_response(f"RVC API call failed: {error_msg}", status_code=response.status_code)

    except requests.exceptions.HTTPError as http_err:
        error_detail = "Unknown error"
        try:
            error_detail = http_err.response.json().get("detail", str(http_err))
        except json.JSONDecodeError:
            error_detail = str(http_err.response.text)
        print(f"HTTP error calling RVC API: {error_detail} (Status code: {http_err.response.status_code})")
        return error_response(f"HTTP error: {error_detail}", status_code=http_err.response.status_code)
    except requests.exceptions.RequestException as req_err:
        print(f"Request error calling RVC API: {req_err}")
        return error_response(f"Request error: {req_err}")
    except Exception as e:
        print(f"An unexpected error occurred in handler: {e}")
        return error_response(f"An unexpected error occurred: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting RunPod handler for RVC API...")

    # Ensure rvc_api.py and its dependencies are in the correct place relative to this script,
    # or adjust RVC_API_SCRIPT path / python execution command.
    # The Dockerfile for RunPod should handle setting up the environment and paths.

    api_server_process = start_rvc_api_server()

    if api_server_process is None:
        print("Failed to start RVC API server. Exiting.")
        exit(1) # Exit if server couldn't start

    if not wait_for_api_ready():
        print("RVC API server did not become ready. Terminating process and exiting.")
        api_server_process.terminate()
        api_server_process.wait()
        exit(1) # Exit if API doesn't become ready

    print("RVC API server is running. Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})

    # Optional: Clean up the API server process when RunPod worker stops (might not be reached in typical serverless)
    print("RunPod worker finished. Terminating RVC API server...")
    api_server_process.terminate()
    api_server_process.wait()
    print("RVC API server terminated.")

