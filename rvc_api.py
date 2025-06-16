import os
import sys
import uvicorn  # type: ignore
from fastapi import FastAPI, HTTPException, UploadFile, BackgroundTasks, File, Form  # type: ignore
from fastapi.responses import StreamingResponse, JSONResponse  # type: ignore
from dotenv import load_dotenv  # type: ignore
from scipy.io import wavfile  # type: ignore
from services.voice_conversion_service import infer

# don't like settings paths like this at all but due bad code its necessary
now_dir = os.getcwd()
sys.path.append(now_dir)
print(now_dir)

# load_dotenv is also very bad practice but necessary due bad code
load_dotenv()

app = FastAPI()


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

    # Use the service function to process the request
    try:
        from services.voice_conversion_service import process_voice_to_voice

        wf = await process_voice_to_voice(
            background_tasks=background_tasks,
            audio_bytes=audio_bytes,
            model_name=model_name,
            index_path=index_path,
            transpose=transpose,
            pitch_extraction_algorithm=pitch_extraction_algorithm,
            search_feature_ratio=search_feature_ratio,
            device=device,
            is_half=is_half,
            filter_radius=filter_radius,
            resample_output=resample_output,
            volume_envelope=volume_envelope,
            voiceless_protection=voiceless_protection
        )

        # Return the response
        return StreamingResponse(wf, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=rvc.wav"})
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


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
    # Use the service function to process the request
    try:
        from services.voice_conversion_service import process_voice_url_to_voice

        wf = await process_voice_url_to_voice(
            background_tasks=background_tasks,
            input_url=input_url,
            model_name=model_name,
            index_path=index_path,
            transpose=transpose,
            pitch_extraction_algorithm=pitch_extraction_algorithm,
            search_feature_ratio=search_feature_ratio,
            device=device,
            is_half=is_half,
            filter_radius=filter_radius,
            resample_output=resample_output,
            volume_envelope=volume_envelope,
            voiceless_protection=voiceless_protection
        )

        # Return the response
        return StreamingResponse(wf, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=rvc.wav"})
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/voice2voice_url_s3", tags=["voice2voice"])
async def voice2voice_url_s3(
    background_tasks: BackgroundTasks,
    input_url: str = Form(...),
    model_name: str = Form(...),  # This will be updated if it's an HF ID
    index_path: str = Form(None),  # This will be updated if HF model and index_path was None
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
    from services.voice_conversion_service import process_voice_to_s3

    # Use the service function to process the request
    try:
        response = await process_voice_to_s3(
            background_tasks=background_tasks,
            input_url=input_url,
            model_name=model_name,
            index_path=index_path,
            transpose=transpose,
            pitch_extraction_algorithm=pitch_extraction_algorithm,
            search_feature_ratio=search_feature_ratio,
            device=device,
            is_half=is_half,
            filter_radius=filter_radius,
            resample_output=resample_output,
            volume_envelope=volume_envelope,
            voiceless_protection=voiceless_protection,
            infer_function=infer,
            now_dir=now_dir
        )
        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/status")
def status():
    return {"status": "ok"}


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
