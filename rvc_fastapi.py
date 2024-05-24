from typing import Union
import os
import sys
import fastapi
import uvicorn
import asyncio

from fastapi import FastAPI, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse
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
        input: Union[str, bytes], # filepath or raw bytes
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

    _, wav_opt = vc.vc_single(
        sid=0,
        input_audio_path=input,
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

    # using virtual file to be able to return it as response
    wf = BytesIO()
    wavfile.write(wf, wav_opt[0], wav_opt[1])
    wf.seek(0)
    return wf


@app.post("/voice2voice", tags=["voice2voice"])
async def voice2voice(
    input_file: UploadFile,
    background_tasks: BackgroundTasks,
    model_name: str,
    index_path: str = None,
    f0up_key: int = 0,
    f0method: str = "rmvpe",
    index_rate: float = 0.66,
    device: str = None,
    is_half: bool = False,
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 1,
    protect: float = 0.33
):
    """
    Endpoint to convert voices from one type to another using a specified model.

    Parameters:
    - input_file: the .wav file to be converted
    - model_name: the name of the model as found in the logs directory
    - index_path: optional path to an index file of the trained model
    - f0up_key: frequency key shifting, 0 (no shift) or 1
    - f0method: method for fundamental frequency extraction (harvest, pm, crepe, rmvpe)
    - index_rate: the rate at which to use the indexing file
    - device: computation device (cuda, cpu, cuda:0, cuda:1, etc.)
    - is_half: whether to use half precision for the model
    - filter_radius: radius of the filter used in processing
    - resample_sr: resample rate, if needed (0 for no resampling)
    - rms_mix_rate: rate to mix in RMS normalization
    - protect: protection factor to prevent clipping
    """
    audio_bytes = await input_file.read()

    # Ensure input file is closed after reading
    await input_file.close()

    # Prepare kwargs for the infer function
    kwargs = locals()
    kwargs["input"] = audio_bytes
    del kwargs["input_file"]

    # Call the infer function
    try:
        # Create a new BytesIO object for each request
#         wf = infer(**kwargs)
         wf = await asyncio.get_event_loop().run_in_executor(
                     executor, infer, audio_bytes, model_name, index_path, f0up_key, f0method, index_rate, device, is_half, filter_radius, resample_sr, rms_mix_rate, protect
                 )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Schedule the close operation for after the response is sent
    background_tasks.add_task(wf.close)

    # Return the response
    return StreamingResponse(wf, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=rvc.wav"})

@app.post("/voice2voice_local", tags=["voice2voice"])
async def voice2voice_local(
    input_path: str,
    model_name: str,
    index_path: str = None,
    opt_path: str = None,
    f0up_key: int = 0,
    f0method: str = "rmvpe",
    index_rate: float = 0.66,
    device: str = None,
    is_half: bool = False,
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 1,
    protect: float = 0.33
):
    """
    :param input_path: the .wav file to be converted
    :param model_name: the name of the model which was previously trained in the gui like the name in logs folder
    :param index_path: the index file of the previously trained model if none then use default dir logs by rvc.
    :param opt_path: if not None then save the result to this path
    :param f0up_key: 0 or 1
    :param f0method: harvest, pm, crepe or rmvpe
    :param index_rate: 0.66
    :param device: if none then use default by rvc. cuda or cpu or specific cuda device "cuda:0", "cuda:1"
    :param is_half: False or True
    :param filter_radius: 3
    :param resample_sr: 0
    :param rms_mix_rate: 1
    :param protect: 0.33
    """
    kwargs = locals()

    # call the infer function
    try:
        wf = infer(**kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # write to file if opt_path is provided
    if opt_path is not None:
        out_name = os.path.basename(opt_path)
        with open(opt_path, "wb") as f:
            f.write(wf.getbuffer())
    else:
        out_name = os.path.basename("input_path")

    return StreamingResponse(wf, media_type="audio/wav", headers={"Content-Disposition": f"attachment; filename={out_name}"})

@app.get("/status")
def status():
    return {"status": "ok"}

# create model cache
model_cache = ModelCache()
# start uvicorn server
uvicorn.run(app, host="0.0.0.0", port=7866)

