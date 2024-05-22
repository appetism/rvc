from typing import Union
import os
import sys
import fastapi
import uvicorn
import asyncio

from functools import partial
from fastapi import FastAPI, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse
from io import BytesIO
from dotenv import load_dotenv
from scipy.io import wavfile
from configs.config import Config
from infer.modules.vc.modules import VC
from concurrent.futures import ProcessPoolExecutor

# don't like settings paths like this at all but due bad code its necessary
now_dir = os.getcwd()
sys.path.append(now_dir)
print(now_dir)

# load_dotenv is also very bad practice but necessary due bad code
load_dotenv()

# Initialize the executor
executor = ProcessPoolExecutor()

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
    return wf

@app.post("/voice2voice")
async def voice2voice(
  input_file: UploadFile,
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
    audio_bytes = await input_file.read()

    # Ensure the input file is closed after reading
    await input_file.close()

    kwargs = locals()
    kwargs["input"] = audio_bytes
    del kwargs["input_file"]

    # Offload the heavy CPU-bound task to the executor
    """ future = executor.submit(
        infer,
        **kwargs
    ) """

    try:
        loop = asyncio.get_event_loop()
        partial_infer = partial(infer,
                                input=audio_bytes,
                                model_name=model_name,
                                index_path=index_path,
                                f0up_key=f0up_key,
                                f0method=f0method,
                                index_rate=index_rate,
                                device=device,
                                is_half=is_half,
                                filter_radius=filter_radius,
                                resample_sr=resample_sr,
                                rms_mix_rate=rms_mix_rate,
                                protect=protect
                            )

        result = await loop.run_in_executor(executor, partial_infer)
        #result = await asyncio.wrap_future(future)
        result.seek(0)  # Rewind the BytesIO object
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(result, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=rvc.wav"})

@app.get("/status")
def status():
    return {"status": "ok"}

# create model cache
model_cache = ModelCache()
# start uvicorn server
uvicorn.run(app, host="0.0.0.0", port=7866)

