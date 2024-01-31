import io
import itertools
import json
import math
import os

from PIL import Image
import numpy as np
import pandas as pd
from typing import Generator
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from imageio import v3 as iio
from .config import (
    Settings,
    DatasetsList,
    create_filepath_dict,
    set_performance_environment_variables
)


# initialize settings
settings = Settings()
filepath = create_filepath_dict(settings)
set_performance_environment_variables()



datasets = {}
for key, path in filepath.items():
    print(f"######## Load Dataset: {key} ##############")
    with open(path, "r") as f:
        cfg = json.load(f)
        if cfg["dataset"]["type"] == "folder":
            if cfg["dataset"]["format"] == "csv":
                df = pd.read_csv(os.path.join(settings.assets_path,cfg["dataset"]["path"]))
        elif cfg["dataset"]["type"] == "array":
            if cfg["dataset"]["format"] == "npy":
                df = np.load(os.path.join(settings.assets_path,cfg["dataset"]["path"]))
        datasets[key] = {"cfg":cfg["dataset"], "data": df}


app = FastAPI()

# TODO only for development
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# No empty response
@app.get("/")
async def redirect():
    """
    Empty requests redirect to the docs page.
    """
    return RedirectResponse(url="/docs")


# Healthcheck endpoint
@app.get("/ping")
async def pong():
    """
    Health check endpoint.

        return:

    "ping": pong! if alive and running.

    "env_test": EnvYes! if .env variables were read.
    """
    return {"ping": "pong!", "env_test": settings.env_test}


# return available datasets
@app.get("/datasets", response_model=DatasetsList)
async def list_datasets():
    return {"datasets": list(filepath.keys())}


@app.get(
    "/image/{dataset}/{idx}", 
        responses={
        200: {"content": {"image/jpeg": {}}},
        500: {"content": {"text/plain": {}}},
    },
    response_class=Response,
)
async def image_from_idx(
    dataset: str, idx: int,size: int = None
):
    """
    Returns a Image for the specified index.

        params:

    dataset: requested dataset shortcode/identifier, str. Can be obtained via /datasets.

    idx: requested index, int. Maximum value can be inquired at /datasets.

    size: window size, int.

        return:

    JPEG image

    """
    #TODO remove later
    if dataset.startswith("laion_262m"):
        dataset = "laion_262m"

    if dataset.startswith("laion_1m"):
        dataset = "laion_1m"

    dict_dataset = datasets[dataset]
    data = datasets[dataset]["data"]
    
    if dict_dataset["cfg"]["type"] == "folder":
        fpath = os.path.join(settings.assets_path,dict_dataset["cfg"]["image_path"],data.iloc[idx]["filename"])
        print(fpath)
        img = iio.imread(fpath)
        
    elif dict_dataset["cfg"]["type"] == "array":
        img = data[idx]
            
    if size is not None:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        #resize image
        img = img.resize((size,size),Image.BILINEAR)
    
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # transform to bytes
    with io.BytesIO() as buf:
        iio.imwrite(buf, img, format="JPEG", extension=".jpeg", plugin="pillow")
        image_bytes = buf.getvalue()
    
    # specify headers & return image bytes
    headers = {"Content-Disposition": 'inline; filename="image.jpeg"'}
    return Response(image_bytes, headers=headers, media_type="image/jpeg")
