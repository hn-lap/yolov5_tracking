import pickle
import re
import warnings
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

warnings.filterwarnings("ignore")


app = FastAPI()


@app.get("/isAlive")
def is_alive():
    return {"responses": "Alive"}


@app.post("/predict")
def tracking():
    pass
