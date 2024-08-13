# remote_app.py
import torch
from fastapi import FastAPI

app = FastAPI()


@app.get("/is_cuda_available")
async def is_available():
    return {"torch.cuda.is_available": f"{torch.cuda.is_available()}"}
