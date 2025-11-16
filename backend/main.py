from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MOVEBASE = "https://www.movebank.org/movebank/service/public/export/CsvStudy"

class DownloadReq(BaseModel):
    study_id: str
    username: str | None = None
    password: str | None = None

@app.post("/api/download-study")
async def download_study(payload: DownloadReq):
    if not payload.study_id:
        raise HTTPException(400, "study_id required")

    username = payload.username or os.getenv("MOVEBANK_USER")
    password = payload.password or os.getenv("MOVEBANK_PASS")

    if not username or not password:
        raise HTTPException(400, "missing credentials")

    url = f"{MOVEBASE}?study_id={payload.study_id}"

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(url, auth=(username, password))

    if resp.status_code != 200:
        raise HTTPException(resp.status_code, "movebank error")

    async def stream():
        for chunk in resp.iter_bytes():
            yield chunk

    return StreamingResponse(stream(), media_type="text/csv")
