from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, httpx

app = FastAPI()

origins = ["*"]  # restrict in production
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

MOVEBASE = "https://www.movebank.org/movebank/service/public/export/CsvStudy"

class DownloadReq(BaseModel):
    study_id: str
    username: str = None
    password: str = None

@app.post("/api/download-study")
async def download_study(payload: DownloadReq):
    if not payload.study_id:
        raise HTTPException(status_code=400, detail="study_id required")
    username = payload.username or os.getenv("MOVEBANK_USER")
    password = payload.password or os.getenv("MOVEBANK_PASS")
    if not username or not password:
        raise HTTPException(status_code=400, detail="missing credentials")
    url = f"{MOVEBASE}?study_id={payload.study_id}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(url, auth=(username, password))
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=str(e))
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="movebank error")
    async def streamer():
        for chunk in resp.iter_bytes():
            yield chunk
    return StreamingResponse(streamer(), media_type="text/csv")
