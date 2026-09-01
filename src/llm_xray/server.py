from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import HOST, PORT
from .model import ModelManager
from .schemas import GenerationRequest, GenerationResponse


BASE_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = BASE_DIR / "web"


app = FastAPI(
    title="LLM-Xray",
    version="0.1.0",
)


model_manager = ModelManager()


@app.on_event("startup")
def startup_event():
    model_manager.load()


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.post("/generate", response_model=GenerationResponse)
def generate(request: GenerationRequest):
    return model_manager.generate(
        request.prompt,
        request.max_new_tokens,
    )


app.mount(
    "/web",
    StaticFiles(directory=WEB_DIR),
    name="web",
)


def main():
    import uvicorn

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
    )