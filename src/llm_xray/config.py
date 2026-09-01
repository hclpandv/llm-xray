import os

MODEL_NAME = os.getenv(
    "LLM_XRAY_MODEL",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
)

HOST = os.getenv("LLM_XRAY_HOST", "127.0.0.1")
PORT = int(os.getenv("LLM_XRAY_PORT", "8000"))