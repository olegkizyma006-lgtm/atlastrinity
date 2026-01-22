import os

import httpx
from mcp.server import FastMCP

from src.brain.config_loader import config
from src.brain.voice.stt import WhisperSTT

# Load config from YAML
try:
    from .config_loader import get_config_value

    STT_MODEL = get_config_value("whisper", "model", "large-v3")
    STT_LANGUAGE = get_config_value("whisper", "language", "uk")
except Exception:
    STT_MODEL = config.get("voice.stt.model", "large-v3")
    STT_LANGUAGE = config.get("voice.stt.language", "uk")

# Initialize FastMCP server
server = FastMCP("whisper-stt")

# Shared STT instance (lazy-loaded fallback)
_local_stt = None


async def get_local_stt():
    global _local_stt
    if _local_stt is None:
        model_name = config.get("voice.stt.model", "large-v3")
        _local_stt = WhisperSTT(model_name=model_name)
    return _local_stt


async def _transcribe_via_brain(audio_path: str):
    """Спроба відправити аудіо на основний сервер AtlasBrain для економії пам'яті"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Перевіряємо чи живий сервер
            health = await client.get("http://127.0.0.1:8000/api/health")
            if health.status_code == 200:
                with open(audio_path, "rb") as f:
                    files = {"audio": (os.path.basename(audio_path), f, "audio/wav")}
                    response = await client.post("http://127.0.0.1:8000/api/stt", files=files)
                    if response.status_code == 200:
                        return response.json().get("text", "")
    except Exception as e:
        print(f"[MCP Whisper] Brain API fallback: {e}")
    return None


@server.tool()
async def transcribe_audio(audio_path: str, language: str | None = None) -> str:
    """Transcribe an audio file to text. Uses Brain server if available to save VRAM."""
    # 1. Try Brain API first (Shared Memory / No duplication)
    text = await _transcribe_via_brain(audio_path)
    if text is not None:
        return text

    # 2. Fallback to local model
    stt = await get_local_stt()
    lang = language or config.get("voice.stt.language", "uk")
    result = await stt.transcribe_file(audio_path, language=lang)
    return result.text


@server.tool()
async def record_and_transcribe(duration: float = 5.0, language: str | None = None) -> str:
    """Record audio from microphone and transcribe it."""
    # Recording usually happens locally on the client side
    stt = await get_local_stt()
    lang = language or config.get("voice.stt.language", "uk")
    result = await stt.record_and_transcribe(duration, language=lang)
    return result.text


if __name__ == "__main__":
    server.run()
