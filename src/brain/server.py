"""
AtlasTrinity Brain Server
Exposes the orchestrator via FastAPI for Electron IPC
"""

import os

# Suppress espnet2 UserWarning about non-writable tensors
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="espnet2.torch_utils.device_funcs")

# Import CONFIG_ROOT before using it
from .config import CONFIG_ROOT  # noqa: E402
from .config_loader import config  # noqa: E402
from .services_manager import ServiceStatus, ensure_all_services  # noqa: E402

# Step 1: Ensure core system services (Redis, Docker) are running
# We'll run this in the background to avoid blocking server binding

# Set API keys as environment variables for internal libraries
copilot_key = config.get_api_key("copilot_api_key")
github_token = config.get_api_key("github_token")

if copilot_key:
    os.environ["COPILOT_API_KEY"] = copilot_key
    print("[Server] ✓ COPILOT_API_KEY loaded from global context")
if github_token:
    os.environ["GITHUB_TOKEN"] = github_token
    print("[Server] ✓ GITHUB_TOKEN loaded from global context")

import asyncio
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from .logger import logger  # noqa: E402
from .orchestrator import SystemState, Trinity  # noqa: E402
from .production_setup import run_production_setup  # noqa: E402
from .voice.stt import SpeechType, WhisperSTT  # noqa: E402

# Global instances (Trinity will now find Redis running)
trinity = Trinity()
stt = WhisperSTT()  # Automatically reads model from config.yaml


class TaskRequest(BaseModel):
    request: str


class AudioRequest(BaseModel):
    action: str  # 'start_recording', 'stop_recording'


class SmartSTTRequest(BaseModel):
    previous_text: str = ""  # Accumulated transcript from previous chunks


# State
current_task = None
is_recording = False

from contextlib import asynccontextmanager  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("AtlasTrinity Brain is waking up...")

    # Initialize services in background
    asyncio.create_task(ensure_all_services())

    # Initialize components
    await trinity.initialize()

    # Production: copy configs from Resources/ to ~/.config/ if needed
    run_production_setup()

    # TTS models are lazy-loaded by ukrainian-tts on first call.
    # Pre-initializing them in background to avoid latency during first response.
    async def warmup():
        try:
            logger.info("[LifeSpan] Warming up voice engines...")
            # Warm up STT model (may be None if faster-whisper not installed)
            model = await stt.get_model()
            if model:
                logger.info("[LifeSpan] STT model loaded successfully.")
            else:
                logger.warning("[LifeSpan] STT model unavailable - faster-whisper not installed.")
            # Warm up TTS engine
            logger.info("[LifeSpan] Starting TTS engine initialization (this may take 2-5 mins)...")
            logger.info("[LifeSpan] SYSTEM IS READY for chat. Voice will be active once initialized.")
            await trinity.voice.get_engine()
            logger.info("[LifeSpan] Voice engines are ready.")
        except Exception as e:
            logger.error(f"[LifeSpan] Warmup error: {e}")

    asyncio.create_task(warmup())

    yield
    # Shutdown
    logger.info("AtlasTrinity Brain is going to sleep...")
    await trinity.shutdown()


app = FastAPI(title="AtlasTrinity Brain", lifespan=lifespan)

# CORS setup for Electron
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat")
async def chat(task: TaskRequest, background_tasks: BackgroundTasks):
    """Send a user request to the system"""

    if current_task and not current_task.done():
        raise HTTPException(status_code=409, detail="System is busy")

    print(f"[SERVER] Received request: {task.request}")
    logger.info(f"Received request: {task.request}")

    # Run orchestration in background/loop
    try:
        result = await trinity.run(task.request)
        return {"status": "completed", "result": result}
    except Exception as e:
        logger.exception(f"Error processing request: {task.request}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Health check for UI"""
    return {"status": "ok", "version": "1.0.1"}


@app.post("/api/session/reset")
async def reset_session():
    """Reset current session"""
    return await trinity.reset_session()


@app.get("/api/sessions")
async def get_sessions():
    """List all available sessions"""
    from .state_manager import state_manager
    return await state_manager.list_sessions()


@app.post("/api/sessions/restore")
async def restore_session(payload: Dict[str, str]):
    """Restore a specific session"""
    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    return await trinity.load_session(session_id)


@app.get("/api/state")
async def get_state():
    """Get current system state for UI polling"""
    state = trinity.get_state()

    # Enrich with service status if not all-ready
    if not ServiceStatus.is_ready:
        state["service_status"] = {
            "status": ServiceStatus.status_message,
            "details": ServiceStatus.details,
        }

    return state


@app.post("/api/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """Convert speech to text using Whisper"""
    try:
        # Save uploaded audio temporarily
        import subprocess
        import tempfile

        # Determine extension based on content_type
        content_type = audio.content_type or "audio/wav"
        if "webm" in content_type:
            suffix = ".webm"
        elif "ogg" in content_type:
            suffix = ".ogg"
        elif "mp3" in content_type:
            suffix = ".mp3"
        else:
            suffix = ".wav"

        # CHECK: Is the agent currently speaking?
        if trinity.voice.is_speaking:
            logger.info("[STT] Agent is speaking, ignoring audio to avoid feedback loop.")
            return {"text": "", "confidence": 0, "ignored": True}

        logger.info(f"[STT] Received audio: content_type={content_type}, using suffix={suffix}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            logger.info(f"[STT] Saved to: {temp_file_path}, size: {len(content)} bytes")

        # Конвертуємо webm/ogg/mp3 → wav для кращої роботи Whisper
        wav_path = temp_file_path
        if suffix != ".wav":
            wav_path = temp_file_path.replace(suffix, ".wav")
            try:
                # Optimized for Whisper large-v3-turbo: High clarity, no aggressive cutoff
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        temp_file_path,
                        "-af",
                        (
                            "highpass=f=80, "  # Remove sub-bass rumble
                            "loudnorm"  # Standardize loudness
                        ),
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-f",
                        "wav",
                        wav_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    logger.info(f"[STT] Converted to WAV: {wav_path}")
                    os.unlink(temp_file_path)
                else:
                    logger.warning(f"[STT] FFmpeg failed: {result.stderr}, using original file")
                    wav_path = temp_file_path
            except FileNotFoundError:
                logger.warning("[STT] FFmpeg not found, using original file")
                wav_path = temp_file_path
            except subprocess.TimeoutExpired:
                logger.warning("[STT] FFmpeg timeout, using original file")
                wav_path = temp_file_path

        # Transcribe using Whisper
        result = await stt.transcribe_file(wav_path)

        # Echo cancellation: Ignore if Whisper heard the agent's own voice
        clean_text = result.text.strip().lower().replace(".", "").replace(",", "")
        last_spoken = trinity.voice.last_text.replace(".", "").replace(",", "")

        # Check for exact match OR if result is part of what agent said
        # (common issue: Whisper catching the end of agent's sentence)
        if clean_text and (
            clean_text == last_spoken
            or clean_text in last_spoken
            or (len(clean_text) > 4 and last_spoken in clean_text)
        ):
            logger.info(f"[STT] Echo detected: '{result.text}', ignoring.")
            return {"text": "", "confidence": 0, "ignored": True}

        logger.info(f"[STT] Result: text='{result.text}', confidence={result.confidence}")

        # Clean up temp file(s)
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        if wav_path != temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

        return {"text": result.text, "confidence": result.confidence}

    except Exception as e:
        logger.exception(f"STT error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stt/smart")
async def smart_speech_to_text(
    audio: UploadFile = File(...), previous_text: str = Form(default="")
):
    """
    Smart STT with Full Duplex support (Barge-in).
    """
    try:
        import subprocess
        import tempfile
        import time

        # Determine extension based on content_type
        content_type = audio.content_type or "audio/wav"
        if "webm" in content_type:
            suffix = ".webm"
        elif "ogg" in content_type:
            suffix = ".ogg"
        elif "mp3" in content_type:
            suffix = ".mp3"
        else:
            suffix = ".wav"

        # --- FULL DUPLEX CHANGE: REMOVED BLOCKING CHECK ---
        # We process audio even if agent is speaking to allow interruption.
        
        # logger.info(f"[STT] Received audio: content_type={content_type}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Convert to WAV
        wav_path = temp_file_path
        if suffix != ".wav":
            wav_path = temp_file_path.replace(suffix, ".wav")
            try:
                # Optimized for Whisper large-v3-turbo
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        temp_file_path,
                        "-af",
                        ("highpass=f=80, " "loudnorm"),
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-f",
                        "wav",
                        "-loglevel", "error", # Less verbose
                        wav_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if result.returncode == 0:
                    os.unlink(temp_file_path)  # Delete original
                else:
                    logger.warning(f"[STT] FFmpeg failed, using original")
                    wav_path = temp_file_path
            except Exception:
                wav_path = temp_file_path

        # Smart analysis with context (async)
        result = await stt.transcribe_with_analysis(wav_path, previous_text=previous_text)
        
        # --- ECHO CANCELLATION & INTERRUPTION LOGIC ---
        
        from difflib import SequenceMatcher

        now = time.time()
        agent_was_speaking = trinity.voice.is_speaking
        
        clean_text = result.text.strip().lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "")
        
        # Get history + current last text
        phrase_history = list(trinity.voice.history)
        if trinity.voice.last_text:
             phrase_history.append(trinity.voice.last_text)
        
        # Calculate overlap similarity against HISTORY
        is_echo = False
        
        # 1. Check against history (robust against delays)
        if clean_text:
            for past_phrase in phrase_history:
                past_clean = past_phrase.strip().lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "")
                if not past_clean: continue
                
                ratio = SequenceMatcher(None, clean_text, past_clean).ratio()
                
                # Check substring match
                if past_clean and clean_text in past_clean:
                        ratio = 1.0
                
                # Strict threshold for history matching
                if ratio > 0.75:
                    is_echo = True
                    logger.info(f"[STT] Echo detected (History Match): '{clean_text}' ~= '{past_clean}' (Ratio: {ratio:.2f})")
                    break

        # 2. Time-based Heuristics for short noises (only if currently speaking or just finished)
        if not is_echo and (agent_was_speaking or (now - trinity.voice.last_speak_time < 3.0)):
             if result.confidence < 0.6:
                 is_echo = True
                 logger.info(f"[STT] Noise detected (Low Conf): '{result.text}'")

             # Special check for very short "interjections"
             if len(clean_text) < 5 and result.confidence < 0.8:
                 is_echo = True
                 logger.info(f"[STT] Short noise ignored: '{result.text}'")

        if is_echo:
            logger.info(f"[STT] Echo/Noise detected: '{result.text}' -> IGNORED")
            return {
                "text": "",
                "speech_type": "noise",
                "confidence": 0,
                "combined_text": previous_text,
                "should_send": False,
                "is_continuation": False,
                "ignored": True,
            }

        # --- BARGE-IN TRIGGER ---
        # If we detected valid NEW speech (not echo) AND agent is speaking -> INTERRUPT!
        if (
            result.speech_type == SpeechType.NEW_PHRASE 
            and result.text 
            and agent_was_speaking
            and not is_echo
            and result.confidence > 0.65 # Bumped slightly to 0.65
        ):
            logger.info(f"[STT] 🛑 BARGE-IN DETECTED: '{result.text}' -> Stopping TTS.")
            trinity.voice.stop() # Stop current speech immediately

        # Standard logging
        if result.text:
             logger.info(
                f"[STT] Result: '{result.text}' (Type: {result.speech_type.value}, Conf: {result.confidence:.2f})"
            )

        # Clean up
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        if wav_path != temp_file_path and os.path.exists(temp_file_path):
             os.unlink(temp_file_path)

        return {
            "text": result.text,
            "speech_type": result.speech_type.value,
            "confidence": result.confidence,
            "combined_text": result.combined_text,
            "should_send": result.should_send,
            "is_continuation": result.is_continuation,
            "no_speech_prob": result.no_speech_prob,
        }

    except Exception as e:
        logger.error(f"Smart STT error: {str(e)}")
        # Don't crash client loop, return empty result
        return {
                "text": "",
                "speech_type": "noise",
                "confidence": 0,
                "combined_text": previous_text,
                "should_send": False,
                "is_continuation": False,
                "ignored": True,
            }


@app.post("/api/voice/transcribe")
async def transcribe_audio(file_path: str):
    """Transcribe a wav file"""
    result = await stt.transcribe_file(file_path)
    return {"text": result.text, "confidence": result.confidence}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
