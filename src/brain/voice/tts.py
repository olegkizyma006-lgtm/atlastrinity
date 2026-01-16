"""
AtlasTrinity TTS - Ukrainian Text-to-Speech

Uses robinhad/ukrainian-tts for agent voices:
- Atlas: Dmytro (male)
- Tetyana: Tetiana (female)
- Grisha: Mykyta (male)

NOTE: TTS models must be set up before first use via setup_dev.py
"""

import os
import tempfile
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import CONFIG_ROOT, MODELS_DIR
from ..config_loader import config

# Lazy import to avoid loading heavy dependencies at startup
TTS_AVAILABLE = None

def _check_tts_available():
    global TTS_AVAILABLE
    if TTS_AVAILABLE is not None:
        return TTS_AVAILABLE
        
    try:
        import ukrainian_tts
        TTS_AVAILABLE = True
        print("[TTS] Ukrainian TTS available")
    except ImportError:
        TTS_AVAILABLE = False
        print(
            "[TTS] Warning: ukrainian-tts not installed. Run: pip install git+https://github.com/robinhad/ukrainian-tts.git"
        )
    return TTS_AVAILABLE


@dataclass
class VoiceConfig:
    """Voice configuration for an agent"""

    name: str
    voice_id: str
    description: str


# Agent voice mappings
AGENT_VOICES = {
    "atlas": VoiceConfig(
        name="Atlas", voice_id="Dmytro", description="Male voice for Meta-Planner"
    ),
    "tetyana": VoiceConfig(
        name="Tetyana", voice_id="Tetiana", description="Female voice for Executor"
    ),
    "grisha": VoiceConfig(name="Grisha", voice_id="Mykyta", description="Male voice for Visor"),
}


class AgentVoice:
    """
    TTS wrapper for agent voices

    Usage:
        voice = AgentVoice("atlas")
        voice.speak("Hello, I am Atlas")
    """

    def __init__(self, agent_name: str, device: str = None):
        """
        Initialize voice for an agent

        Args:
            agent_name: One of 'atlas', 'tetyana', 'grisha'
            device: 'cpu', 'cuda', or 'mps' (Apple Silicon). If None, reads from config.yaml
        """
        self.agent_name = agent_name.lower()

        # Get device from config.yaml with fallback
        voice_config = config.get("voice.tts", {})
        self.device = device or voice_config.get("device", "mps")

        if self.agent_name not in AGENT_VOICES:
            raise ValueError(
                f"Unknown agent: {agent_name}. Must be one of: {list(AGENT_VOICES.keys())}"
            )

        self.config = AGENT_VOICES[self.agent_name]
        self._tts = None
        self._voice_enum = None  # Cache enum

        # Get voice enum
        if _check_tts_available():
            # Lazy import Voices as well
            try:
                from ukrainian_tts.tts import Voices

                self._voice_enum = getattr(Voices, self.config.voice_id, Voices.Dmytro)
                self._voice = self._voice_enum.value
            except Exception as e:
                print(f"[TTS] Failed to import Voices: {e}")
                # Set default voice
                self._voice = "Dmytro"
        else:
            self._voice = None

    @property
    def tts(self):
        """Lazy initialize TTS engine"""
        if self._tts is None and _check_tts_available():
            # Import only here to avoid issues during startup
            try:
                from ukrainian_tts.tts import TTS as UkrainianTTS

                global TTS
                TTS = UkrainianTTS
            except Exception as e:
                print(f"[TTS] Failed to import Ukrainian TTS: {e}")
                return None

            # Models should already be in MODELS_DIR from setup_dev.py
            if not MODELS_DIR.exists():
                print(f"[TTS] ⚠️  Models directory not found: {MODELS_DIR}")
                print("[TTS] Run setup_dev.py first to download TTS models")
                return None

            required_files = ["model.pth", "feats_stats.npz", "spk_xvector.ark"]
            missing = [f for f in required_files if not (MODELS_DIR / f).exists()]

            if missing:
                print(f"[TTS] ⚠️  Missing TTS model files: {missing}")
                print("[TTS] Run setup_dev.py to download them")
                return None

            try:
                print("[TTS] Initializing engine on " + str(self.device) + "...")
                print(
                    "downloading https://github.com/robinhad/ukrainian-tts/releases/download/v6.0.0"
                )
                self._tts = TTS(cache_folder=str(MODELS_DIR))
                print("downloaded.")
                print(f"[TTS] ✅ {self.config.name} voice ready on {self.device}")
            except Exception as e:
                print(f"[TTS] Error: {e}")
                import traceback

                traceback.print_exc()
                return None
        return self._tts

    def speak(self, text: str, output_file: Optional[str] = None) -> Optional[str]:
        """
        Generate speech from text

        Args:
            text: Ukrainian text to speak
            output_file: Optional path to save audio. If None, uses temp file

        Returns:
            Path to the generated audio file, or None if TTS not available
        """
        if not _check_tts_available():
            print(f"[TTS] [{self.config.name}]: {text}")
            return None

        if not text:
            return None

        # Determine output path
        if output_file is None:
            output_file = os.path.join(
                tempfile.gettempdir(), f"tts_{self.agent_name}_{hash(text) % 10000}.wav"
            )

        try:
            with open(output_file, mode="wb") as f:
                # Import Stress and Voices only here
                from ukrainian_tts.tts import Stress, Voices

                _, accented_text = self.tts.tts(
                    text, self._voice, Stress.Dictionary.value, f  # Use cached value
                )

            print(f"[TTS] [{self.config.name}]: {text}")
            return output_file

        except Exception as e:
            print(f"[TTS] Error generating speech: {e}")
            return None

    def speak_and_play(self, text: str) -> bool:
        """
        Generate speech and play it immediately (macOS)

        Args:
            text: Ukrainian text to speak

        Returns:
            True if successfully played, False otherwise
        """
        audio_file = self.speak(text)

        if audio_file and os.path.exists(audio_file):
            return self._play_audio(audio_file)

        return False

    def _play_audio(self, file_path: str) -> bool:
        """Play audio file on macOS"""
        try:
            import subprocess

            subprocess.run(["afplay", file_path], check=True, capture_output=True)
            return True
        except Exception as e:
            print(f"[TTS] Error playing audio: {e}")
            return False


class VoiceManager:
    """
    Centralized TTS manager for all agents
    """

    def __init__(self, device: str = "cpu"):
        voice_config = config.get("voice.tts", {})
        self.enabled = voice_config.get("enabled", True)  # Check enabled flag
        self.device = device
        self._tts = None
        self.is_speaking = False  # Flag to prevent self-listening
        self.last_text = ""  # Last spoken text for echo filtering
        self.last_speak_time = 0.0  # End time of the last agent phrase
        self._lock = None  # To be initialized in the loop

    async def get_engine(self):
        """Async version to get engine with lazy initialization"""
        if not self.enabled:
            print("[TTS] TTS is disabled in config")
            return None

        await self._initialize_if_needed_async()
        return self._tts

    @property
    def engine(self):
        """Legacy sync engine access (will block if not already initialized)"""
        if not self.enabled:
            return None
        self._initialize_if_needed()
        return self._tts

    def _initialize_if_needed(self):
        """Synchronous initialization (not recommended for main thread)"""
        if self._tts is None and _check_tts_available():
            import asyncio
            try:
                # If we are in a loop, try to use the async version? 
                # No, property must be sync.
                self._load_engine_sync()
            except Exception as e:
                print(f"[TTS] Sync initialization error: {e}")

    async def _initialize_if_needed_async(self):
        if self._tts is None:
            # Check availability in a thread because it involves a heavy import
            available = await asyncio.to_thread(_check_tts_available)
            if available:
                print(f"[TTS] Initializing engine on {self.device} (Async)...")
                await asyncio.to_thread(self._load_engine_sync)
            else:
                print("[TTS] Voice engine skip: ukrainian-tts not installed.")

    def _load_engine_sync(self):
        if self._tts is not None:
            return
            
        cache_dir = MODELS_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            print("[TTS] Loading ukrainian-tts and Stanza resources...")
            import os
            from contextlib import contextmanager
            from ukrainian_tts.tts import TTS as UkrainianTTS

            @contextmanager
            def tmp_cwd(path):
                old_path = os.getcwd()
                os.chdir(path)
                try:
                    yield
                finally:
                    os.chdir(old_path)

            with tmp_cwd(str(cache_dir)):
                print("[TTS] Downloading/Verifying models in models/tts...")
                self._tts = UkrainianTTS(cache_folder=str(cache_dir), device=self.device)
                print("[TTS] Engine object created successfully.")
        except Exception as e:
            print(f"[TTS] Failed to initialize engine: {e}")
            self._tts = None

    async def speak(self, agent_id: str, text: str) -> Optional[str]:
        """
        Generate and play speech for specific agent
        """
        if self._lock is None:
            import asyncio
            self._lock = asyncio.Lock()

        async with self._lock:
            if not _check_tts_available() or not text:
                print(f"[TTS] [{agent_id.upper()}] (Text-only): {text}")
                return None

            agent_id = agent_id.lower()
            if agent_id not in AGENT_VOICES:
                print(f"[TTS] Unknown agent: {agent_id}")
                return None

        # Import Voices and Stress here
        from ukrainian_tts.tts import Stress, Voices

        config = AGENT_VOICES[agent_id]
        voice_enum = getattr(Voices, config.voice_id).value

        # Generate to temp file
        output_file = os.path.join(
            tempfile.gettempdir(), f"tts_{agent_id}_{hash(text) % 10000}.wav"
        )

        try:
            # Generate (CPU intensive, run in thread to avoid blocking loop)
            import asyncio

            async def _gen_f(c_text, c_idx):
                c_id = f"{agent_id}_{c_idx}_{hash(c_text) % 10000}"
                c_file = os.path.join(tempfile.gettempdir(), f"tts_{c_id}.wav")
                def _do_gen():
                    with open(c_file, mode="wb") as f:
                        self.engine.tts(c_text, voice_enum, Stress.Dictionary.value, f)
                await asyncio.to_thread(_do_gen)
                return c_file

            # 1. Split text into sentences/chunks
            import re
            chunks = re.split(r'([.!?]+(?:\s+|$))', text)
            processed_chunks = []
            for i in range(0, len(chunks)-1, 2):
                processed_chunks.append(chunks[i] + chunks[i+1])
            if len(chunks) % 2 == 1 and chunks[-1]:
                 processed_chunks.append(chunks[-1])
            
            final_chunks = [c.strip() for c in processed_chunks if c.strip()]
            
            # 2. Refine chunks: Merge very short sentences (< 40 chars)
            min_len = 40
            refined_chunks = []
            temp_chunk = ""
            
            for chunk in final_chunks:
                if temp_chunk:
                    temp_chunk += " " + chunk
                else:
                    temp_chunk = chunk
                
                if len(temp_chunk) >= min_len:
                    refined_chunks.append(temp_chunk)
                    temp_chunk = ""
            
            if temp_chunk:
                if refined_chunks:
                    refined_chunks[-1] += " " + temp_chunk
                else:
                    refined_chunks.append(temp_chunk)
            
            final_chunks = refined_chunks or [text]
            
            print(f"[TTS] [{config.name}] Starting pipelined playback for {len(final_chunks)} chunks...")
            import time
            start_time = time.time()
            
            # 2. Pipelined Generation & Playback
            # Generate the first chunk immediately
            current_file = await _gen_f(final_chunks[0], 0)
            first_chunk_time = time.time() - start_time
            print(f"[TTS] [{config.name}] First chunk ready in {first_chunk_time:.2f}s")
            
            for idx in range(len(final_chunks)):
                # Start generating the NEXT chunk in the background if it exists
                next_gen_task = None
                if idx + 1 < len(final_chunks):
                    next_gen_task = asyncio.create_task(_gen_f(final_chunks[idx+1], idx+1))
                    print(f"[TTS] [{config.name}] ⏩ Background generating chunk {idx+2}...")

                # Play current chunk
                if not os.path.exists(current_file):
                    print(f"[TTS] [{config.name}] ⚠ Error: File not found for playback: {current_file}")
                    continue

                print(f"[TTS] [{config.name}] 🔊 Speaking chunk {idx+1}/{len(final_chunks)}: {final_chunks[idx][:50]}...")
                self.last_text = final_chunks[idx].strip().lower()
                
                self.is_speaking = True
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "afplay", current_file,
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await proc.communicate()
                    if proc.returncode != 0:
                        print(f"[TTS] [{config.name}] ⚠ afplay failed with code {proc.returncode}. Error: {stderr.decode()}")
                except Exception as e:
                    print(f"[TTS] [{config.name}] ⚠ Execution error (afplay): {e}")
                finally:
                    self.is_speaking = False
                
                # Cleanup current file after playback
                if os.path.exists(current_file):
                    os.remove(current_file)

                # Prepare for next iteration: wait for the background generation to finish
                if next_gen_task:
                    current_file = await next_gen_task

            # Final grace period
            await asyncio.sleep(0.3)
            import time
            self.last_speak_time = time.time()
            return "pipelined_playback_completed"

        except Exception as e:
            print(f"[TTS] Error: {e}")
            return None
