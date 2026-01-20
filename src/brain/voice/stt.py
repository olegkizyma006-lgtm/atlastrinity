"""
AtlasTrinity STT - Speech-to-Text using OpenAI Whisper

Primary: OpenAI Whisper (local)
Fallback: Browser Web Speech API (handled in frontend)
"""

import asyncio
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from ..config import CONFIG_ROOT
from ..config_loader import config
from ..logger import logger

# Lazy import to avoid loading heavy dependencies at startup
WHISPER_AVAILABLE = None
WhisperModel = None


def _check_whisper_available():
    global WHISPER_AVAILABLE, WhisperModel
    if WHISPER_AVAILABLE is not None:
        return WHISPER_AVAILABLE

    try:
        from faster_whisper import WhisperModel as _WhisperModel

        WhisperModel = _WhisperModel
        WHISPER_AVAILABLE = True
    except ImportError:
        WHISPER_AVAILABLE = False
        print("[STT] Warning: faster-whisper not installed. Run: pip install faster-whisper")
    return WHISPER_AVAILABLE


# Lazy import for audio recording
AUDIO_AVAILABLE = None
sd = None
sf = None


def _check_audio_available():
    global AUDIO_AVAILABLE, sd, sf
    if AUDIO_AVAILABLE is not None:
        return AUDIO_AVAILABLE

    try:
        import sounddevice as _sd
        import soundfile as _sf

        sd = _sd
        sf = _sf
        AUDIO_AVAILABLE = True
    except ImportError:
        AUDIO_AVAILABLE = False
        print("[STT] Warning: sounddevice/soundfile not installed. Audio recording disabled.")
    return AUDIO_AVAILABLE


class SpeechType(str, Enum):
    """Type of detected speech"""

    SAME_USER = "same_user"  # Same user continues
    NEW_PHRASE = "new_phrase"  # New phrase from the same user
    BACKGROUND_NOISE = "noise"  # Background noise
    OTHER_VOICE = "other_voice"  # Another voice or side conversation
    SILENCE = "silence"  # Silence
    OFF_TOPIC = "off_topic"  # Off-topic or side conversation


@dataclass
class TranscriptionResult:
    """Result of speech transcription"""

    text: str
    language: str
    confidence: float
    segments: list
    no_speech_prob: float = 0.0  # Probability that it's not speech
    speech_type: SpeechType = SpeechType.SAME_USER


@dataclass
class SmartSTTResult:
    """Result of smart STT analysis"""

    text: str
    speech_type: SpeechType
    confidence: float
    should_send: bool  # Trigger to send aggregated text
    combined_text: str  # Combined text if available
    is_continuation: bool  # If it continues previous phrase
    no_speech_prob: float = 0.0  # Probability of no speech


class WhisperSTT:
    """
    Speech-to-Text using Faster Whisper (CTranslate2)
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        # Get STT config from config.yaml
        stt_config = config.get("voice.stt", {})

        # Use model from config (priority - config.yaml)
        self.model_name = model_name or stt_config.get("model", "large-v3-turbo")

        # Mapping for optimized turbo models
        if self.model_name in ["turbo", "large-v3-turbo"]:
            # Check if we should use the specific ctranslate2 conversion or let faster-whisper handle it
            # Standard faster-whisper now supports "large-v3-turbo" directly if available on HF
            pass

        self.language = stt_config.get("language", "uk")

        configured_device = device or stt_config.get("device", "auto")

        if configured_device == "auto":
            import torch

            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                # CPU works faster/better with ctranslate2 on Mac than experimental mps (fp16 issues)
                self.device = "cpu"
        elif configured_device == "mps":
            # faster-whisper still doesn't have official stable MPS support in CTranslate2
            # Use 'cpu' with 'int8' for best performance on Mac
            self.device = "cpu"
        else:
            self.device = configured_device

        self._model = None
        self._load_lock = asyncio.Lock()
        self._transcribe_lock = asyncio.Lock()  # Prevent concurrent transcriptions from overloading CPU
        self.download_root = CONFIG_ROOT / "models" / "faster-whisper"

        # Compute type selection based on device
        if self.device == "cuda":
            self.compute_type = "float16"
        else:
            self.compute_type = "int8"  # Best for CPU/MPS stability and speed

        # Stateful tracking for Smart STT
        self.last_speech_time = 0.0
        self.silence_threshold = 6.5  # Seconds of silence before sending phrase (slightly more than UI 6s)

    def _filter_text(self, text: str) -> str:
        """Filter out common hallucinations and noise patterns"""
        t = text.strip()
        if not t:
            return ""

        # Common Whisper hallucinations
        blacklist = [
            "дякую за увагу",
            "дякую за перегляд",
            "підпишіться на канал",
            "підписуйтесь",
            "ставте лайки",
            "на все добре",
            "до зустрічі",
            "thank you for watching",
            "subscribe",
            "thanks for watching",
            "субтитри",
            "субтитры",
            "оля шор",
            "а.семкин",
            "а.егорова",
            "о.голубкін",
        ]

        t_lower = t.lower()
        for p in blacklist:
            if p in t_lower:
                # If hallucination is a significant part of the text, it's likely garbage
                if len(p) > len(t) * 0.7:
                    return ""
                # Otherwise just strip it (rare case)
                t = t.replace(p, "").strip()
                t = t.replace(p.capitalize(), "").strip()

        # Clean up repeated punctuation or weird chars
        import re
        t = re.sub(r'([.!?])\1+', r'\1', t)
        
        return t

    async def get_model(self):
        """Lazy-load Faster Whisper model non-blockingly"""
        async with self._load_lock:
            if self._model is None:
                if not _check_whisper_available():
                    logger.error("[STT] faster-whisper is not installed. Cannot load WhisperModel.")
                    return None

                print(f"[STT] Loading Faster-Whisper model: {self.model_name} on {self.device}...")
                self.download_root.mkdir(parents=True, exist_ok=True)

                def load():
                    return cast(Any, WhisperModel)(
                        self.model_name,
                        device=self.device,
                        compute_type=self.compute_type,
                        download_root=str(self.download_root),
                    )

                self._model = await asyncio.to_thread(load)
                print(f"[STT] Model loaded successfully from {self.download_root}")
            return self._model

    async def transcribe_file(
        self, audio_path: str, language: str | None = None
    ) -> TranscriptionResult:
        language = language or self.language

        if not _check_whisper_available():
            return TranscriptionResult(text="", language="uk", confidence=0, segments=[])

        model = await self.get_model()

        try:
            # Faster Whisper parameters for high accuracy
            def transcribe():
                segments, info = cast(Any, model).transcribe(
                    audio_path,
                    language=language,
                    beam_size=5,  # Better accuracy for large-v3
                    temperature=0.0,  # Deterministic output
                    initial_prompt="Це професійна розмова з AI-асистентом Атласом. Пиши чистою українською мовою з правильними розділовими знаками. Наприклад: 'Слава Україні! Як твої справи? Потрібно виконати це завдання.' Використовуй коми, крапки, знаки питання та оклику відповідно до інтонації.",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=1000),
                )
                return list(segments), info

            async with self._transcribe_lock:
                segments_list, info = await asyncio.to_thread(transcribe)

            full_text = " ".join([s.text for s in segments_list]).strip()
            full_text = self._filter_text(full_text)

            # Calculate average confidence
            if segments_list:
                avg_prob = sum(s.avg_logprob for s in segments_list) / len(segments_list)
                # Convert logprob to 0-1 range (approximate)
                confidence = max(0.0, min(1.0, (avg_prob + 3.0) / 3.0))
            else:
                confidence = 0.0

            return TranscriptionResult(
                text=full_text,
                language=info.language,
                confidence=confidence,
                segments=[{"text": s.text, "start": s.start, "end": s.end} for s in segments_list],
                no_speech_prob=(
                    1.0 - info.probability_of_speech
                    if hasattr(info, "probability_of_speech")
                    else 0.0
                ),
            )
        except Exception as e:
            print(f"[STT] Transcription error: {e}")
            return TranscriptionResult(text="", language=language, confidence=0, segments=[])

    async def transcribe_with_analysis(
        self, audio_path: str, previous_text: str = "", language: str | None = None
    ) -> SmartSTTResult:
        import time

        now = time.time()

        result = await self.transcribe_file(audio_path, language)
        speech_type = self._analyze_speech_type(result, previous_text)

        # Phrase continuation: if same user or new phrase (meaningful)
        is_meaningful = speech_type in [SpeechType.SAME_USER, SpeechType.NEW_PHRASE]

        if is_meaningful and result.text.strip():
            # Update last activity time (real speech)
            self.last_speech_time = now
            combined_text = self._combine_phrases(previous_text, result.text)
            should_send = False  # Wait for silence
        else:
            combined_text = previous_text

            # If we have accumulated text, check if it's time to send
            if previous_text.strip() and self.last_speech_time > 0:
                silence_duration = now - self.last_speech_time

                # If SILENCE or NOISE (non-speech) for > 3 seconds
                # Including NOISE here ensures we don't wait forever in loud environments
                is_waiting_type = speech_type in [
                    SpeechType.SILENCE,
                    SpeechType.BACKGROUND_NOISE,
                ]

                if is_waiting_type and silence_duration >= self.silence_threshold:
                    logger.info(
                        f"[STT] Debounce timeout ({silence_duration:.1f}s), sending aggregated text."
                    )
                    should_send = True
                    # Reset timer
                    self.last_speech_time = 0.0
                else:
                    should_send = False
            else:
                should_send = False

        return SmartSTTResult(
            text=result.text,
            speech_type=speech_type,
            confidence=result.confidence,
            should_send=should_send,
            combined_text=combined_text,
            is_continuation=is_meaningful,
            no_speech_prob=result.no_speech_prob,
        )

    def _analyze_speech_type(self, result: TranscriptionResult, previous_text: str) -> SpeechType:
        text = result.text.strip().lower()

        if not text or result.no_speech_prob > 0.7:
            return SpeechType.SILENCE

        # 1. Aggressive blacklist for common hallucinations
        hard_blacklist = [
            "оля шор",
            "субтитри",
            "субтитры",
            "дякую за перегляд",
            "підпишіться",
            "підписуйтесь",
            "дякую за увагу",
            "на все добре",
            "а.семкин",
            "а.егорова",
            "о.голубкін",
            "музика",
            "playing music",
            "про що ви знаєте",
            "про те, що ви не знаєте",
        ]

        if any(p in text for p in hard_blacklist):
            return SpeechType.BACKGROUND_NOISE

        # 1.1 Short hallucinations
        if len(text) < 3 and text in ["оля", "шор", "про"]:
            return SpeechType.BACKGROUND_NOISE

        # 2. Low confidence
        if result.confidence < 0.35:
            # Common filler words/noise if low confidence
            noise_patterns = [".", "..", "...", "м", "мм", "[music]", "[noise]", "♪", "♫"]
            if any(p in text for p in noise_patterns) or len(text) < 2:
                return SpeechType.BACKGROUND_NOISE

        # 3. Single word check
        words = text.split()
        if len(words) == 1:
            if result.confidence < 0.80:
                return SpeechType.BACKGROUND_NOISE
            
            # Common interjections
            common_interjections = ["так", "ні", "ок", "дякую", "привіт", "зрозумів", "отже", "тобто"]
            if text in common_interjections and result.confidence < 0.92:
                if not previous_text:
                    return SpeechType.BACKGROUND_NOISE

        # 4. Looping check
        if len(words) > 5:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.4:
                return SpeechType.BACKGROUND_NOISE

        return SpeechType.NEW_PHRASE

    def _combine_phrases(self, prev_text: str, new_text: str) -> str:
        prev = prev_text.strip()
        new = new_text.strip()
        if not prev:
            return new[0].upper() + new[1:] if len(new) > 1 else new.upper()
        if not new:
            return prev

        # Add space and capitalization check
        # If previous phrase ends with punctuation
        if prev[-1] in ".!?":
            return f"{prev} {new[0].upper() + new[1:] if len(new) > 1 else new.upper()}"
        else:
            return f"{prev} {new}"

    def list_audio_devices(self) -> list:
        if not _check_audio_available():
            return []
        devices = cast(Any, sd).query_devices()
        return [
            {"id": i, "name": d["name"], "channels": d["max_input_channels"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]

    async def record_and_transcribe(
        self, duration: float = 5.0, language: str | None = None
    ) -> TranscriptionResult:
        """Record audio and transcribe it"""
        if not _check_audio_available():
            return TranscriptionResult(
                text="Audio recording not available",
                language="uk",
                confidence=0,
                segments=[],
            )

        fs = 16000
        print(f"[STT] Recording for {duration} seconds...")
        recording = await asyncio.to_thread(
            cast(Any, sd).rec, int(duration * fs), samplerate=fs, channels=1
        )
        await asyncio.to_thread(cast(Any, sd).wait)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav_path = tf.name
            await asyncio.to_thread(cast(Any, sf).write, wav_path, recording, fs)

        try:
            result = await self.transcribe_file(wav_path, language)
            return result
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)


# MCP Wrapper
class WhisperMCPServer:
    def __init__(self):
        self.stt = WhisperSTT()

    async def transcribe_audio(self, audio_path: str, language: str | None = "uk"):
        result = await self.stt.transcribe_file(audio_path, language)
        return {"text": result.text, "confidence": result.confidence}

    async def record_and_transcribe(self, duration: float = 5.0, language: str | None = None):
        result = await self.stt.record_and_transcribe(duration, language)
        return {"text": result.text, "confidence": result.confidence}


if __name__ == "__main__":
    print("Whisper STT Module Loaded")
