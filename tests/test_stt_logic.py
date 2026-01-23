import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.brain.voice.stt import SpeechType, TranscriptionResult, WhisperSTT


def test_stt_blacklist():
    print("Initializing WhisperSTT (mocked)...")
    stt = WhisperSTT()

    # Test cases for hallucinations
    test_cases = [
        ("привіток справи", 0.85, SpeechType.BACKGROUND_NOISE),
        ("оля шор", 0.9, SpeechType.BACKGROUND_NOISE),
        ("привіток", 0.8, SpeechType.BACKGROUND_NOISE),
        ("Добрий день, як справи?", 0.9, SpeechType.NEW_PHRASE),  # Should NOT be noise
        ("промилки", 0.7, SpeechType.BACKGROUND_NOISE),
        ("так", 0.8, SpeechType.BACKGROUND_NOISE),  # Short phrase check
        ("ні", 0.8, SpeechType.BACKGROUND_NOISE),  # Short phrase check
        ("Зроби мені каву", 0.9, SpeechType.NEW_PHRASE),
    ]

    success_count = 0
    for text, conf, expected_type in test_cases:
        result = TranscriptionResult(
            text=text, language="uk", confidence=conf, segments=[], no_speech_prob=0.1,
        )
        actual_type = stt._analyze_speech_type(result, "")

        if actual_type == expected_type:
            print(f"PASS: '{text}' -> {actual_type}")
            success_count += 1
        else:
            print(f"FAIL: '{text}' -> expected {expected_type}, got {actual_type}")

    if success_count == len(test_cases):
        print("STT Blacklist verification SUCCESS.")
    else:
        print(f"STT Blacklist verification FAILED ({success_count}/{len(test_cases)} passed).")


if __name__ == "__main__":
    test_stt_blacklist()
