import asyncio
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.brain.voice.tts import VoiceManager


async def test_tts():
    print("Initializing VoiceManager...")
    vm = VoiceManager(device="cpu")

    print("Testing 'afplay' directly...")
    # Create a silent wav or use a system one if exists
    # For now, just check if afplay command exists
    import subprocess

    try:
        subprocess.run(["afplay", "--help"], check=False, capture_output=True)
        print("'afplay' command is available.")
    except Exception as e:
        print(f"ERROR: 'afplay' not available: {e}")

    print("Generating and playing speech...")
    # This will actually try to speak and play
    result = await vm.speak("atlas", "Привіт! Це тест системи озвучення.")
    print(f"Speak result: {result}")

    if result == "pipelined_playback_completed":
        print("TTS verification script finished successfully.")
    else:
        print(f"TTS verification script failed or was interrupted: {result}")


if __name__ == "__main__":
    asyncio.run(test_tts())
