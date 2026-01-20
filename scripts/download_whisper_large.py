from faster_whisper import WhisperModel
import os
from pathlib import Path

def download():
    model_name = "large-v3"
    download_root = Path.home() / ".config" / "atlastrinity" / "models" / "faster-whisper"
    
    print(f"Starting download of {model_name} to {download_root}...")
    
    # This will trigger the download and conversion/verification
    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        download_root=str(download_root)
    )
    
    print("Download and initialization successful!")

if __name__ == "__main__":
    download()
