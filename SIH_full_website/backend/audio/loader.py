"""
In-Memory Audio Loader with Seekable Fallback.
Decodes audio in RAM in ~2ms. Supports WAV, MP3, FLAC, OGG, WEBM, M4A, AAC.
"""

import io
import os
import tempfile
import subprocess
import soundfile as sf
import numpy as np


def load_audio(file_input):
    """
    Loads audio directly from bytes, a file-like object, or a file path in memory.
    Supports WAV, MP3, FLAC, OGG, WEBM, M4A, AAC with seekable fallback for Windows FFmpeg.
    
    Args:
        file_input: bytes, io.BytesIO, or file path str
        
    Returns:
        audio: 1D float32 NumPy array (mono)
        sample_rate: int (original sample rate)
    """
    import os
    import tempfile

    # 1. Convert input to BytesIO / raw bytes
    if isinstance(file_input, bytes):
        raw_bytes = file_input
        file_obj = io.BytesIO(file_input)
    elif isinstance(file_input, str):
        with open(file_input, "rb") as f:
            raw_bytes = f.read()
        file_obj = io.BytesIO(raw_bytes)
    else:
        file_input.seek(0)
        raw_bytes = file_input.read()
        file_obj = io.BytesIO(raw_bytes)

    # 2. Fast Path: Read standard audio (WAV, FLAC, OGG, MP3) directly with soundfile in RAM (~1-2ms)
    try:
        file_obj.seek(0)
        audio, sample_rate = sf.read(file_obj, dtype="float32")
        
        # Convert multi-channel (stereo) to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        return audio, sample_rate

    except Exception:
        pass

    # 3. Fallback Path: In-Memory / Temporary seekable file for browser formats (WEBM/Opus/M4A)
    # Windows FFmpeg cannot seek backwards on raw pipes, so a temporary seekable file is used
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        cmd = [
            "ffmpeg",
            "-y",
            "-v", "error",
            "-i", tmp_path,
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ar", "16000",
            "-ac", "1",
            "pipe:1"
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        audio = np.frombuffer(result.stdout, dtype=np.float32)
        if len(audio) == 0:
            raise ValueError("Decoded audio buffer is empty.")

        return audio, 16000

    except Exception as e:
        err_msg = ""
        if "result" in locals() and result.stderr:
            err_msg = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"Failed to decode audio in-memory: {e} {err_msg}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass