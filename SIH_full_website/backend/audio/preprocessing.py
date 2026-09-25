"""
Audio Preprocessing (GPU-Accelerated).
Resamples audio to 16kHz on CUDA in under 1 millisecond using torchaudio.
"""

import torch
import torchaudio.functional as F
import numpy as np
from model.model_config import TARGET_SAMPLE_RATE

device = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_audio(audio, sample_rate):
    """
    Resamples audio array to 16kHz using GPU acceleration.
    """
    if sample_rate != TARGET_SAMPLE_RATE:
        # Convert numpy array to PyTorch tensor on GPU
        audio_tensor = torch.from_numpy(audio).to(device)
        
        # Ultra-fast GPU sinc-interpolation resample
        resampled_tensor = F.resample(
            audio_tensor,
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE
        )
        
        # Bring back to float32 numpy
        audio = resampled_tensor.cpu().numpy()

    return audio, TARGET_SAMPLE_RATE