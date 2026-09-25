"""
Pretrained Wav2Vec2 Audio Deepfake Detector from Hugging Face
Supports single chunk and batched multi-chunk inference.
"""

import torch
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor
)
from model.model_config import (
    MODEL_NAME,
    REAL_LABEL,
    FAKE_LABEL
)

# Load model and feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)

# Set device and evaluation mode
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


def predict_chunks_batched(chunks, sample_rate):
    """
    Runs batched inference on a list of audio chunks simultaneously in a single forward pass.
    
    Args:
        chunks: List of 1D numpy arrays (audio chunks)
        sample_rate: Audio sampling rate (e.g. 16000)
        
    Returns:
        List of tuples: [(prob_real, prob_fake), ...] for each chunk
    """
    if not chunks:
        return []

    # Process all chunks together with padding
    inputs = feature_extractor(
        chunks,
        sampling_rate=sample_rate,
        padding=True,
        return_tensors="pt"
    )

    # Move inputs to device (GPU/CPU)
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Inference mode is faster than no_grad()
    with torch.inference_mode():
        outputs = model(**inputs)

    # Convert logits to probabilities for all chunks
    probabilities = torch.softmax(outputs.logits, dim=-1)

    results = []
    for prob in probabilities:
        prob_real = prob[REAL_LABEL].item()
        prob_fake = prob[FAKE_LABEL].item()
        results.append((prob_real, prob_fake))

    return results


def predict_chunk(chunk, sample_rate):
    """
    Single-chunk prediction (kept for single audio slice compatibility).
    """
    inputs = feature_extractor(
        chunk,
        sampling_rate=sample_rate,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)

    prob_real = probabilities[0][REAL_LABEL].item()
    prob_fake = probabilities[0][FAKE_LABEL].item()

    return prob_real, prob_fake