"""
NLP Scam Detector using Hugging Face Cloud Inference API.
Model: extreme1122/scam-distilbert
"""

import os
from typing import Dict, Any, Optional
from huggingface_hub import InferenceClient

MODEL_ID = "extreme1122/scam-distilbert"

EMPTY_OR_NOISE_TEXTS = {
    "no speech detected.",
    "no clear speech detected.",
}

_client: Optional[InferenceClient] = None
HF_TOKEN = os.getenv("HF_TOKEN")

def get_inference_client() -> Optional[InferenceClient]:
    """Lazy initialize and return the Hugging Face InferenceClient."""
    global _client
    if _client is None:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            _client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
    return _client


def predict_scam(text: str) -> Dict[str, Any]:
    """
    Predicts scam probability by sending text to the Hugging Face Cloud API.
    """
    cleaned_text = (text or "").strip()

    # Skip empty/noise text
    if not cleaned_text or cleaned_text.lower() in EMPTY_OR_NOISE_TEXTS:
        return {
            "prediction": "NON_SCAM",
            "scam_probability": 0.0,
            "non_scam_probability": 1.0,
        }

    client = get_inference_client()
    if not client:
        # Fallback if HF_TOKEN is not configured
        return {
            "prediction": "NON_SCAM",
            "scam_probability": 0.0,
            "non_scam_probability": 1.0,
        }

    try:
        # Remote inference call to Hugging Face
        results = client.text_classification(cleaned_text, model=MODEL_ID)

        probabilities = {
            item.label.upper(): float(item.score) for item in results
        }

        scam_prob = 0.0
        non_scam_prob = 0.0

        for label, score in probabilities.items():
            if "SCAM" in label and "NON" not in label:
                scam_prob = score
            elif "NON" in label or "HAM" in label or "LEGIT" in label:
                non_scam_prob = score

        # Handle generic labels (LABEL_0 / LABEL_1)
        if scam_prob == 0.0 and non_scam_prob == 0.0:
            if "LABEL_1" in probabilities:
                scam_prob = probabilities["LABEL_1"]
            if "LABEL_0" in probabilities:
                non_scam_prob = probabilities["LABEL_0"]

        if scam_prob > 0 and non_scam_prob == 0:
            non_scam_prob = 1.0 - scam_prob
        elif non_scam_prob > 0 and scam_prob == 0:
            scam_prob = 1.0 - non_scam_prob

        prediction = "SCAM" if scam_prob >= non_scam_prob else "NON_SCAM"

        return {
            "prediction": prediction,
            "scam_probability": round(scam_prob, 4),
            "non_scam_probability": round(non_scam_prob, 4),
        }

    except Exception as e:
        print(f"[Remote NLP Error] Hugging Face inference failed: {e}")
        return {
            "prediction": "NON_SCAM",
            "scam_probability": 0.0,
            "non_scam_probability": 1.0,
        }