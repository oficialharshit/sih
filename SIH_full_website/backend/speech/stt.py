"""
Ultra-Fast Speech-to-Text using faster-whisper with Distil-Whisper or Tiny.
Transcribes audio in ~30-50 milliseconds.
"""

import torch
import numpy as np
from faster_whisper import WhisperModel

# Detect device & precision
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

# ⚡ Use multilingual Whisper model "small" (244M params) for maximum accuracy on Hindi, Hinglish & English
MODEL_SIZE = "small"

model = WhisperModel(
    MODEL_SIZE,
    device=device,
    compute_type=compute_type
)


def transcribe_audio(audio, sample_rate=16000, language=None):
    """
    Multilingual transcription using Whisper-Small supporting Hindi, English, and Hinglish.
    Allows language hints ('auto', 'hi', 'en') with contextual priming prompts.
    """
    try:
        if isinstance(audio, np.ndarray):
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

        target_lang = None
        prompt = "Bhai, turant paise transfer kar de, OTP batao, police case ban jayega, digital arrest, account band, jaldi bhejo."

        if language:
            lang_code = language.strip().lower()
            if lang_code in ("hi", "hindi", "hinglish"):
                target_lang = "hi"
                prompt = "नमस्ते भाई, तुरंत पैसे ट्रांसफर कर दे, ओटीपी बताओ, पुलिस केस बन जाएगा, खाता ब्लॉक, सीबीआई ऑफिसर, जल्दी भेजो।"
            elif lang_code in ("en", "english"):
                target_lang = "en"
                prompt = "Urgent wire transfer, bank account verification, password, police arrest warrant, immediate funds."

        # beam_size=3 provides high beam search decoding accuracy
        segments, info = model.transcribe(
            audio,
            beam_size=3,
            language=target_lang,
            vad_filter=True,
            initial_prompt=prompt
        )

        transcription = " ".join([segment.text for segment in segments]).strip()
        return transcription if transcription else "No speech detected."

    except Exception as e:
        print(f"[STT Error] Transcription failed: {e}")
        return ""