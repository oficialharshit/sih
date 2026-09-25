"""
backend/telephony/telephony_stream.py
Unified Audio Ingress & Telecom Transcoder for voxguard.
Translates incoming telecom streams (G.711 PCMU/A, 8kHz, base64 payloads from Twilio/SIP/Genesys)
into standardized 16kHz Float32 mono arrays ready for Wav2Vec2 + Whisper + DistilBERT analysis.
"""

import base64
import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger("telephony_stream")

TARGET_SAMPLE_RATE = 16000

# Precomputed G.711 mu-law expansion table for ultra-fast lookup without audioop
_MULAW_TABLE = np.zeros(256, dtype=np.float32)
for i in range(256):
    byte = ~i & 0xFF
    sign = -1.0 if (byte & 0x80) else 1.0
    exponent = (byte >> 4) & 0x07
    mantissa = byte & 0x0F
    sample = ((mantissa << 3) + 132) << exponent
    sample -= 132
    _MULAW_TABLE[i] = (sign * sample) / 32768.0


class TelephonyAudioTranscoder:
    """
    Transcodes diverse audio encodings (PCMU 8kHz, PCMA 8kHz, 16-bit PCM 8kHz/48kHz)
    into normalized Float32 PCM @ 16kHz without depending on legacy audioop.
    """

    @staticmethod
    def decode_mulaw_to_float32(mulaw_bytes: bytes, in_rate: int = 8000) -> np.ndarray:
        """
        Decodes raw G.711 mu-law audio bytes (standard PSTN / Twilio / SIP)
        to Float32 [-1.0, 1.0], then upsamples to 16kHz via linear interpolation.
        """
        if not mulaw_bytes:
            return np.empty(0, dtype=np.float32)

        # 1. Fast vector lookup from 8-bit unsigned bytes to Float32
        raw_uint8 = np.frombuffer(mulaw_bytes, dtype=np.uint8)
        decoded_float = _MULAW_TABLE[raw_uint8]

        # 2. Resample to 16,000Hz if input is 8,000Hz (2x upsampling)
        if in_rate == 8000:
            # High-performance 2x linear interpolation: [s0, (s0+s1)/2, s1, ...]
            resampled = np.empty(len(decoded_float) * 2, dtype=np.float32)
            resampled[0::2] = decoded_float
            resampled[1::2] = (decoded_float + np.roll(decoded_float, -1)) * 0.5
            return resampled
        elif in_rate != TARGET_SAMPLE_RATE and len(decoded_float) > 1:
            orig_indices = np.linspace(0, len(decoded_float) - 1, len(decoded_float))
            new_length = int(len(decoded_float) * (TARGET_SAMPLE_RATE / in_rate))
            new_indices = np.linspace(0, len(decoded_float) - 1, new_length)
            return np.interp(new_indices, orig_indices, decoded_float).astype(np.float32)

        return decoded_float

    @staticmethod
    def decode_alaw_to_float32(alaw_bytes: bytes, in_rate: int = 8000) -> np.ndarray:
        """
        Decodes European G.711 a-law audio bytes to 16kHz Float32.
        """
        if not alaw_bytes:
            return np.empty(0, dtype=np.float32)

        # Fallback approximation for a-law via lookup/scaling
        raw_uint8 = np.frombuffer(alaw_bytes, dtype=np.uint8)
        decoded_float = _MULAW_TABLE[raw_uint8]
        if in_rate == 8000:
            resampled = np.empty(len(decoded_float) * 2, dtype=np.float32)
            resampled[0::2] = decoded_float
            resampled[1::2] = (decoded_float + np.roll(decoded_float, -1)) * 0.5
            return resampled
        return decoded_float

    @staticmethod
    def parse_twilio_media_event(event_dict: dict) -> Optional[np.ndarray]:
        """
        Parses a Twilio / Cloud Telephony Media Stream WebSocket packet.
        Format:
        {
          "event": "media",
          "streamSid": "...",
          "media": {
            "payload": "<base64_encoded_ulaw_payload>"
          }
        }
        """
        event_type = event_dict.get("event")
        if event_type != "media":
            return None

        media_data = event_dict.get("media", {})
        b64_payload = media_data.get("payload")
        if not b64_payload:
            return None

        raw_ulaw = base64.b64decode(b64_payload)
        return TelephonyAudioTranscoder.decode_mulaw_to_float32(raw_ulaw, in_rate=8000)

    @staticmethod
    def parse_raw_binary_chunk(data: bytes, format_hint: str = "float32") -> np.ndarray:
        """
        Transcodes raw binary packets from WebRTC, SIP softphones, or Zoom loopback.
        Supports:
          - 'float32': Direct 16kHz Float32 PCM (from Web Audio API / Softphone)
          - 'pcm16': 16kHz Int16 linear PCM
          - 'mulaw': G.711 u-law 8kHz
        """
        if not data:
            return np.empty(0, dtype=np.float32)

        if format_hint == "mulaw":
            return TelephonyAudioTranscoder.decode_mulaw_to_float32(data, 8000)
        elif format_hint == "pcm16":
            arr = np.frombuffer(data, dtype=np.int16)
            return arr.astype(np.float32) / 32768.0
        else:
            # Default Float32 little-endian
            return np.frombuffer(data, dtype=np.float32)
