"""
backend/analysis/streaming.py
Real-Time Audio Stream Session Manager.
Maintains rolling window of audio chunks, executes low-latency acoustic (Wav2Vec2)
and DSP/Prosody feature analysis, tracks continuous threat scores, and verifies claimed speaker identity.
"""
import numpy as np
import logging
from typing import Optional, List, Dict, Any
from model.detector import predict_chunks_batched
from audio.dsp_prosody import extract_dsp_and_prosody
from analysis.speaker_verifier import verify_speaker, detect_claimed_identity_from_text

logger = logging.getLogger("streaming_detector")

TARGET_SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 2.0
CHUNK_SAMPLES = int(TARGET_SAMPLE_RATE * CHUNK_DURATION_SEC)  # 32,000 samples
WINDOW_SIZE = 5  # Keep last 5 chunks (~10s) in rolling buffer


class AudioStreamSession:
    """
    Manages a live WebSocket audio stream for a single caller session.
    Buffers incoming PCM Float32 audio, segments into 2-second windows,
    and calculates rolling confidence scores in real-time.
    """
    def __init__(self, claimed_identity: Optional[str] = None):
        self.claimed_identity = claimed_identity
        self.audio_buffer: List[float] = []
        self.chunk_history: List[Dict[str, Any]] = []
        self.total_samples_received: int = 0
        self.chunk_counter: int = 0

    def add_pcm_samples(self, new_samples: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Appends raw PCM float32 samples to the stream buffer.
        If at least 1 chunk (32,000 samples = 2 sec) is accumulated,
        runs analysis and returns the updated rolling metrics.
        """
        if len(new_samples) == 0:
            return None

        # Ensure float32 1D array
        if new_samples.dtype != np.float32:
            new_samples = new_samples.astype(np.float32)

        self.audio_buffer.extend(new_samples.tolist())
        self.total_samples_received += len(new_samples)

        # Process if we have accumulated at least 2.0 seconds of audio
        if len(self.audio_buffer) >= CHUNK_SAMPLES:
            chunk = np.array(self.audio_buffer[:CHUNK_SAMPLES], dtype=np.float32)
            # Advance buffer by 50% overlap (1 second step = 16,000 samples) for responsive sliding window
            step = int(CHUNK_SAMPLES // 2)
            self.audio_buffer = self.audio_buffer[step:]

            self.chunk_counter += 1
            return self._analyze_chunk(chunk)

        return None

    def _analyze_chunk(self, chunk: np.ndarray) -> Dict[str, Any]:
        """
        Executes fast Wav2Vec2 + DSP/Prosody analysis on the current chunk.
        Runs in ~20-30ms.
        """
        # 1. Wav2Vec2 batched inference
        batch_results = predict_chunks_batched([chunk], TARGET_SAMPLE_RATE)
        prob_real, prob_fake = batch_results[0]

        # 2. Forensic DSP & Prosody
        dsp_metrics = extract_dsp_and_prosody(chunk, TARGET_SAMPLE_RATE)
        dsp_anomaly = dsp_metrics.get("dsp_prosody_score", 0.0)

        # 3. Asymmetric fusion
        if prob_fake >= 0.50:
            fused_fake = max(prob_fake, 0.70 * prob_fake + 0.30 * dsp_anomaly)
        elif dsp_anomaly >= 0.50:
            fused_fake = max(prob_fake, 0.40 * prob_fake + 0.60 * dsp_anomaly)
        else:
            fused_fake = 0.70 * prob_fake + 0.30 * dsp_anomaly

        fused_fake = round(min(1.0, max(0.0, float(fused_fake))), 4)
        fused_real = round(1.0 - fused_fake, 4)

        # 4. Speaker Biometric Verification (if claimed or auto-detected)
        speaker_res = None
        target_id = self.claimed_identity
        if target_id and target_id.lower() not in ("none", "general", ""):
            # If a caller ID hint was passed (e.g. 'caller_id:+919820012345' or name)
            if target_id.lower().startswith("caller_id:"):
                hint = target_id.split(":", 1)[1]
                auto_spk = detect_claimed_identity_from_text("", caller_id_hint=hint)
                if auto_spk:
                    target_id = auto_spk["id"]
                    speaker_res = verify_speaker(chunk, TARGET_SAMPLE_RATE, target_id)
                    if speaker_res:
                        speaker_res["detection_source"] = auto_spk["detection_source"]
            else:
                speaker_res = verify_speaker(chunk, TARGET_SAMPLE_RATE, target_id)

        chunk_data = {
            "chunk_id": self.chunk_counter,
            "prob_fake": fused_fake,
            "prob_real": fused_real,
            "raw_w2v_fake": round(float(prob_fake), 4),
            "dsp_anomaly": round(float(dsp_anomaly), 4),
            "speaker_verification": speaker_res
        }

        # Keep rolling window of last WINDOW_SIZE chunks
        self.chunk_history.append(chunk_data)
        if len(self.chunk_history) > WINDOW_SIZE:
            self.chunk_history.pop(0)

        # 5. Compute Rolling Window Consensus Metrics
        return self._compute_rolling_verdict(chunk_data)

    def _compute_rolling_verdict(self, current_chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates the rolling window threat index across the last 5 chunks (10 seconds).
        """
        fakes = [c["prob_fake"] for c in self.chunk_history]
        window_avg_fake = sum(fakes) / len(fakes)
        strong_fake_count = sum(1 for f in fakes if f >= 0.70)
        strong_fake_ratio = strong_fake_count / len(fakes)

        # Fast alert threshold logic
        if strong_fake_ratio >= 0.40 or (current_chunk["prob_fake"] >= 0.80 and window_avg_fake >= 0.50):
            risk = "HIGH"
            threat_status = "SYNTHETIC_CLONE_DETECTED"
            threat_level = max(window_avg_fake, current_chunk["prob_fake"])
        elif window_avg_fake >= 0.45 or current_chunk["prob_fake"] >= 0.65:
            risk = "MEDIUM"
            threat_status = "SUSPICIOUS_AUDIO_ANOMALY"
            threat_level = window_avg_fake
        else:
            risk = "LOW"
            threat_status = "AUTHENTIC_SPEECH"
            threat_level = min(window_avg_fake, 0.35)

        threat_level = round(float(threat_level), 4)

        # Determine speaker biometric status
        speaker_verdict = "NOT_CLAIMED"
        spk = current_chunk.get("speaker_verification")
        if spk and spk.get("status"):
            if risk == "HIGH":
                speaker_verdict = "SYNTHETIC_IMPERSONATION_ATTACK"
            elif not spk.get("is_matched", False):
                speaker_verdict = "HUMAN_IMPERSONATOR_ALERT"
            else:
                speaker_verdict = "AUTHENTIC_EXECUTIVE_CONFIRMED"

        return {
            "type": "STREAM_CHUNK_ANALYSIS",
            "chunk_id": current_chunk["chunk_id"],
            "current_chunk_fake": current_chunk["prob_fake"],
            "current_chunk_real": current_chunk["prob_real"],
            "rolling_threat_index": threat_level,
            "rolling_risk": risk,
            "threat_status": threat_status,
            "speaker_verdict": speaker_verdict,
            "speaker_verification": spk,
            "history_window": [
                {"chunk_id": c["chunk_id"], "prob_fake": c["prob_fake"]}
                for c in self.chunk_history
            ]
        }
