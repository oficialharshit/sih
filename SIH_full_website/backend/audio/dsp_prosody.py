"""
backend/audio/dsp_prosody.py
Forensic DSP (Spectral & Vocoder artifacts) and Prosodic (Pitch, Jitter, Shimmer, Rhythm) Feature Extractor.
"""
import numpy as np
import librosa


def extract_dsp_and_prosody(audio_chunk, sr=16000):
    """
    Analyzes an audio chunk for vocoder artifacts, pitch micro-tremors (jitter),
    amplitude micro-tremors (shimmer), speech rhythm dynamics, and phase consistency.
    Automatically adapts to narrowband (8kHz PSTN/telephony) vs wideband (16kHz VoIP).
    Runs in ~5-10ms.
    """
    if len(audio_chunk) < 512:
        return {
            "dsp_prosody_score": 0.0,
            "spectral_flatness": 0.0,
            "hf_energy_ratio": 0.0,
            "f0_std": 0.0,
            "jitter_approx": 0.0,
            "shimmer_approx": 0.0,
            "speech_rhythm_ratio": 0.0,
            "phase_inconsistency": 0.0,
            "bandwidth_mode": "narrowband" if sr <= 8000 else "wideband",
            "is_robotic_pitch": False
        }

    # Ensure float32
    if audio_chunk.dtype != np.float32:
        audio_chunk = audio_chunk.astype(np.float32)

    # 1. DSP: Spectral Flatness (vocoder noise vs tone ratio)
    flatness = librosa.feature.spectral_flatness(y=audio_chunk)
    mean_flatness = float(np.mean(flatness))

    # 2. DSP: Frequency Energy Distribution (Adaptive for 8kHz Telephony vs 16kHz HD Audio)
    fft_vals = np.abs(np.fft.rfft(audio_chunk))
    freqs = np.fft.rfftfreq(len(audio_chunk), 1.0 / sr)
    total_energy = np.sum(fft_vals ** 2) + 1e-9

    # Narrowband (8kHz G.711) cutoff at 3200Hz; Wideband (16kHz) cutoff at 4000Hz
    cutoff_hz = 3200 if sr <= 8000 else 4000
    hf_energy = np.sum(fft_vals[freqs > cutoff_hz] ** 2)
    hf_ratio = float(hf_energy / total_energy)

    # 3. DSP: Phase Derivative Inconsistency (Phase Discontinuity in Neural Synthesizers)
    # Neural vocoders (HiFi-GAN, diffusion) often exhibit abrupt phase derivative jumps
    stft = librosa.stft(audio_chunk, n_fft=512, hop_length=256)
    phases = np.angle(stft)
    phase_unwrapped = np.unwrap(phases, axis=0)
    phase_derivative = np.diff(phase_unwrapped, axis=0)
    phase_inconsistency = float(np.std(phase_derivative))

    # 4. Prosody: Pitch (F0) & Jitter approximation
    fmin = 70
    fmax = min(400, int(sr / 2 - 100))
    pitches, magnitudes = librosa.piptrack(y=audio_chunk, sr=sr, fmin=fmin, fmax=fmax)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    valid_pitches = pitch_values[pitch_values > 0]

    if len(valid_pitches) > 10:
        f0_std = float(np.std(valid_pitches))
        pitch_diffs = np.abs(np.diff(valid_pitches))
        jitter_approx = float(np.mean(pitch_diffs) / (np.mean(valid_pitches) + 1e-6))
    else:
        f0_std = 0.0
        jitter_approx = 0.0

    # 5. Prosody: Shimmer approximation (Cycle-to-cycle amplitude micro-tremors)
    rms = librosa.feature.rms(y=audio_chunk)[0]
    valid_rms = rms[rms > 0.005]  # ignore pure silence
    if len(valid_rms) > 5:
        amp_diffs = np.abs(np.diff(valid_rms))
        shimmer_approx = float(np.mean(amp_diffs) / (np.mean(valid_rms) + 1e-6))
    else:
        shimmer_approx = 0.0

    # 6. Prosody: Speech Rhythm (voiced-to-total frame ratio)
    voiced_frames = len(valid_pitches)
    total_frames = len(pitch_values) + 1e-6
    rhythm_ratio = float(min(1.0, voiced_frames / total_frames))

    # 7. Forensic Anomaly Score (0.0 = Real/Natural, 1.0 = High Synthetic Anomaly)
    anomaly_signals = []
    
    # Check A: Vocoder spectral distribution & high-frequency flatness
    flatness_threshold = 0.06 if sr <= 8000 else 0.04
    hf_threshold = 0.35 if sr <= 8000 else 0.28
    if mean_flatness > flatness_threshold or hf_ratio > hf_threshold:
        anomaly_signals.append(0.80)
    else:
        anomaly_signals.append(0.10)

    # Check B: Phase Inconsistency (Diffusion/Neural vocoders typically have phase std > 2.8)
    if phase_inconsistency > 2.85:
        anomaly_signals.append(0.85)
    elif phase_inconsistency > 2.40:
        anomaly_signals.append(0.50)
    else:
        anomaly_signals.append(0.15)

    # Check C: Unnatural pitch smoothness (diffusion & neural TTS models lack human vocal cord micro-tremors)
    if len(valid_pitches) > 10:
        if jitter_approx < 0.012:
            anomaly_signals.append(0.85)  # Suppressed human jitter
        elif jitter_approx < 0.020:
            anomaly_signals.append(0.50)
        else:
            anomaly_signals.append(0.10)
    else:
        anomaly_signals.append(0.20)

    # Check D: Artificial amplitude regularity (synthetic shimmer suppression)
    if len(valid_rms) > 5:
        if shimmer_approx < 0.025:
            anomaly_signals.append(0.80)
        elif shimmer_approx < 0.040:
            anomaly_signals.append(0.45)
        else:
            anomaly_signals.append(0.10)
    else:
        anomaly_signals.append(0.20)

    dsp_prosody_score = float(np.mean(anomaly_signals))

    return {
        "dsp_prosody_score": round(dsp_prosody_score, 3),
        "spectral_flatness": round(mean_flatness, 4),
        "hf_energy_ratio": round(hf_ratio, 3),
        "phase_inconsistency": round(phase_inconsistency, 3),
        "bandwidth_mode": "narrowband" if sr <= 8000 else "wideband",
        "f0_std": round(f0_std, 2),
        "jitter_approx": round(jitter_approx, 4),
        "shimmer_approx": round(shimmer_approx, 4),
        "speech_rhythm_ratio": round(rhythm_ratio, 3),
        "is_robotic_pitch": bool(jitter_approx < 0.008 and len(valid_pitches) > 10)
    }