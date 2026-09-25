"""
backend/analysis/speaker_verifier.py
Biometric Speaker Verification Engine for CXO / Target Impersonation Detection.
Extracts a 128-dimensional acoustic vocal tract fingerprint and calculates
cosine similarity against enrolled executive voiceprints.
"""

import os
import json
from typing import Optional, List, Dict
import numpy as np
import librosa

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAULT_DIR = os.path.join(BASE_DIR, "voice_vault")
os.makedirs(VAULT_DIR, exist_ok=True)


def extract_voiceprint(audio, sr=16000):
    """
    Extracts a 128-dimensional biometric vocal fingerprint from audio.
    Combines Mel-Frequency Cepstral Coefficients (MFCCs), Delta dynamics,
    spectral contrast, pitch chroma, and formant energy.
    Each feature group is standardized and normalized independently so no single
    frequency dimension dominates, enabling sharp discrimination between different human voices.
    """
    if len(audio) < 1600:  # Minimum 0.1 sec
        return np.zeros(128, dtype=np.float32)

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    def safe_unit_norm(vec):
        """Helper to safely normalize feature group to unit sphere."""
        n = np.linalg.norm(vec)
        return (vec / n).astype(np.float32) if n > 1e-6 else np.zeros_like(vec, dtype=np.float32)

    # 1. 20-band MFCC (mean + std = 40 dims) -> Core vocal tract acoustic shape
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_block = safe_unit_norm(np.concatenate([mfcc_mean, mfcc_std]))  # 40 dims

    # 2. Delta & Delta-Delta MFCC (temporal speech dynamics = 40 dims)
    mfcc_delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(mfcc_delta, axis=1)
    delta_std = np.std(mfcc_delta, axis=1)
    delta_block = safe_unit_norm(np.concatenate([delta_mean, delta_std]))  # 40 dims

    # 3. Spectral Contrast (vocal tract resonance peaks = 14 dims)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)
    contrast_block = safe_unit_norm(np.concatenate([contrast_mean, contrast_std]))  # 14 dims

    # 4. Chroma / Harmonic Energy (pitch tone distribution = 24 dims)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    chroma_block = safe_unit_norm(np.concatenate([chroma_mean, chroma_std]))  # 24 dims

    # 5. Formant / Centroid dynamics (10 dims - normalized to prevent magnitude explosion)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=audio)
    rms = librosa.feature.rms(y=audio)

    # Scale frequency values down from Hz (0-8000) to normalized units
    spec_features = np.array([
        np.mean(centroid) / (sr / 2.0), np.std(centroid) / (sr / 2.0),
        np.mean(bandwidth) / (sr / 2.0), np.std(bandwidth) / (sr / 2.0),
        np.mean(rolloff) / (sr / 2.0), np.std(rolloff) / (sr / 2.0),
        np.mean(flatness), np.std(flatness),
        np.mean(rms), np.std(rms)
    ], dtype=np.float32)
    spec_block = safe_unit_norm(spec_features)  # 10 dims

    # Weighted concatenation: prioritize MFCC & Deltas (vocal tract signature)
    # 40 (MFCC * 1.5) + 40 (Delta * 1.2) + 14 (Contrast * 1.0) + 24 (Chroma * 0.8) + 10 (Spec * 0.5) = 128 dims
    vector = np.concatenate([
        mfcc_block * 1.5,
        delta_block * 1.2,
        contrast_block * 1.0,
        chroma_block * 0.8,
        spec_block * 0.5
    ]).astype(np.float32)

    # Final L2 normalization to unit sphere for cosine distance
    norm = np.linalg.norm(vector)
    if norm > 1e-6:
        vector = vector / norm
    else:
        vector = np.zeros(128, dtype=np.float32)

    return vector


def enroll_speaker(name: str, audio, sr=16000, role="Executive"):
    """
    Enrolls a new executive/manager into the biometric voice vault.
    """
    cleaned_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()
    if not cleaned_name:
        raise ValueError("Invalid speaker name.")

    vector = extract_voiceprint(audio, sr)
    profile_data = {
        "name": cleaned_name,
        "role": role,
        "vector": vector.tolist(),
        "dimension": len(vector)
    }

    file_path = os.path.join(VAULT_DIR, f"{cleaned_name.replace(' ', '_').lower()}.json")
    with open(file_path, "w") as f:
        json.dump(profile_data, f, indent=2)

    return profile_data


def list_enrolled_speakers():
    """
    Returns a list of all currently enrolled identities in the vault.
    """
    speakers = []
    if not os.path.exists(VAULT_DIR):
        return speakers

    for fname in os.listdir(VAULT_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(VAULT_DIR, fname), "r") as f:
                    data = json.load(f)
                    speakers.append({
                        "id": os.path.splitext(fname)[0],
                        "name": data.get("name", fname),
                        "role": data.get("role", "Authorized Individual")
                    })
            except Exception:
                pass
    return speakers


def detect_claimed_identity_from_text(transcript: str, caller_id_hint: str = None) -> Optional[dict]:
    """
    Auto-Detect Claimed Identity from:
    1. Caller ID / PBX Phone Number / SIP Header metadata hint.
    2. Spoken conversational introduction in the transcript (e.g., 'Hello, this is Rajesh Sharma',
       'I am Rajesh', 'speaking on behalf of Rajesh Sharma', 'Priya Nair from finance').
    
    Returns:
        dict: {"id": profile_id, "name": name, "role": role, "detection_source": source} or None
    """
    enrolled = list_enrolled_speakers()
    if not enrolled:
        return None

    # 1. Check Caller ID / Telecom hint first (Highest priority if provided)
    if caller_id_hint:
        hint_clean = caller_id_hint.lower().strip()
        for spk in enrolled:
            spk_id = spk["id"].lower()
            spk_name = spk["name"].lower()
            # Match phone number alias, full name, or first/last name in caller ID string
            name_tokens = spk_name.split()
            if spk_id in hint_clean or spk_name in hint_clean or any(t in hint_clean for t in name_tokens):
                return {**spk, "detection_source": f"PBX / Caller ID Match ('{caller_id_hint}')"}

    # 2. Check Spoken Transcript (NLP Introduction Extraction)
    if not transcript:
        return None

    text_lower = transcript.lower()

    # Introduction conversational triggers
    triggers = [
        "this is", "i am", "i'm", "speaking", "here", "it's", "call from",
        "on behalf of", "representing", "ceo", "cfo", "director"
    ]

    for spk in enrolled:
        name_parts = spk["name"].lower().split()  # e.g., ["rajesh", "sharma"]
        first_name = name_parts[0]
        full_name = spk["name"].lower()

        # Check full name mention: e.g. "this is rajesh sharma" or "rajesh sharma here"
        if full_name in text_lower:
            return {
                **spk,
                "detection_source": f"Spoken Self-Identification ('{spk['name']}') in live transcript"
            }

        # Check first name preceded or followed by an introduction trigger
        for trigger in triggers:
            if f"{trigger} {first_name}" in text_lower or f"{first_name} {trigger}" in text_lower:
                return {
                    **spk,
                    "detection_source": f"Spoken Intro Trigger ('{trigger} {first_name.capitalize()}')"
                }

    # Check for role triggers: e.g., "this is the ceo", "chief executive"
    if "ceo" in text_lower or "chief executive" in text_lower:
        for spk in enrolled:
            if "ceo" in spk.get("role", "").lower() or "chief executive" in spk.get("role", "").lower():
                return {**spk, "detection_source": "Spoken Executive Title ('CEO') in transcript"}

    if "cfo" in text_lower or "finance officer" in text_lower or "treasurer" in text_lower:
        for spk in enrolled:
            if "cfo" in spk.get("role", "").lower() or "financial" in spk.get("role", "").lower():
                return {**spk, "detection_source": "Spoken Executive Title ('CFO') in transcript"}

    return None


def verify_speaker(audio, sr=16000, claimed_identity_id=None):
    """
    Compares incoming audio against a claimed identity profile.
    
    Returns:
        dict: {
            "match_score": float (0.0 to 1.0),
            "is_matched": bool,
            "claimed_name": str,
            "status": "VERIFIED" | "IMPERSONATION_MISMATCH" | "NO_CLAIMANT"
        }
    """
    if not claimed_identity_id or claimed_identity_id.lower() in ("none", "general", ""):
        return {
            "match_score": 0.0,
            "is_matched": False,
            "claimed_name": "None (Generic Deepfake Check)",
            "status": "NO_CLAIMANT"
        }

    profile_path = os.path.join(VAULT_DIR, f"{claimed_identity_id}.json")
    if not os.path.exists(profile_path):
        # Fallback check by name
        for fname in os.listdir(VAULT_DIR):
            if fname.lower().startswith(claimed_identity_id.lower()):
                profile_path = os.path.join(VAULT_DIR, fname)
                break

    if not os.path.exists(profile_path):
        return {
            "match_score": 0.0,
            "is_matched": False,
            "claimed_name": claimed_identity_id,
            "status": "PROFILE_NOT_FOUND"
        }

    try:
        with open(profile_path, "r") as f:
            target_profile = json.load(f)

        target_vector = np.array(target_profile["vector"], dtype=np.float32)
        query_vector = extract_voiceprint(audio, sr)

        # Cosine similarity (-1.0 to 1.0)
        norm_q = np.linalg.norm(query_vector)
        norm_t = np.linalg.norm(target_vector)

        if norm_q > 1e-6 and norm_t > 1e-6:
            cosine_sim = float(np.dot(query_vector, target_vector) / (norm_q * norm_t))
        else:
            cosine_sim = 0.0

        # Realistic Cosine Similarity calibration for normalized vocal fingerprints:
        # Cross-speaker cosine similarity typically ranges from 0.20 to 0.65
        # Same-speaker cosine similarity typically ranges from 0.78 to 0.98
        # We use an S-curve centered at 0.72 with steep discrimination
        if cosine_sim <= 0.50:
            # Clear mismatch: map [0.0, 0.50] -> [0.0, 0.25]
            calibrated_score = float(max(0.0, cosine_sim * 0.50))
        elif cosine_sim < 0.72:
            # Transition / Borderline region: map [0.50, 0.72] -> [0.25, 0.55]
            calibrated_score = float(0.25 + (cosine_sim - 0.50) / 0.22 * 0.30)
        else:
            # High-confidence genuine match: map [0.72, 1.0] -> [0.55, 1.0]
            calibrated_score = float(min(1.0, 0.55 + (cosine_sim - 0.72) / 0.28 * 0.45))

        # Threshold: >= 60% calibrated score indicates the same person
        is_matched = calibrated_score >= 0.60
        status = "VERIFIED" if is_matched else "IMPERSONATION_MISMATCH"

        return {
            "match_score": round(calibrated_score, 4),
            "raw_cosine": round(cosine_sim, 4),
            "is_matched": is_matched,
            "claimed_name": target_profile.get("name", claimed_identity_id),
            "role": target_profile.get("role", "Authorized Individual"),
            "status": status
        }

    except Exception as e:
        return {
            "match_score": 0.0,
            "is_matched": False,
            "claimed_name": claimed_identity_id,
            "status": f"ERROR: {str(e)}"
        }
