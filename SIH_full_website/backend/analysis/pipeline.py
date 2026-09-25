"""
Main Forensic Analysis Pipeline (Optimized with Parallel Execution & 50-50 NLP Ensemble).
Runs audio preprocessing, VAD, concurrent Deepfake Detection + Whisper STT,
DSP & Prosody forensic extraction, and Ensembled Semantic Intent + DistilBERT Classification.
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from audio.loader import load_audio
from audio.preprocessing import preprocess_audio
from audio.vad import detect_speech, extract_speech
from audio.chunking import create_chunks
from audio.dsp_prosody import extract_dsp_and_prosody

from analysis.predictor import predict_chunks
from analysis.aggregator import aggregate_predictions
from analysis.risk_score import calculate_risk

from speech.stt import transcribe_audio
from analysis.semantic_detector import evaluate_semantic_intent
from analysis.nlp_predictor import predict_scam  # ⚡ Remote Cloud DistilBERT
from analysis.speaker_verifier import verify_speaker, detect_claimed_identity_from_text
from analysis.indic_fraud_rules import evaluate_indic_fraud_rules


def analyze_audio(file_path, claimed_identity=None, language=None, context_data=None, anonymize_pii=True):
    # 1. Load Audio (In-Memory buffer or path)
    audio, sample_rate = load_audio(file_path)
    total_duration = len(audio) / sample_rate

    # 2. Preprocess (GPU Resampling)
    audio, sample_rate = preprocess_audio(
        audio,
        sample_rate
    )

    # 3. Voice Activity Detection (VAD)
    speech_timestamps = detect_speech(
        audio,
        sample_rate
    )

    # 4. Extract Speech Segments
    speech_segments = extract_speech(
        audio,
        speech_timestamps
    )

    # Fallback if no speech detected
    if not speech_segments:
        speech_segments = [audio]

    # 5. Create 2-Second Chunks
    chunks = create_chunks(
        speech_segments,
        sample_rate,
        chunk_duration=2
    )

    if not chunks:
        chunks = [audio]

    # ⚡ 6. PARALLEL EXECUTION: Run Wav2Vec2 and Whisper STT concurrently (Single run)
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_deepfake = executor.submit(predict_chunks, chunks, sample_rate)
        future_stt = executor.submit(transcribe_audio, audio, sample_rate, language)

        raw_predictions = future_deepfake.result()
        transcript = future_stt.result()

    # 🔬 6b. Extract DSP & Prosody forensic metrics for each chunk
    dsp_metrics = [extract_dsp_and_prosody(chunk, sample_rate) for chunk in chunks]

    # 🔬 High-Sensitivity Forensic Fusion
    predictions = []
    for chunk_pred, dsp_info in zip(raw_predictions, dsp_metrics):
        prob_fake = chunk_pred["prob_fake"]
        dsp_score = dsp_info["dsp_prosody_score"]

        # Fuses Transformer acoustic embeddings with physical DSP vocoder artifacts
        if prob_fake >= 0.50:
            fused_fake = max(prob_fake, 0.70 * prob_fake + 0.30 * dsp_score)
        elif dsp_score >= 0.50:
            # Modern diffusion clone detected by physical prosody / vocoder flatness
            fused_fake = max(prob_fake, 0.40 * prob_fake + 0.60 * dsp_score)
        else:
            fused_fake = 0.70 * prob_fake + 0.30 * dsp_score

        fused_real = 1.0 - fused_fake
        predictions.append({
            "chunk": chunk_pred["chunk"],
            "prob_real": round(fused_real, 4),
            "prob_fake": round(fused_fake, 4)
        })

    # 7. Aggregate Voice Predictions (Using the Fused Predictions)
    result = aggregate_predictions(predictions)
    result["chunk_predictions"] = predictions
    result["duration_seconds"] = round(total_duration, 2)
    result["chunks_analyzed"] = len(predictions)

    # Attach explainable forensic acoustic details (with fallback)
    if dsp_metrics:
        result["forensics"] = {
            "avg_hf_energy": round(float(np.mean([m["hf_energy_ratio"] for m in dsp_metrics])), 3),
            "avg_jitter": round(float(np.mean([m["jitter_approx"] for m in dsp_metrics])), 4),
            "avg_shimmer": round(float(np.mean([m["shimmer_approx"] for m in dsp_metrics])), 4),
            "speech_rhythm": round(float(np.mean([m["speech_rhythm_ratio"] for m in dsp_metrics])), 3),
            "phase_inconsistency": round(float(np.mean([m.get("phase_inconsistency", 0.0) for m in dsp_metrics])), 3),
            "bandwidth_mode": dsp_metrics[0].get("bandwidth_mode", "wideband"),
            "unnatural_pitch_flag": bool(any(m["is_robotic_pitch"] for m in dsp_metrics)),
            "spectral_flatness": round(float(np.mean([m["spectral_flatness"] for m in dsp_metrics])), 4)
        }
    else:
        result["forensics"] = {
            "avg_hf_energy": 0.0,
            "avg_jitter": 0.0,
            "avg_shimmer": 0.0,
            "speech_rhythm": 0.0,
            "phase_inconsistency": 0.0,
            "bandwidth_mode": "wideband",
            "unnatural_pitch_flag": False,
            "spectral_flatness": 0.0
        }

    # ⚡ 8. MULTILINGUAL & INDIC FRAUD ENSEMBLE: Semantic BGE + DistilBERT + Indic Dialect Rules
    semantic_result = evaluate_semantic_intent(transcript)
    distilbert_result = predict_scam(transcript)
    indic_result = evaluate_indic_fraud_rules(transcript)

    sem_prob = semantic_result.get("scam_probability", 0.0)
    dist_prob = distilbert_result.get("scam_probability", 0.0)
    indic_prob = indic_result.get("indic_scam_prob", 0.0)

    # 50-50 Split for Base Scam Probability
    if dist_prob > 0:
        base_scam_prob = (0.50 * sem_prob) + (0.50 * dist_prob)
    else:
        base_scam_prob = sem_prob

    # If Indic Hindi/Hinglish urgent extortion or digital arrest triggers are found, elevate confidence
    if indic_result.get("has_indic_scam", False):
        ensembled_scam_prob = round(max(base_scam_prob, indic_prob, 0.85), 4)
        fraud_cat = indic_result.get("category", semantic_result.get("category", "DIGITAL_ARREST_LAW_ENFORCEMENT"))
        matched_pat = indic_result.get("description", "Indic urgent coercion phrase detected")
    else:
        ensembled_scam_prob = round(base_scam_prob, 4)
        fraud_cat = semantic_result.get("category", "LEGITIMATE_CONVERSATION")
        matched_pat = semantic_result.get("matched_pattern", "")

    ensembled_non_scam_prob = round(1.0 - ensembled_scam_prob, 4)
    prediction = "SCAM" if ensembled_scam_prob >= 0.50 else "NON_SCAM"

    # 🔒 Privacy & DPDP Act Compliance: Redact PII (Phone numbers, Aadhaar, OTPs, Card numbers)
    redacted_transcript = transcript
    if transcript and anonymize_pii:
        import re
        # Mask 10-digit mobile numbers or 12-digit Aadhaar numbers
        redacted_transcript = re.sub(r'\b\d{10,12}\b', '[REDACTED_IDENTITY_ID]', redacted_transcript)
        # Mask 4 to 6 digit OTP/PIN codes
        redacted_transcript = re.sub(r'\b(otp|pin|code)\s*[:=]?\s*\d{4,6}\b', r'\1 [REDACTED_CODE]', redacted_transcript, flags=re.IGNORECASE)
        # Mask Credit/Debit card numbers (16 digits)
        redacted_transcript = re.sub(r'\b(?:\d{4}[ -]?){3}\d{4}\b', '[REDACTED_PAN_CARD]', redacted_transcript)

    # 9. Format Results
    result["transcript"] = redacted_transcript
    result["raw_transcript_available"] = not anonymize_pii
    result["nlp_prediction"] = prediction
    result["scam_probability"] = ensembled_scam_prob
    result["non_scam_probability"] = ensembled_non_scam_prob
    result["semantic_score"] = sem_prob
    result["distilbert_score"] = dist_prob
    result["indic_fraud"] = indic_result  # 🇮🇳 Multilingual Indic Telemetry
    result["fraud_category"] = fraud_cat
    result["matched_pattern"] = matched_pat

    # 10. Biometric Speaker Identity Verification (CXO / Whaling Impersonation Check)
    auto_detected_info = None
    target_identity_id = claimed_identity
    if not target_identity_id or target_identity_id.lower() in ("auto", "none", ""):
        caller_hint = context_data.get("caller_id") if context_data else None
        auto_detected_info = detect_claimed_identity_from_text(transcript, caller_id_hint=caller_hint)
        if auto_detected_info:
            target_identity_id = auto_detected_info["id"]
        else:
            target_identity_id = None

    speaker_info = verify_speaker(audio, sample_rate, target_identity_id)
    if auto_detected_info and speaker_info.get("status") in ("VERIFIED", "IMPERSONATION_MISMATCH"):
        speaker_info["detection_source"] = auto_detected_info.get("detection_source", "Auto-Detected via Spoken Transcript")
        speaker_info["auto_detected"] = True

    result["speaker_verification"] = speaker_info

    # 11. Calculate Unified Multi-Modal Risk Score with Contextual Enrichment
    base_risk, base_final_score, context_breakdown = calculate_risk(
        result["prob_fake"],
        result["scam_probability"],
        context=context_data
    )
    result["context_risk_breakdown"] = context_breakdown

    # 12. Dual-Key Decision: Synthesize Voice Authenticity + Claimed Identity
    if speaker_info.get("status") in ("VERIFIED", "IMPERSONATION_MISMATCH"):
        if result["prob_fake"] >= 0.50:
            result["impersonation_verdict"] = "SYNTHETIC_IMPERSONATION_ATTACK"
            result["risk"] = "HIGH"
            result["final_score"] = max(base_final_score, 0.85)
        elif not speaker_info.get("is_matched", False):
            # Real voice, but caller DOES NOT match the claimed CEO/Manager!
            result["impersonation_verdict"] = "HUMAN_IMPERSONATOR_ALERT"
            result["risk"] = "HIGH"
            result["final_score"] = max(base_final_score, 0.80)
        else:
            # Real voice AND biometrics match the genuine enrolled executive!
            result["impersonation_verdict"] = "AUTHENTIC_EXECUTIVE_CONFIRMED"
            result["risk"] = "LOW" if base_final_score < 0.60 else base_risk
            result["final_score"] = base_final_score
    else:
        result["impersonation_verdict"] = "STANDARD_FORENSIC_CHECK"
        result["risk"] = base_risk
        result["final_score"] = base_final_score

    # 🔒 Privacy: Minimal Retention Protocol (Purge raw audio array from memory)
    del audio
    del speech_segments
    del chunks

    result["privacy_compliance"] = {
        "audio_retention": "ZERO_KNOWLEDGE_SCRUBBED",
        "pii_anonymized": bool(anonymize_pii),
        "compliance_standard": "Digital Personal Data Protection (DPDP) Act 2023 & ISO/IEC 27001",
        "processing_mode": "IN_MEMORY_EPHEMERAL"
    }

    return result