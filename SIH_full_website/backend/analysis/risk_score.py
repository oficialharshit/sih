"""
Unified Multi-Modal Risk Scoring Engine with Contextual Enrichment.
Combines Voice Deepfake Probability + 50-50 Ensembled NLP Scam Probability +
Contextual Telemetry (Call Origin, Transaction Criticality, Caller History Trust).
"""
from typing import Optional, Dict, Any, Tuple


def calculate_risk(
    prob_fake: float,
    scam_probability: float,
    context: Optional[Dict[str, Any]] = None
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Calculates unified risk score:
    - Base Acoustic & Intent Fusion:
      - 70% Voice Deepfake Model
      - 30% Combined NLP Scam Model (50% Semantic + 50% DistilBERT)
    - Contextual Risk Enrichment:
      - Call Origin / SIP Trunk reputation
      - Transaction Criticality (High-value approvals: RTGS, privileged password reset)
      - Caller History Trust Level
    """
    base_score = (
        0.70 * prob_fake +
        0.30 * scam_probability
    )

    context = context or {}
    call_origin_risk = float(context.get("call_origin_risk", 0.10))  # 0.0=internal PBX, 1.0=unverified VoIP
    txn_amount = float(context.get("transaction_amount_inr", 0.0))
    caller_trust = float(context.get("caller_trust", 0.50))  # 0.0=unknown/suspicious, 1.0=trusted regular caller
    is_privileged = bool(context.get("is_privileged_action", False))

    # Calculate Transaction Criticality Factor
    if txn_amount >= 1000000.0 or is_privileged:  # >= 10 Lakhs or admin credentials
        txn_factor = 0.25
    elif txn_amount >= 100000.0:  # >= 1 Lakh
        txn_factor = 0.15
    elif txn_amount > 0.0:
        txn_factor = 0.05
    else:
        txn_factor = 0.0

    # Caller Trust Attenuation: High trust reduces false positive escalation slightly,
    # Low trust with unknown origin elevates risk
    origin_factor = call_origin_risk * 0.15
    trust_discount = (caller_trust - 0.50) * 0.10  # -0.05 to +0.05

    # Contextual Multiplier
    context_adjustment = txn_factor + origin_factor - trust_discount
    final_score = max(0.0, min(1.0, base_score + context_adjustment))

    # Threshold-based dynamic risk classification
    if final_score < 0.35:
        risk = "LOW"
    elif final_score < 0.68:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    context_breakdown = {
        "base_score": round(base_score, 4),
        "context_adjustment": round(context_adjustment, 4),
        "txn_factor": round(txn_factor, 3),
        "origin_risk": round(call_origin_risk, 3),
        "caller_trust": round(caller_trust, 3),
        "final_threat_index": round(final_score, 4)
    }

    return risk, round(final_score, 4), context_breakdown