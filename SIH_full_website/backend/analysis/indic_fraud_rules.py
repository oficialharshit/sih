"""
backend/analysis/indic_fraud_rules.py
Specialized Multilingual, Hindi & Hinglish Fraud & Coercion Rules Engine.
Detects high-urgency financial fraud, digital arrest, KYC suspension, and credential
exfiltration patterns across Hindi, Hinglish (Romanized Hindi), and Indic regional dialects.
"""
import re
from typing import Dict, Any, List

# 1. High-Confidence Hindi / Hinglish Urgent Scam Patterns
INDIC_SCAM_PATTERNS = [
    # Category 1: Digital Arrest & Police Coercion (CBI, Police, Customs, Court, FIR)
    {
        "category": "DIGITAL_ARREST_LAW_ENFORCEMENT",
        "weight": 0.95,
        "regex": r"\b(police\s*case|digital\s*arrest|arrest\s*warrant|cbi\s*officer|cyber\s*crime|fir\s*darj|jail\s*bhej|court\s*summons|narcotics\s*parcel|customs\s*seized|illegal\s*drugs|thana|chalan)\b",
        "description": "Digital arrest or police coercion threat in Hindi/Hinglish"
    },
    {
        "category": "DIGITAL_ARREST_LAW_ENFORCEMENT",
        "weight": 0.90,
        "regex": r"\b(case\s*ban\s*jayega|hathkadi|giraftari|jail\s*hogi|police\s*aayegi|law\s*enforcement|customs\s*officer)\b",
        "description": "Urgent threat of legal action or physical arrest"
    },

    # Category 2: Credential & OTP Theft (OTP, PIN, Password, Aadhaar)
    {
        "category": "CREDENTIAL_EXFILTRATION",
        "weight": 0.95,
        "regex": r"\b(otp\s*batao|otp\s*share|code\s*batao|pin\s*batao|cvv\s*batao|password\s*batao|sms\s*padho|digits\s*bolo|aadhaar\s*verify)\b",
        "description": "Direct Hindi demand to disclose OTP, PIN, or security code"
    },
    {
        "category": "CREDENTIAL_EXFILTRATION",
        "weight": 0.90,
        "regex": r"\b(jo\s*code\s*aaya|number\s*batao|verification\s*code|pan\s*card\s*number|atm\s*pin|debit\s*card\s*details)\b",
        "description": "Pretexting request for authentication credentials"
    },

    # Category 3: Urgent UPI & Bank Wire Demand (Immediate transfer, UPI PIN)
    {
        "category": "UPI_PAYMENT_COERCION",
        "weight": 0.95,
        "regex": r"\b(turant\s*paise|paise\s*transfer|paisa\s*bhejo|jaldi\s*bhejo|abhi\s*transfer|upi\s*pin\s*dalo|qr\s*code\s*scan|gpay\s*karo|phonepe\s*karo|paytm\s*karo|paise\s*bhej|transfer\s*kardo|transfer\s*krde|transfer\s*kar|transferkar|dega|turant\s*transfer|rupay|ruku\s*pe|paise|तुरंत\s*पैसे|पैसे\s*ट्रांसफर|पैसे\s*भेज|जल्दी\s*भेज)\b",
        "description": "Urgent coercive demand to transfer funds via UPI or net banking"
    },
    {
        "category": "UPI_PAYMENT_COERCION",
        "weight": 0.88,
        "regex": r"\b(galti\s*se\s*paise|wapas\s*bhejo|refund\s*milne\s*ke\s*liye|cashback\s*claim|security\s*deposit|bail\s*ke\s*paise|गलती\s*से\s*पैसे|वापस\s*भेजो)\b",
        "description": "Accidental transfer refund scam or fake cashback prompt"
    },

    # Category 4: KYC Expiration & Bank Account / SIM Suspension
    {
        "category": "SERVICE_DISCONNECTION_KYC",
        "weight": 0.92,
        "regex": r"\b(khata\s*block|account\s*band|sim\s*block|kyc\s*update\s*karo|bijli\s*kat|power\s*cut|connection\s*kat\s*jayega|24\s*ghante\s*mein|खाता\s*बंद|बिजली\s*कट|सिम\s*बंद)\b",
        "description": "Threat of bank account or utility disconnection due to pending KYC"
    },

    # Category 5: Distress Extortion & Medical / Kidnapping Emergency
    {
        "category": "EMERGENCY_EXTORTION",
        "weight": 0.95,
        "regex": r"\b(beta\s*hospital|beti\s*ka\s*accident|hospital\s*mein\s*bharti|khoon\s*chahiye|emergency\s*mein\s*hu|kidnap\s*kar\s*liya|bachane\s*ke\s*liye|एक्सीडेंट\s*हो\s*गया|हॉस्पिटल|किडनैप)\b",
        "description": "Distress extortion fabricating a medical emergency or kidnapping"
    },

    # Category 6: Screen Sharing & Remote Access APKs
    {
        "category": "REMOTE_ACCESS_MALWARE",
        "weight": 0.95,
        "regex": r"\b(anydesk\s*download|teamviewer\s*install|quicksupport|screen\s*share\s*karo|apk\s*file|app\s*install\s*kijiye|स्क्रीन\s*शेयर)\b",
        "description": "Request to install remote access malware or screen-sharing tools"
    }
]


def evaluate_indic_fraud_rules(transcript: str) -> Dict[str, Any]:
    """
    Evaluates transcript for Hindi and Hinglish regional fraud keywords,
    coercive urgency patterns, and Indic scam dialect cues.
    
    Returns:
        dict: {
            "has_indic_scam": bool,
            "indic_scam_prob": float (0.0 to 1.0),
            "matched_keywords": List[str],
            "category": str,
            "description": str
        }
    """
    if not transcript or not transcript.strip():
        return {
            "has_indic_scam": False,
            "indic_scam_prob": 0.0,
            "matched_keywords": [],
            "category": "LEGITIMATE_CONVERSATION",
            "description": "No speech detected"
        }

    text_lower = transcript.lower()
    matched_keywords: List[str] = []
    top_weight = 0.0
    detected_category = "LEGITIMATE_CONVERSATION"
    detected_desc = "No Indic fraud cues detected"

    for pattern in INDIC_SCAM_PATTERNS:
        matches = re.findall(pattern["regex"], text_lower, flags=re.IGNORECASE)
        if matches:
            for m in matches:
                if isinstance(m, tuple):
                    matched_keywords.extend([x for x in m if x])
                else:
                    matched_keywords.append(m)

            if pattern["weight"] > top_weight:
                top_weight = pattern["weight"]
                detected_category = pattern["category"]
                detected_desc = pattern["description"]

    # Unique keywords preserve order
    seen = set()
    dedup_keywords = []
    for k in matched_keywords:
        k_clean = k.strip()
        if k_clean and k_clean not in seen:
            seen.add(k_clean)
            dedup_keywords.append(k_clean)

    has_indic_scam = len(dedup_keywords) > 0 and top_weight >= 0.70

    return {
        "has_indic_scam": has_indic_scam,
        "indic_scam_prob": round(top_weight, 3) if has_indic_scam else 0.0,
        "matched_keywords": dedup_keywords,
        "category": detected_category if has_indic_scam else "LEGITIMATE_CONVERSATION",
        "description": detected_desc if has_indic_scam else "Clean conversational dynamics"
    }
