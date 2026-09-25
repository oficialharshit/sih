"""
Semantic Intent Fraud Detector.
Uses BGE-Small dense vector embeddings + Contrastive Anchors to identify
scam categories and intent in ~3ms without relying on rigid keywords.
"""

import torch
from sentence_transformers import SentenceTransformer, util

# 1. Master Knowledge Base of 10 Scam Categories
UNIVERSAL_SCAM_ANCHORS = {
    "CREDENTIAL_EXFILTRATION": [
        "Share, read, or speak the one-time verification code, OTP, or secret password sent to your device.",
        "Open the newly received SMS or message and read back the digits, numbers, or characters written there.",
        "Tell me the security code or authentication digits appearing on your screen.",
        "Confirm your debit card CVV, ATM PIN, or Internet banking login password."
    ],
    "DIGITAL_ARREST_LAW_ENFORCEMENT": [
        "An arrest warrant, FIR, or court summons has been issued against your name or Aadhaar card.",
        "Your parcel containing illegal narcotics, drugs, or fake passports was seized by customs or police.",
        "You are under digital arrest and must remain on video call with cyber crime or CBI officers.",
        "Pay a security deposit or bail amount immediately to clear your name from a criminal investigation."
    ],
    "UPI_PAYMENT_COERCION": [
        "Open your GooglePay, PhonePe, or Paytm app and approve or authorize the payment request.",
        "Scan this QR code or enter your UPI PIN to receive your cashback, refund, or prize money.",
        "You were sent excess money by mistake, transfer the remaining balance back immediately.",
        "Send a small test transaction of 1 rupee to verify and activate your account."
    ],
    "SERVICE_DISCONNECTION_KYC": [
        "Your bank account, credit card, or SIM card will be permanently blocked or frozen within 24 hours.",
        "Your electricity or gas connection will be disconnected tonight due to an unpaid bill update.",
        "Your KYC verification is pending or expired; update it immediately to prevent suspension.",
        "Link your PAN card to your bank account immediately by clicking the verification link."
    ],
    "REMOTE_ACCESS_MALWARE": [
        "Download and install AnyDesk, QuickSupport, TeamViewer, or RustDesk on your phone or computer.",
        "Click the link to download and install the customer support verification APK file.",
        "Share your mobile screen or accept the remote connection prompt so we can fix the issue.",
        "Turn on screen sharing or accessibility permissions in your phone settings."
    ],
    "EMERGENCY_IMPERSONATION": [
        "Your son, daughter, or family member met with a fatal accident and is in critical hospital condition.",
        "Your relative has been arrested in a police raid and needs immediate cash for bail.",
        "We have kidnapped your loved one, pay the ransom money immediately or they will be harmed.",
        "I am calling from a hospital emergency room requesting immediate payment for blood or surgery."
    ],
    "JOB_INVESTMENT_FRAUD": [
        "Earn 3000 to 5000 rupees daily by rating hotels on Google Maps or liking YouTube videos.",
        "Join our VIP Telegram group for guaranteed stock market tips and high daily returns.",
        "Deposit money into this crypto trading account to double your investment in 24 hours.",
        "Pay a registration fee, uniform charge, or document processing fee to confirm your job offer."
    ],
    "LOTTERY_REWARD_PHISHING": [
        "Congratulations! You won a multi-lakh lottery in KBC, Car lucky draw, or festive contest.",
        "Your credit card reward points worth thousands are expiring today; claim cash before midnight.",
        "Pay government tax, GST, or processing fees to release your lottery prize money.",
        "You won a gift voucher, click the link to claim your free reward."
    ],
    "FRIENDLY_PRETEXTING": [
        "Hey bro, this is your friend calling from a new number, I am stuck in an emergency and need money.",
        "This is your company CEO or manager, I am in a meeting, buy Google Play or Apple gift cards urgently.",
        "I need a small favor, transfer funds to this account and I will pay you back in the evening.",
        "Your account has a small issue, just share that code you got and I will sort it out for you."
    ],
    "TECH_SUPPORT_INVOICE": [
        "Your computer is infected with viruses or spyware; call our technician to prevent data theft.",
        "Your annual anti-virus or software subscription of 499 dollars has been renewed automatically.",
        "To cancel your unauthorized subscription charge, connect with our cancellation department.",
        "A hacker has accessed your email and camera; pay now to prevent your private videos from leaking."
    ]
}

# 2. Master Safe Anchors (Normal Everyday Chat & Official Introductions)
SAFE_ANCHORS = [
    "Casual conversation between friends or family about meeting, food, movies, work, or sports.",
    "Scheduling a business meeting, office discussion, or regular project update.",
    "Food delivery, cab driver, or courier service confirming your street address.",
    "Friendly greetings, asking how someone is doing, or planning a weekend hangout.",
    "Hello, introducing yourself by name, stating who you are, saying I am Rajesh or employee introduction.",
    "Professional self-introduction, giving your name, job title, and polite greeting."
]

# 3. Initialize Model & Pre-encode Anchors on Startup
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)

scam_sentences = []
scam_categories = []
for category, sentences in UNIVERSAL_SCAM_ANCHORS.items():
    for sentence in sentences:
        scam_sentences.append(sentence)
        scam_categories.append(category)

scam_embeddings = embedder.encode(scam_sentences, convert_to_tensor=True)
safe_embeddings = embedder.encode(SAFE_ANCHORS, convert_to_tensor=True)


def evaluate_semantic_intent(transcript: str):
    """
    Evaluates transcript for scam intent in under 4ms using contrastive vector similarity.
    """
    if not transcript or not transcript.strip() or transcript.strip() in ["No speech detected.", "No clear speech detected."]:
        return {
            "prediction": "NON_SCAM",
            "scam_probability": 0.0,
            "non_scam_probability": 1.0,
            "category": "LEGITIMATE_CONVERSATION",
            "matched_pattern": "No speech detected"
        }

    transcript_vec = embedder.encode(transcript, convert_to_tensor=True)

    # Cosine similarities
    scam_sims = util.cos_sim(transcript_vec, scam_embeddings)[0]
    safe_sims = util.cos_sim(transcript_vec, safe_embeddings)[0]

    best_scam_idx = scam_sims.argmax().item()
    best_scam_score = scam_sims[best_scam_idx].item()
    best_category = scam_categories[best_scam_idx]
    best_scam_anchor = scam_sentences[best_scam_idx]

    best_safe_score = safe_sims.max().item()

    # Contrastive Decision Logic (requires clear margin over safe anchors)
    is_scam = (best_scam_score >= 0.68) and (best_scam_score > best_safe_score + 0.05)
    is_suspicious = (best_scam_score >= 0.60) and (best_scam_score > best_safe_score + 0.05)

    if is_scam:
        prediction = "SCAM"
        scam_prob = min(0.99, best_scam_score)
    elif is_suspicious:
        prediction = "SUSPICIOUS"
        scam_prob = round(best_scam_score, 3)
    else:
        prediction = "NON_SCAM"
        scam_prob = max(0.01, round(max(0.0, 1.0 - best_safe_score) * 0.4, 3))

    return {
        "prediction": prediction,
        "scam_probability": round(scam_prob, 3),
        "non_scam_probability": round(1.0 - scam_prob, 3),
        "category": best_category if (is_scam or is_suspicious) else "LEGITIMATE_CONVERSATION",
        "matched_pattern": best_scam_anchor if (is_scam or is_suspicious) else "Normal conversational dialogue"
    }