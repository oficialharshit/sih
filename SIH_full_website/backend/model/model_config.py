MODEL_NAME = "abhishtagatya/wav2vec2-base-960h-itw-deepfake"

TARGET_SAMPLE_RATE = 16000

REAL_LABEL = 0  # 'bona-fide'
FAKE_LABEL = 1  # 'spoof'
HF_TOKEN = "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"

# ================= ENSEMBLE WEIGHTS (Must sum to 1.0) =================
WEIGHT_VOICE_DEEPFAKE = 0.50   # 50% Wav2Vec2 Acoustic Model
WEIGHT_SEMANTIC_INTENT = 0.30  # 30% Semantic BGE Categories
WEIGHT_DISTILBERT_NLP = 0.20   # 20% DistilBERT Scam Classifier

# ================= ENSEMBLE WEIGHTS =================
WEIGHT_WAV2VEC2 = 0.70    # 70% Deep Learning Transformer
WEIGHT_DSP_PROSODY = 0.30 # 30% Physical Signal & Prosody Check