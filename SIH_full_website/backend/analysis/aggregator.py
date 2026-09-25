####### 6th  step #######
''' this step will calculate the total probablity of the prediction 
basically it will combine the probablities of every chunk 
next pass -> risk_score.py'''


def aggregate_predictions(predictions):
    """
    Robust Forensic Consensus Aggregator:
    Distinguishes genuine AI voice clones from isolated microphone background noise.
    """
    if not predictions:
        raise ValueError("No predictions available")

    fake_scores = [p["prob_fake"] for p in predictions]

    # 1. Standard Arithmetic Mean
    mean_fake = sum(fake_scores) / len(fake_scores)

    # 2. Strong Synthetic Chunk Analysis
    strong_fake_chunks = [f for f in fake_scores if f >= 0.70]
    strong_fake_ratio = len(strong_fake_chunks) / len(fake_scores)
    max_fake = max(fake_scores) if fake_scores else 0.0

    # 3. Top half average
    k = max(1, len(fake_scores) // 2)
    top_k_fake = sorted(fake_scores, reverse=True)[:k]
    top_k_avg = sum(top_k_fake) / len(top_k_fake)

    # Forensic consensus:
    # A single isolated glitch should NOT override when the rest of speech is authentic (>70% of chunks are human)
    if strong_fake_ratio >= 0.50:
        agg_fake = (0.60 * top_k_avg) + (0.40 * mean_fake)
    elif strong_fake_ratio >= 0.25 and mean_fake >= 0.35:
        agg_fake = (0.50 * top_k_avg) + (0.50 * mean_fake)
    elif max_fake >= 0.90 and mean_fake >= 0.30 and len(fake_scores) <= 2:
        agg_fake = (0.50 * max_fake) + (0.50 * mean_fake)
    else:
        # High consensus human speech
        agg_fake = mean_fake

    agg_fake = round(min(1.0, max(0.0, agg_fake)), 4)
    agg_real = round(1.0 - agg_fake, 4)

    label = "fake" if agg_fake >= 0.50 else "real"

    return {
        "prob_real": agg_real,
        "prob_fake": agg_fake,
        "label": label
    }