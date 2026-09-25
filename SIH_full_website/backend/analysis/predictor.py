######### 5th step ###########
"""
Processes all audio chunks simultaneously using batched inference from model/detector.py
and formats the chunk predictions for aggregator.py and the frontend visualizer.
"""

from model.detector import predict_chunks_batched


def predict_chunks(chunks, sample_rate):
    """
    Batched chunk prediction.
    Runs all chunks in a single forward pass on the GPU.
    """
    if not chunks:
        return []

    # 1. Run all chunks at once in one GPU pass
    batch_results = predict_chunks_batched(chunks, sample_rate)

    # 2. Format results for aggregator.py & the frontend
    predictions = []
    for i, (prob_real, prob_fake) in enumerate(batch_results):
        predictions.append({
            "chunk": i + 1,
            "prob_real": prob_real,
            "prob_fake": prob_fake
        })

    return predictions