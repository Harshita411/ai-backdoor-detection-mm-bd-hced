import numpy as np

def run_mmbd(model):
    weights = model.fc.weight.detach().cpu().numpy()

    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    W = weights / (norms + 1e-8)

    margins = []

    for i in range(len(W)):
        dists = [np.linalg.norm(W[i] - W[j]) for j in range(len(W)) if i != j]
        margins.append(min(dists))

    avg_margin = float(np.mean(margins))

    return {
        "result": "BACKDOORED" if avg_margin < 0.25 else "CLEAN",
        "avg_margin": avg_margin
    }