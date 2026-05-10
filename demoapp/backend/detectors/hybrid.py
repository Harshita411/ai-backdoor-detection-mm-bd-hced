import torch
import numpy as np

def compute_gap(model, dataloader):
    gaps = []

    for x, _ in dataloader:
        out = model(x)
        probs = torch.softmax(out, dim=1)

        top2 = torch.topk(probs, 2, dim=1).values
        gap = (top2[:, 0] - top2[:, 1]).detach().numpy()

        gaps.extend(gap)

    return float(np.mean(gaps))


def run_hybrid(model, dataloader, mmbd):
    gap = compute_gap(model, dataloader)

    if gap > 0.30 and mmbd["result"] == "CLEAN":
        return {"result": "SUSPICIOUS", "conf_gap": gap}

    if gap > 0.30:
        return {"result": "BACKDOORED", "conf_gap": gap}

    return {"result": "CLEAN", "conf_gap": gap}