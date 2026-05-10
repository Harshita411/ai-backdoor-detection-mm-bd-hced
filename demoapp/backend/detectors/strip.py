import torch
import torch.nn.functional as F
import numpy as np

def entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

def run_strip(model, dataloader):
    model.eval()

    clean_e = []
    perturbed_e = []

    data_iter = iter(dataloader)

    for _ in range(30):
        x, _ = next(data_iter)

        out = model(x)
        probs = F.softmax(out, dim=1)
        clean_e.extend(entropy(probs).detach().numpy())

        # perturbation
        idx = torch.randperm(x.size(0))
        x2 = x[idx]
        x_mix = (x + x2) / 2

        out_mix = model(x_mix)
        probs_mix = F.softmax(out_mix, dim=1)

        perturbed_e.extend(entropy(probs_mix).detach().numpy())

    ec = float(np.mean(clean_e))
    ep = float(np.mean(perturbed_e))

    return {
        "result": "BACKDOORED" if ep < 0.3 else "CLEAN",
        "entropy_clean": ec,
        "entropy_perturbed": ep
    }