import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models import resnet18

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_SAMPLES = 2000
EPS = 1e-10

transform = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=True
)

def load_model(path):
    from src.resnet import ResNet18

    model = ResNet18()
    state_dict = torch.load(path, map_location=DEVICE)

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model
def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + EPS), dim=1)

def compute_conf_gap(probs):
    top2 = torch.topk(probs, 2, dim=1).values
    return top2[:, 0] - top2[:, 1]

def compute_hced(probs):
    K = probs.shape[1]
    entropy = compute_entropy(probs)
    gap = compute_conf_gap(probs)
    norm_entropy = entropy / np.log(K)
    hced = gap * (1 - norm_entropy)
    return hced, entropy, gap

def run_hced(model, loader, max_samples=1000):
    hced_scores = []
    entropies = []
    gaps = []
    count = 0

    with torch.no_grad():
        for images, _ in tqdm(loader):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            hced, entropy, gap = compute_hced(probs)

            hced_scores.extend(hced.cpu().numpy())
            entropies.extend(entropy.cpu().numpy())
            gaps.extend(gap.cpu().numpy())

            count += images.size(0)
            if count >= max_samples:
                break

    return np.array(hced_scores), np.array(entropies), np.array(gaps)

if __name__ == "__main__":

    clean_model_path = "clean0/model.pth"
    trojan_model_path = "model0/model.pth"

    print("Loading models...")
    clean_model = load_model(clean_model_path)
    trojan_model = load_model(trojan_model_path)

    print("Running HCED on clean model...")
    clean_hced, clean_entropy, clean_gap = run_hced(clean_model, testloader, NUM_SAMPLES)

    print("Running HCED on trojan model...")
    trojan_hced, trojan_entropy, trojan_gap = run_hced(trojan_model, testloader, NUM_SAMPLES)

    print("\nCLEAN MODEL:")
    print("Avg Entropy:", clean_entropy.mean())
    print("Avg Gap:", clean_gap.mean())
    print("Avg HCED:", clean_hced.mean())

    print("\nTROJAN MODEL:")
    print("Avg Entropy:", trojan_entropy.mean())
    print("Avg Gap:", trojan_gap.mean())
    print("Avg HCED:", trojan_hced.mean())

    clean_mean = clean_hced.mean()
    trojan_mean = trojan_hced.mean()
    threshold = (clean_mean + trojan_mean) / 2

    print("\nThreshold:", threshold)

    print("Clean classified as:", "BACKDOORED" if clean_mean > threshold else "CLEAN")
    print("Trojan classified as:", "BACKDOORED" if trojan_mean > threshold else "CLEAN")

    plt.figure(figsize=(10,6))

plt.hist(clean_hced, bins=50, alpha=0.6, label="Clean Model", density=True)
plt.hist(trojan_hced, bins=50, alpha=0.6, label="Trojan Model", density=True)

plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f"Threshold ({threshold:.3f})")

plt.title("HCED Score Distribution (Clean vs Trojan)")
plt.xlabel("HCED Score")
plt.ylabel("Density")
plt.legend()
plt.grid(True)

plt.savefig("hced_histogram.png", dpi=300)
plt.show()
