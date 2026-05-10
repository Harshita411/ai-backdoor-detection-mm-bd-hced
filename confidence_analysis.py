import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from src.resnet import ResNet18

device = 'cpu'

# -----------------------------
# Load Clean and Trojan Models
# -----------------------------

clean_model = ResNet18()
trojan_model = ResNet18()

clean_model.load_state_dict(
    torch.load('./clean0/model.pth', weights_only=False)
)

trojan_model.load_state_dict(
    torch.load('./model0/model.pth', weights_only=False)
)

clean_model.eval()
trojan_model.eval()

# -----------------------------
# CIFAR-10 Transform
# -----------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    ),
])

# -----------------------------
# Load CIFAR-10 Test Dataset
# -----------------------------

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=False,
    transform=transform
)

# -----------------------------
# Confidence Analysis
# -----------------------------

num_samples = 20

clean_total_conf = 0
trojan_total_conf = 0
difference_total = 0

print("\n===================================")
print("Running Confidence Stability Analysis")
print("===================================\n")

for i in range(num_samples):

    image, label = testset[i]

    image = image.unsqueeze(0)

    # -----------------------------
    # Clean Model Prediction
    # -----------------------------

    with torch.no_grad():
        clean_output = clean_model(image)
        clean_probs = F.softmax(clean_output, dim=1)

    # -----------------------------
    # Trojaned Model Prediction
    # -----------------------------

    with torch.no_grad():
        trojan_output = trojan_model(image)
        trojan_probs = F.softmax(trojan_output, dim=1)

    # -----------------------------
    # Confidence Scores
    # -----------------------------

    clean_confidence = torch.max(clean_probs).item()
    trojan_confidence = torch.max(trojan_probs).item()

    # -----------------------------
    # Predicted Classes
    # -----------------------------

    clean_prediction = torch.argmax(clean_probs).item()
    trojan_prediction = torch.argmax(trojan_probs).item()

    # -----------------------------
    # Difference Calculation
    # -----------------------------

    difference = abs(clean_confidence - trojan_confidence)

    clean_total_conf += clean_confidence
    trojan_total_conf += trojan_confidence
    difference_total += difference

    # -----------------------------
    # Per-image Output
    # -----------------------------

    print(f"Image {i+1}")

    print(
        f"Clean Model  -> Prediction: {clean_prediction}, "
        f"Confidence: {clean_confidence:.4f}"
    )

    print(
        f"Trojan Model -> Prediction: {trojan_prediction}, "
        f"Confidence: {trojan_confidence:.4f}"
    )

    print(f"Confidence Difference: {difference:.4f}")

    # Suspicious behavior flag
    if trojan_confidence > clean_confidence:
        print("Suspicious confidence increase detected.")

    print("-----------------------------------")

# -----------------------------
# Final Statistical Analysis
# -----------------------------

avg_clean = clean_total_conf / num_samples
avg_trojan = trojan_total_conf / num_samples
avg_difference = difference_total / num_samples

print("\n===================================")
print("FINAL CONFIDENCE ANALYSIS")
print("===================================\n")

print(f"Average Clean Confidence   : {avg_clean:.4f}")
print(f"Average Trojan Confidence  : {avg_trojan:.4f}")
print(f"Average Confidence Difference : {avg_difference:.4f}")

if avg_trojan > avg_clean:
    print("\nOverall suspicious behavioral pattern detected.")

print("\nAnalysis Complete.\n")
