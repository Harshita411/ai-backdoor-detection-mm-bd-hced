import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

def get_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)