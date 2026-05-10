import torch
import torchvision.models as models

def load_model(model_path, num_classes=10):
    model = models.resnet18(num_classes=num_classes)

    state = torch.load(model_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    model.eval()

    return model