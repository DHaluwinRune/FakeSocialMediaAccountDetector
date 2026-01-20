from __future__ import annotations

import torch
from torch import nn
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_model(pretrained: bool = True) -> nn.Module:
    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
        except AttributeError:
            weights = None

    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def load_model_checkpoint(
    checkpoint_path,
    device: torch.device | None = None,
    pretrained: bool = False,
) -> tuple[nn.Module, int]:
    device = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    image_size = int(checkpoint.get("image_size", 224))

    model = build_model(pretrained=pretrained)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, image_size
