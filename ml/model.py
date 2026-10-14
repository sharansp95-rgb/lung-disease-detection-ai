"""
Transfer-learning image classifier wrapping a torchvision backbone.
Class count and class names are never hardcoded here -- passed in by the caller.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

SUPPORTED_BACKBONES = ("densenet121", "resnet18")


class ImageClassifier(nn.Module):
    """Wraps a torchvision backbone with its final layer replaced for
    `num_classes`. Exposes freeze_backbone()/unfreeze_backbone() so train.py
    can do a frozen-backbone warmup before full fine-tuning."""

    def __init__(self, backbone: str = "densenet121", num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone {backbone!r}, expected one of {SUPPORTED_BACKBONES}")
        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone == "densenet121":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.densenet121(weights=weights)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
            self._head_names = {"classifier"}
        else:  # resnet18
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            self._head_names = {"fc"}

    def forward(self, x):
        return self.backbone(x)

    def forward_features(self, x) -> torch.Tensor:
        """Penultimate-layer embedding (input to the final classification
        layer), for use as a fixed feature extractor -- e.g. out-of-
        distribution detection via distance from a training-set centroid.
        Architecture-specific since torchvision doesn't expose this
        uniformly across backbones."""
        if self.backbone_name == "densenet121":
            features = self.backbone.features(x)
            out = F.relu(features, inplace=False)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            return torch.flatten(out, 1)
        else:  # resnet18
            out = self.backbone.conv1(x)
            out = self.backbone.bn1(out)
            out = self.backbone.relu(out)
            out = self.backbone.maxpool(out)
            out = self.backbone.layer1(out)
            out = self.backbone.layer2(out)
            out = self.backbone.layer3(out)
            out = self.backbone.layer4(out)
            out = self.backbone.avgpool(out)
            return torch.flatten(out, 1)

    def _is_head_param(self, param_name: str) -> bool:
        return param_name.split(".")[0] in self._head_names

    def freeze_backbone(self):
        for name, p in self.backbone.named_parameters():
            p.requires_grad = self._is_head_param(name)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> ImageClassifier:
    return ImageClassifier(backbone=backbone, num_classes=num_classes, pretrained=pretrained)
