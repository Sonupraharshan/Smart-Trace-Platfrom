"""
ResNet50 Steel Defect Classifier
===================================
Transfer learning with fine-tuning:
- Pretrained ImageNet backbone
- Frozen early layers (conv1 → layer2)
- Fine-tuned deep layers (layer3 → layer4)
- Custom classification head for 5 classes
"""

import torch
import torch.nn as nn
from torchvision import models

from ml_pipeline.config import NUM_CLASSES, DEVICE


class SteelDefectClassifier(nn.Module):
    """
    ResNet50-based classifier for steel surface defects.

    Architecture:
        ResNet50 backbone (pretrained) → AdaptiveAvgPool → Dropout → FC(2048, 5)

    The first layers are frozen to leverage pretrained features.
    Only layer3, layer4, and the new FC head are trained.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Freeze early layers (conv1 → layer2)
        frozen_layers = [
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.layer1,
            self.backbone.layer2,
        ]
        for layer in frozen_layers:
            for param in layer.parameters():
                param.requires_grad = False

        # Replace the classification head
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Logits of shape (B, num_classes)
        """
        return self.backbone(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embedding from the avgpool layer (before FC).

        Used by the similarity retrieval module (Module 4).

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Feature vector of shape (B, 2048)
        """
        # Forward through all layers except the final fc
        modules = list(self.backbone.children())[:-1]  # Exclude fc
        feature_extractor = nn.Sequential(*modules)
        features = feature_extractor(x)
        features = torch.flatten(features, 1)
        return features

    def get_trainable_params(self):
        """
        Return parameter groups with different learning rates.

        Returns:
            List of dicts for optimizer:
            - Fine-tuned backbone layers (lower LR)
            - New FC head (higher LR)
        """
        # Backbone fine-tuned params (layer3 + layer4)
        backbone_params = []
        for name, param in self.backbone.named_parameters():
            if param.requires_grad and "fc" not in name:
                backbone_params.append(param)

        # FC head params
        fc_params = list(self.backbone.fc.parameters())

        return [
            {"params": backbone_params, "lr": 1e-4},
            {"params": fc_params, "lr": 1e-3},
        ]


def load_classifier(model_path: str = None, device: torch.device = DEVICE):
    """
    Load a trained classifier from disk.

    Args:
        model_path: Path to the saved .pth file.
        device: Device to load model onto.

    Returns:
        Model in eval mode.
    """
    from ml_pipeline.config import MODEL_PATH

    if model_path is None:
        model_path = str(MODEL_PATH)

    model = SteelDefectClassifier(pretrained=False)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
