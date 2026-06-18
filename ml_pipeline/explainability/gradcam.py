"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
========================================================
Produces visual explanations for CNN predictions by highlighting
the regions of the image that most influence the classification.

Targets the last convolutional layer of ResNet50 (layer4[-1]).

Outputs:
  - Heatmap (jet colormap)
  - Overlay (original + heatmap)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ml_pipeline.config import INPUT_IMG_SIZE


class GradCAM:
    """
    Grad-CAM implementation for ResNet50-based classifiers.

    Usage:
        gradcam = GradCAM(model, target_layer=model.backbone.layer4[-1])
        original, heatmap, overlay = gradcam.generate(image_tensor, target_class=3)
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: The trained classifier model
            target_layer: The convolutional layer to extract gradients from
                          (e.g., model.backbone.layer4[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor: torch.Tensor,
                 target_class: int = None,
                 original_image: np.ndarray = None) -> tuple:
        """
        Generate Grad-CAM visualization.

        Args:
            image_tensor: Preprocessed image tensor (1, C, H, W) or (C, H, W)
            target_class: Class to generate explanation for.
                          If None, uses the predicted class.
            original_image: Original (unnormalized) image as numpy array (H, W, 3) RGB.
                            If None, the overlay will use the tensor visualization.

        Returns:
            Tuple of (original_image, heatmap, overlay) as numpy arrays (H, W, 3)
            with values in [0, 255] and dtype uint8.
        """
        self.model.eval()

        # Ensure batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)
        image_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(image_tensor)
        probabilities = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Grad-CAM computation
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global average pooling of gradients → channel weights
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=device)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU — only keep positive influence
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        if original_image is not None:
            output_height, output_width = original_image.shape[:2]
        else:
            output_height = INPUT_IMG_SIZE
            output_width = INPUT_IMG_SIZE

        # Resize to the display image size. When the original image is
        # available, keep its dimensions instead of forcing a square.
        cam_resized = cv2.resize(cam, (output_width, output_height))

        # Create heatmap (jet colormap)
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Prepare original image
        if original_image is not None:
            orig = original_image.astype(np.uint8, copy=False)
        else:
            # Denormalize tensor
            orig = self._denormalize_tensor(image_tensor[0])

        # Create overlay
        overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

        return orig, heatmap, overlay, cam_resized

    @staticmethod
    def _denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
        """Convert a normalized tensor back to a displayable numpy image."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        img = tensor.cpu() * std + mean
        img = img.clamp(0, 1)
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return img

    def get_prediction_info(self, image_tensor: torch.Tensor) -> dict:
        """
        Get prediction class and confidence without generating Grad-CAM.

        Args:
            image_tensor: Preprocessed image tensor

        Returns:
            Dict with predicted_class and confidence
        """
        self.model.eval()
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return {
            "predicted_class": predicted.item(),
            "confidence": confidence.item(),
            "all_probabilities": probabilities[0].cpu().numpy().tolist(),
        }
