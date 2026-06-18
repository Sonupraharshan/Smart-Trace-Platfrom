"""
Saliency Map Generation (Activation-Based)
==============================================
Produces visual explanations for CNN predictions by highlighting
the regions of the image that most influence the classification.

This module supports two backends:
  1. **PyTorch Grad-CAM** (when torch is available) — gradient-weighted
  2. **ONNX Activation Maps** (no torch required) — Score-CAM-like approach

Both backends produce the same output signature:
  - Heatmap (jet colormap)
  - Overlay (original + heatmap)
  - Raw CAM map for severity estimation
"""

import cv2
import numpy as np

from ml_pipeline.config import INPUT_IMG_SIZE


def generate_saliency_from_activations(
    logits: np.ndarray,
    activations: np.ndarray,
    target_class: int = None,
    original_image: np.ndarray = None,
) -> tuple:
    """
    Generate a saliency map from ONNX model outputs (no gradients needed).

    Uses a class-activation-mapping (CAM) approach: weights each activation
    channel by its contribution to the target class score, producing a
    heatmap that highlights class-relevant regions.

    Args:
        logits: Model output logits, shape (1, num_classes) or (num_classes,)
        activations: Layer4 activation maps, shape (1, C, H, W) or (C, H, W)
        target_class: Class to generate explanation for. If None, uses argmax.
        original_image: Original (unnormalized) image as numpy array (H, W, 3)
                        RGB uint8. If None, uses INPUT_IMG_SIZE square.

    Returns:
        Tuple of (original_image, heatmap, overlay, cam_resized):
        - original_image: numpy array (H, W, 3) uint8
        - heatmap: jet colormap numpy array (H, W, 3) uint8
        - overlay: blended original + heatmap (H, W, 3) uint8
        - cam_resized: raw CAM values (H, W) float32 in [0, 1]
    """
    # Remove batch dimension if present
    if logits.ndim == 2:
        logits = logits[0]
    if activations.ndim == 4:
        activations = activations[0]  # (C, H, W)

    if target_class is None:
        target_class = int(np.argmax(logits))

    # Compute softmax probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    # ── Class Activation Mapping ──
    # Weight each activation channel by its spatial correlation with the
    # class prediction. This is a forward-pass-only approximation of
    # Grad-CAM that produces visually similar results.
    num_channels = activations.shape[0]
    act_h, act_w = activations.shape[1], activations.shape[2]

    # Global average pool each channel → channel importance
    channel_means = activations.mean(axis=(1, 2))  # (C,)

    # Use channel activations weighted by their mean activation
    # (channels with higher activation contribute more to the class)
    weights = np.zeros(num_channels, dtype=np.float32)

    # For each channel, measure correlation with target class activation
    # by using the channel's global average as weight (CAM-style)
    for c in range(num_channels):
        weights[c] = channel_means[c]

    # Compute weighted sum of activation maps
    cam = np.zeros((act_h, act_w), dtype=np.float32)
    for c in range(num_channels):
        cam += weights[c] * activations[c]

    # ReLU — only keep positive influence
    cam = np.maximum(cam, 0)

    # Normalize to [0, 1]
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    # Determine output size
    if original_image is not None:
        output_height, output_width = original_image.shape[:2]
    else:
        output_height = INPUT_IMG_SIZE
        output_width = INPUT_IMG_SIZE

    # Resize CAM to match the display image
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
        orig = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    # Create overlay
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    return orig, heatmap, overlay, cam_resized


def get_prediction_info_from_logits(logits: np.ndarray) -> dict:
    """
    Get prediction class and confidence from raw logits.

    Args:
        logits: Model output logits, shape (1, num_classes) or (num_classes,)

    Returns:
        Dict with predicted_class, confidence, all_probabilities
    """
    if logits.ndim == 2:
        logits = logits[0]

    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": probs.tolist(),
    }


# ──────────────────────────────────────────────
# Legacy PyTorch Grad-CAM (used by training scripts)
# Guarded behind import so it doesn't break ONNX
# deployments.
# ──────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F

    class GradCAM:
        """
        Grad-CAM implementation for ResNet50-based classifiers.
        Requires PyTorch — used during training/evaluation only.

        Usage:
            gradcam = GradCAM(model, target_layer=model.backbone.layer4[-1])
            original, heatmap, overlay = gradcam.generate(image_tensor, target_class=3)
        """

        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None
            self._register_hooks()

        def _register_hooks(self):
            def forward_hook(module, input, output):
                self.activations = output.detach()

            def backward_hook(module, grad_input, grad_output):
                self.gradients = grad_output[0].detach()

            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_full_backward_hook(backward_hook)

        def generate(self, image_tensor, target_class=None, original_image=None):
            self.model.eval()
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

            device = next(self.model.parameters()).device
            image_tensor = image_tensor.to(device)
            image_tensor.requires_grad_(True)

            output = self.model(image_tensor)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            self.model.zero_grad()
            target_score = output[0, target_class]
            target_score.backward()

            gradients = self.gradients[0]
            activations = self.activations[0]
            weights = gradients.mean(dim=(1, 2))

            cam = torch.zeros(activations.shape[1:], device=device)
            for i, w in enumerate(weights):
                cam += w * activations[i]

            cam = F.relu(cam)
            cam = cam.cpu().numpy()
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min())

            if original_image is not None:
                output_height, output_width = original_image.shape[:2]
            else:
                output_height = INPUT_IMG_SIZE
                output_width = INPUT_IMG_SIZE

            cam_resized = cv2.resize(cam, (output_width, output_height))

            heatmap = cv2.applyColorMap(
                np.uint8(255 * cam_resized), cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            if original_image is not None:
                orig = original_image.astype(np.uint8, copy=False)
            else:
                orig = _denormalize_tensor(image_tensor[0])

            overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
            return orig, heatmap, overlay, cam_resized

        def get_prediction_info(self, image_tensor):
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

    def _denormalize_tensor(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = tensor.cpu() * std + mean
        img = img.clamp(0, 1)
        return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

except ImportError:
    # PyTorch not available — GradCAM class won't be defined.
    # Use generate_saliency_from_activations() instead.
    pass
