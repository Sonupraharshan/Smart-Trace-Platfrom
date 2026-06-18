"""
Unified Inference Engine
===========================
Single entry point that orchestrates all ML modules:
  1. Classification (ResNet50)
  2. Severity estimation (Grad-CAM proxy mask)
  3. Explainability (Grad-CAM)
  4. Similarity retrieval (FAISS)

This is the ONLY module that the Django views call.
All models are loaded once at startup and reused.
"""

import cv2
import uuid
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from PIL import Image

from ml_pipeline.config import (
    DEVICE,
    MODEL_PATH,
    CLASS_NAMES,
    INPUT_IMG_SIZE,
    GRADCAM_OUTPUT_DIR,
    EMBEDDING_DIM,
)
from ml_pipeline.classifier.model import SteelDefectClassifier
from ml_pipeline.explainability.gradcam import GradCAM
from ml_pipeline.severity.estimator import estimate_severity_from_gradcam
from ml_pipeline.retrieval.search import SimilaritySearchEngine
from ml_pipeline.data.augmentations import get_val_transforms


class InferenceEngine:
    """
    Unified inference engine for the Smart Quality Inspection Platform.

    Loads all models once and exposes a single `inspect_image()` method.
    Falls back gracefully if model or index files are missing.
    """

    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.model = None
        self.gradcam = None
        self.search_engine = None
        self.feature_extractor = None
        self.transform = get_val_transforms()
        self._model_loaded = False

        self._load_model()
        self._load_search_engine()

    def _load_model(self):
        """Load the trained classifier and set up Grad-CAM."""
        model_path = Path(MODEL_PATH)

        if not model_path.exists():
            print(f"[InferenceEngine] WARNING: Model not found at {model_path}")
            print("[InferenceEngine] Running in DEMO mode with untrained model.")
            # Load untrained model for demo purposes
            self.model = SteelDefectClassifier(pretrained=True)
        else:
            print(f"[InferenceEngine] Loading model from {model_path}")
            self.model = SteelDefectClassifier(pretrained=False)
            state_dict = torch.load(
                str(model_path), map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict)
            self._model_loaded = True

        self.model.to(self.device)
        self.model.eval()

        # Set up Grad-CAM targeting the last conv block
        target_layer = self.model.backbone.layer4[-1]
        self.gradcam = GradCAM(self.model, target_layer)

        # Set up feature extractor (backbone without FC)
        modules = list(self.model.backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules).to(self.device)
        self.feature_extractor.eval()

        print("[InferenceEngine] Model and Grad-CAM ready.")

    def _load_search_engine(self):
        """Load the FAISS similarity search engine."""
        self.search_engine = SimilaritySearchEngine()
        if self.search_engine.is_loaded:
            print("[InferenceEngine] FAISS search engine ready.")
        else:
            print("[InferenceEngine] WARNING: FAISS index not found. "
                  "Similarity search disabled.")

    def inspect_image(self, image_path: str) -> dict:
        """
        Run full inspection pipeline on a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with all inspection results:
            {
                "image_path": str,
                "predicted_class": int,
                "defect_label": str,
                "confidence": float,
                "all_probabilities": list,
                "severity_score": float,
                "severity_category": str,
                "defect_area_pct": float,
                "is_defective": bool,
                "gradcam_path": str,
                "similar_images": list[dict],
            }
        """
        # Load and preprocess image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        augmented = self.transform(image=original_image)
        image_tensor = augmented["image"].unsqueeze(0).to(self.device)

        # 1. Classification
        prediction_info = self.gradcam.get_prediction_info(image_tensor)
        predicted_class = prediction_info["predicted_class"]
        confidence = prediction_info["confidence"]

        # 2. Grad-CAM
        _orig_vis, heatmap, overlay, cam_map = self.gradcam.generate(
            image_tensor,
            target_class=predicted_class,
            original_image=original_image,
        )

        # Save Grad-CAM overlay
        gradcam_filename = f"gradcam_{uuid.uuid4().hex[:8]}.png"
        GRADCAM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        gradcam_path = str(GRADCAM_OUTPUT_DIR / gradcam_filename)

        # Save heatmap and overlay one below another. The original image is
        # already shown separately in the UI, so don't repeat it here.
        combined = np.vstack([heatmap, overlay])
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.imwrite(gradcam_path, combined_bgr)

        # 3. Severity estimation from Grad-CAM map
        if predicted_class == 0:
            severity_info = {
                "severity_score": 0.0,
                "severity_category": "No Defect",
                "defect_area_pct": 0.0,
            }
        else:
            severity_info = estimate_severity_from_gradcam(cam_map, threshold=0.5)

        # 4. Similarity retrieval
        embedding = self._extract_embedding(image_tensor)
        similar_images = self.search_engine.find_similar(embedding)

        # Assemble result
        result = {
            "image_path": image_path,
            "predicted_class": predicted_class,
            "defect_label": CLASS_NAMES.get(predicted_class, f"Class {predicted_class}"),
            "confidence": round(confidence, 4),
            "all_probabilities": prediction_info["all_probabilities"],
            "severity_score": severity_info["severity_score"],
            "severity_category": severity_info["severity_category"],
            "defect_area_pct": severity_info["defect_area_pct"],
            "is_defective": predicted_class != 0,
            "gradcam_path": gradcam_path,
            "gradcam_filename": gradcam_filename,
            "similar_images": similar_images,
        }

        return result

    def _extract_embedding(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Extract feature embedding from a single image tensor."""
        with torch.no_grad():
            features = self.feature_extractor(image_tensor)
            features = torch.flatten(features, 1)
        return features.cpu().numpy()

    @property
    def is_ready(self) -> bool:
        """Check if the engine is ready for inference."""
        return self.model is not None


# ──────────────────────────────────────────────
# Singleton pattern for Django integration
# ──────────────────────────────────────────────
_engine_instance = None


def get_inference_engine() -> InferenceEngine:
    """
    Get or create the singleton InferenceEngine.

    This ensures models are loaded only once across all Django views.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = InferenceEngine()
    return _engine_instance
