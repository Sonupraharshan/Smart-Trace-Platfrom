"""
Unified Inference Engine
===========================
Single entry point that orchestrates all ML modules:
  1. Classification (ResNet50 via ONNX Runtime or PyTorch)
  2. Severity estimation (activation-based saliency proxy mask)
  3. Explainability (saliency maps)
  4. Similarity retrieval (FAISS)

This is the ONLY module that the Django views call.
All models are loaded once at startup and reused.

Backend selection:
  - If ONNX models exist → uses ONNX Runtime (lightweight, for Vercel)
  - If PyTorch .pth exists → falls back to PyTorch (local development)
"""

import cv2
import uuid
import numpy as np
from pathlib import Path

from ml_pipeline.config import (
    DEVICE,
    MODEL_PATH,
    CLASS_NAMES,
    INPUT_IMG_SIZE,
    GRADCAM_OUTPUT_DIR,
    EMBEDDING_DIM,
    ONNX_MODEL_PATH,
    ONNX_FEATURES_PATH,
    ONNX_ACTIVATIONS_PATH,
)
from ml_pipeline.severity.estimator import estimate_severity_from_gradcam
from ml_pipeline.retrieval.search import SimilaritySearchEngine


class InferenceEngine:
    """
    Unified inference engine for the Smart Quality Inspection Platform.

    Loads all models once and exposes a single `inspect_image()` method.
    Falls back gracefully if model or index files are missing.

    Supports two backends:
      - ONNX Runtime (preferred for deployment — no PyTorch needed)
      - PyTorch (fallback for local development)
    """

    def __init__(self):
        self.backend = None  # "onnx" or "pytorch"
        self._model_loaded = False

        # ONNX sessions
        self._classifier_session = None
        self._features_session = None
        self._activations_session = None

        # PyTorch models (only if backend == "pytorch")
        self._pt_model = None
        self._pt_gradcam = None
        self._pt_feature_extractor = None
        self._pt_transform = None
        self._pt_device = None

        # Shared
        self.search_engine = None

        self._load_model()
        self._load_search_engine()

    def _load_model(self):
        """Load ML models — prefer ONNX, fall back to PyTorch."""

        # ── Try ONNX first ──
        if ONNX_ACTIVATIONS_PATH.exists() and ONNX_FEATURES_PATH.exists():
            self._load_onnx()
            return

        # ── Fall back to PyTorch ──
        if MODEL_PATH.exists() or True:  # Allow untrained demo mode
            self._load_pytorch()
            return

        print("[InferenceEngine] WARNING: No model files found. "
              "Engine will not be able to run inference.")

    def _load_onnx(self):
        """Load ONNX Runtime sessions."""
        import onnxruntime as ort

        print("[InferenceEngine] Loading ONNX Runtime backend...")

        # Use CPU execution provider (Vercel has no GPU)
        providers = ["CPUExecutionProvider"]

        # Activations model (logits + layer4 activations for saliency)
        self._activations_session = ort.InferenceSession(
            str(ONNX_ACTIVATIONS_PATH), providers=providers
        )

        # Features model (embeddings for FAISS)
        self._features_session = ort.InferenceSession(
            str(ONNX_FEATURES_PATH), providers=providers
        )

        # Classifier model (optional, activations model can also provide logits)
        if ONNX_MODEL_PATH.exists():
            self._classifier_session = ort.InferenceSession(
                str(ONNX_MODEL_PATH), providers=providers
            )

        self.backend = "onnx"
        self._model_loaded = True
        print("[InferenceEngine] ONNX models loaded successfully.")

    def _load_pytorch(self):
        """Load PyTorch model (fallback for local dev)."""
        try:
            import torch
            import torch.nn as nn
            from ml_pipeline.classifier.model import SteelDefectClassifier
            from ml_pipeline.explainability.gradcam import GradCAM
            from ml_pipeline.data.augmentations import get_val_transforms
        except ImportError:
            print("[InferenceEngine] PyTorch not available and no ONNX models found.")
            return

        device = DEVICE
        self._pt_device = device
        model_path = Path(MODEL_PATH)

        if not model_path.exists():
            print(f"[InferenceEngine] WARNING: Model not found at {model_path}")
            print("[InferenceEngine] Running in DEMO mode with untrained model.")
            model = SteelDefectClassifier(pretrained=True)
        else:
            print(f"[InferenceEngine] Loading PyTorch model from {model_path}")
            model = SteelDefectClassifier(pretrained=False)
            state_dict = torch.load(
                str(model_path), map_location=device, weights_only=True
            )
            model.load_state_dict(state_dict)
            self._model_loaded = True

        model.to(device)
        model.eval()
        self._pt_model = model

        # Grad-CAM
        target_layer = model.backbone.layer4[-1]
        self._pt_gradcam = GradCAM(model, target_layer)

        # Feature extractor
        modules = list(model.backbone.children())[:-1]
        self._pt_feature_extractor = nn.Sequential(*modules).to(device)
        self._pt_feature_extractor.eval()

        # Transforms
        self._pt_transform = get_val_transforms()

        self.backend = "pytorch"
        print(f"[InferenceEngine] PyTorch backend ready (device: {device}).")

    def _load_search_engine(self):
        """Load the FAISS similarity search engine."""
        self.search_engine = SimilaritySearchEngine()
        if self.search_engine.is_loaded:
            print("[InferenceEngine] FAISS search engine ready.")
        else:
            print("[InferenceEngine] WARNING: FAISS index not found. "
                  "Similarity search disabled.")

    # ──────────────────────────────────────────
    # Public API — identical regardless of backend
    # ──────────────────────────────────────────

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
        if self.backend == "onnx":
            return self._inspect_onnx(image_path)
        elif self.backend == "pytorch":
            return self._inspect_pytorch(image_path)
        else:
            raise RuntimeError(
                "No ML backend available. Ensure ONNX models or PyTorch "
                "model files are present."
            )

    # ──────────────────────────────────────────
    # ONNX Runtime inference path
    # ──────────────────────────────────────────

    def _inspect_onnx(self, image_path: str) -> dict:
        """Run inference using ONNX Runtime."""
        from ml_pipeline.data.augmentations import preprocess_image_numpy
        from ml_pipeline.explainability.gradcam import (
            generate_saliency_from_activations,
            get_prediction_info_from_logits,
        )

        # Load and preprocess image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Preprocess for model input
        input_tensor = preprocess_image_numpy(original_image)

        # 1. Classification + activations (single forward pass)
        logits, activations = self._activations_session.run(
            None, {"input": input_tensor}
        )

        prediction_info = get_prediction_info_from_logits(logits)
        predicted_class = prediction_info["predicted_class"]
        confidence = prediction_info["confidence"]

        # 2. Saliency map (activation-based, no gradients)
        _orig_vis, heatmap, overlay, cam_map = generate_saliency_from_activations(
            logits=logits,
            activations=activations,
            target_class=predicted_class,
            original_image=original_image,
        )

        # Save saliency overlay
        gradcam_filename = f"gradcam_{uuid.uuid4().hex[:8]}.png"
        GRADCAM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        gradcam_path = str(GRADCAM_OUTPUT_DIR / gradcam_filename)

        combined = np.vstack([heatmap, overlay])
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.imwrite(gradcam_path, combined_bgr)

        # 3. Severity estimation
        if predicted_class == 0:
            severity_info = {
                "severity_score": 0.0,
                "severity_category": "No Defect",
                "defect_area_pct": 0.0,
            }
        else:
            severity_info = estimate_severity_from_gradcam(cam_map, threshold=0.5)

        # 4. Similarity retrieval
        embedding = self._extract_embedding_onnx(input_tensor)
        similar_images = self.search_engine.find_similar(embedding)

        return {
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

    def _extract_embedding_onnx(self, input_tensor: np.ndarray) -> np.ndarray:
        """Extract feature embedding using ONNX features model."""
        result = self._features_session.run(None, {"input": input_tensor})
        return result[0]  # shape (1, 2048)

    def extract_embedding(self, image_path: str) -> np.ndarray:
        """
        Extract feature embedding for a given image path using the active backend.

        Args:
            image_path: Path to the image file

        Returns:
            Feature embedding vector as a numpy array of shape (1, 2048) or (2048,)
        """
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        if self.backend == "onnx":
            from ml_pipeline.data.augmentations import preprocess_image_numpy
            input_tensor = preprocess_image_numpy(original_image)
            return self._extract_embedding_onnx(input_tensor)
        elif self.backend == "pytorch":
            augmented = self._pt_transform(image=original_image)
            image_tensor = augmented["image"].unsqueeze(0).to(self._pt_device)
            return self._extract_embedding_pytorch(image_tensor)
        else:
            raise RuntimeError("No ML backend loaded.")

    # ──────────────────────────────────────────
    # PyTorch inference path (fallback)
    # ──────────────────────────────────────────

    def _inspect_pytorch(self, image_path: str) -> dict:
        """Run inference using PyTorch (original code path)."""
        import torch

        # Load and preprocess image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        augmented = self._pt_transform(image=original_image)
        image_tensor = augmented["image"].unsqueeze(0).to(self._pt_device)

        # 1. Classification
        prediction_info = self._pt_gradcam.get_prediction_info(image_tensor)
        predicted_class = prediction_info["predicted_class"]
        confidence = prediction_info["confidence"]

        # 2. Grad-CAM
        _orig_vis, heatmap, overlay, cam_map = self._pt_gradcam.generate(
            image_tensor,
            target_class=predicted_class,
            original_image=original_image,
        )

        # Save Grad-CAM overlay
        gradcam_filename = f"gradcam_{uuid.uuid4().hex[:8]}.png"
        GRADCAM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        gradcam_path = str(GRADCAM_OUTPUT_DIR / gradcam_filename)

        combined = np.vstack([heatmap, overlay])
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.imwrite(gradcam_path, combined_bgr)

        # 3. Severity estimation
        if predicted_class == 0:
            severity_info = {
                "severity_score": 0.0,
                "severity_category": "No Defect",
                "defect_area_pct": 0.0,
            }
        else:
            severity_info = estimate_severity_from_gradcam(cam_map, threshold=0.5)

        # 4. Similarity retrieval
        embedding = self._extract_embedding_pytorch(image_tensor)
        similar_images = self.search_engine.find_similar(embedding)

        return {
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

    def _extract_embedding_pytorch(self, image_tensor) -> np.ndarray:
        """Extract feature embedding using PyTorch feature extractor."""
        import torch

        with torch.no_grad():
            features = self._pt_feature_extractor(image_tensor)
            features = torch.flatten(features, 1)
        return features.cpu().numpy()

    @property
    def is_ready(self) -> bool:
        """Check if the engine is ready for inference."""
        return self.backend is not None

    @property
    def device(self):
        """Return the compute device (for compatibility with views.py)."""
        if self.backend == "pytorch":
            return self._pt_device
        return "cpu"


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
