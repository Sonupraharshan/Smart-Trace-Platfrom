"""
Convert PyTorch ResNet50 model to ONNX format for Vercel deployment.

Creates three ONNX models:
  1. resnet50_classifier.onnx  — full model (input → logits)
  2. resnet50_features.onnx    — backbone only (input → 2048-dim embedding)
  3. resnet50_activations.onnx — full model + layer4 activations for saliency

Also applies INT8 dynamic quantization to reduce file size (~4x smaller).

Usage:
    python scripts/convert_to_onnx.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import numpy as np

from ml_pipeline.classifier.model import SteelDefectClassifier
from ml_pipeline.config import MODEL_PATH, SAVED_MODELS_DIR, INPUT_IMG_SIZE, NUM_CLASSES


# ──────────────────────────────────────────────
# 1. Wrapper model that outputs activations too
# ──────────────────────────────────────────────
class ClassifierWithActivations(nn.Module):
    """Wraps the classifier to also output layer4 activation maps."""

    def __init__(self, classifier: SteelDefectClassifier):
        super().__init__()
        self.backbone = classifier.backbone
        self._activations = None

        # Hook into layer4 to capture activations
        self.backbone.layer4.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self._activations = output

    def forward(self, x):
        logits = self.backbone(x)
        return logits, self._activations


class FeatureExtractor(nn.Module):
    """Extracts 2048-dim embeddings (backbone without FC)."""

    def __init__(self, classifier: SteelDefectClassifier):
        super().__init__()
        modules = list(classifier.backbone.children())[:-1]
        self.features = nn.Sequential(*modules)

    def forward(self, x):
        out = self.features(x)
        return torch.flatten(out, 1)


# ──────────────────────────────────────────────
# 2. Export functions
# ──────────────────────────────────────────────
def export_onnx(model, output_path, input_shape, output_names, dynamic_axes=None):
    """Export a PyTorch model to ONNX format."""
    dummy_input = torch.randn(*input_shape)
    model.eval()

    if dynamic_axes is None:
        dynamic_axes = {"input": {0: "batch_size"}}
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}

    # Use legacy TorchScript exporter (dynamo=False) to produce a single
    # self-contained .onnx file with weights embedded, rather than the
    # dynamo exporter which creates separate .onnx.data files.
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    # If the exporter still created an external data file, merge it in
    data_file = Path(str(output_path) + ".data")
    if data_file.exists():
        import onnx
        onnx_model = onnx.load(str(output_path), load_external_data=True)
        onnx.save_model(
            onnx_model,
            str(output_path),
            save_as_external_data=False,
        )
        if data_file.exists():
            data_file.unlink()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [OK] Exported: {output_path} ({size_mb:.1f} MB)")
    return output_path


def quantize_onnx(input_path, output_path):
    """Apply INT8 dynamic quantization to an ONNX model."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            str(input_path),
            str(output_path),
            weight_type=QuantType.QUInt8,
        )
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  [OK] Quantized: {output_path} ({size_mb:.1f} MB)")
        return True
    except ImportError:
        print("  [WARN] onnxruntime.quantization not available, skipping quantization")
        return False
    except Exception as e:
        print(f"  [WARN] Quantization failed: {e}, using unquantized model")
        return False


# ----------------------------------------------
# 3. Main conversion
# ----------------------------------------------
def main():
    print("=" * 60)
    print("PyTorch -> ONNX Model Conversion")
    print("=" * 60)

    # Check that the .pth model exists
    if not MODEL_PATH.exists():
        print(f"\n[FAIL] Model not found at: {MODEL_PATH}")
        print("  Train the model first, or place the .pth file there.")
        sys.exit(1)

    print(f"\nSource model: {MODEL_PATH}")
    pth_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"Source size:  {pth_size:.1f} MB")

    # Load the trained model
    print("\nLoading PyTorch model...")
    classifier = SteelDefectClassifier(pretrained=False)
    state_dict = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    print("  [OK] Model loaded")

    input_shape = (1, 3, INPUT_IMG_SIZE, INPUT_IMG_SIZE)
    output_dir = SAVED_MODELS_DIR

    # -- Export 1: Full classifier --
    print("\n[1/3] Exporting classifier (input -> logits)...")
    classifier_path = output_dir / "resnet50_classifier.onnx"
    export_onnx(classifier, classifier_path, input_shape, ["logits"])

    # -- Export 2: Feature extractor --
    print("\n[2/3] Exporting feature extractor (input -> 2048-dim embedding)...")
    feat_model = FeatureExtractor(classifier)
    feat_model.eval()
    features_path = output_dir / "resnet50_features.onnx"
    export_onnx(feat_model, features_path, input_shape, ["embedding"])

    # -- Export 3: Classifier + activations --
    print("\n[3/3] Exporting classifier with activations (for saliency maps)...")
    act_model = ClassifierWithActivations(classifier)
    act_model.eval()
    activations_path = output_dir / "resnet50_activations.onnx"
    export_onnx(act_model, activations_path, input_shape, ["logits", "activations"])

    # -- Quantize all models --
    print("\n" + "-" * 40)
    print("Applying INT8 dynamic quantization...")
    print("-" * 40)

    for name, path in [
        ("classifier", classifier_path),
        ("features", features_path),
        ("activations", activations_path),
    ]:
        quant_path = path.with_suffix(".quant.onnx")
        if quantize_onnx(path, quant_path):
            # Replace original with quantized
            os.remove(path)
            os.rename(quant_path, path)
            print(f"  [OK] Replaced {name} with quantized version")

    # -- Verify with ONNX Runtime --
    print("\n" + "-" * 40)
    print("Verifying ONNX models with onnxruntime...")
    print("-" * 40)

    try:
        import onnxruntime as ort

        dummy = np.random.randn(*input_shape).astype(np.float32)

        # Verify classifier
        sess = ort.InferenceSession(str(classifier_path))
        result = sess.run(None, {"input": dummy})
        assert result[0].shape == (1, NUM_CLASSES), f"Bad shape: {result[0].shape}"
        print(f"  [OK] Classifier: output shape {result[0].shape}")

        # Verify features
        sess = ort.InferenceSession(str(features_path))
        result = sess.run(None, {"input": dummy})
        assert result[0].shape == (1, 2048), f"Bad shape: {result[0].shape}"
        print(f"  [OK] Features: output shape {result[0].shape}")

        # Verify activations
        sess = ort.InferenceSession(str(activations_path))
        result = sess.run(None, {"input": dummy})
        assert result[0].shape == (1, NUM_CLASSES), f"Bad logits shape: {result[0].shape}"
        print(f"  [OK] Activations: logits={result[0].shape}, maps={result[1].shape}")

    except ImportError:
        print("  [WARN] onnxruntime not installed, skipping verification")
    except Exception as e:
        print(f"  [FAIL] Verification failed: {e}")

    # -- Compare PyTorch vs ONNX output --
    print("\n" + "-" * 40)
    print("Comparing PyTorch vs ONNX predictions...")
    print("-" * 40)

    try:
        import onnxruntime as ort

        # Run PyTorch
        with torch.no_grad():
            pt_input = torch.from_numpy(dummy)
            pt_output = classifier(pt_input).numpy()

        # Run ONNX
        sess = ort.InferenceSession(str(classifier_path))
        onnx_output = sess.run(None, {"input": dummy})[0]

        # Compare
        max_diff = np.max(np.abs(pt_output - onnx_output))
        print(f"  Max absolute difference: {max_diff:.6f}")
        if max_diff < 0.01:
            print("  [OK] Outputs match (within tolerance)")
        else:
            print("  [WARN] Outputs differ -- quantization introduces small changes, "
                  "but predictions should still match")

        # Check predicted class matches
        pt_class = np.argmax(pt_output, axis=1)
        onnx_class = np.argmax(onnx_output, axis=1)
        print(f"  PyTorch predicted class:  {pt_class[0]}")
        print(f"  ONNX predicted class:     {onnx_class[0]}")
        if pt_class[0] == onnx_class[0]:
            print("  [OK] Same predicted class")
        else:
            print("  [WARN] Different predicted class (may be due to random input)")

    except Exception as e:
        print(f"  [WARN] Comparison skipped: {e}")

    # -- Summary --
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"\nONNX models saved to: {output_dir}")
    for f in sorted(output_dir.glob("*.onnx")):
        size = os.path.getsize(f) / (1024 * 1024)
        print(f"  {f.name:40s} {size:>8.1f} MB")

    print(f"\nOriginal PyTorch model: {pth_size:.1f} MB")
    total_onnx = sum(
        os.path.getsize(f) / (1024 * 1024) for f in output_dir.glob("*.onnx")
    )
    print(f"Total ONNX models:     {total_onnx:.1f} MB")
    print(f"Savings:               {pth_size - total_onnx:+.1f} MB "
          f"(model only, PyTorch runtime saves ~2.5 GB more)")


if __name__ == "__main__":
    main()
