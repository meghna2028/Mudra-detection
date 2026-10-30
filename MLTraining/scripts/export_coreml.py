"""
Exports Trained Bharatanatyam Mudra Classifier to Apple Core ML format (.mlpackage and compiled .mlmodel).
Configures inputs, outputs, probability dictionaries, and metadata.
Copies the exported model directly to mudradetection/Resources/ for Swift integration.
"""

import os
import json
import torch
import numpy as np
import coremltools as ct
from feature_extractor import FEATURE_NAMES, NUM_FEATURES
from train import MudraMLP, MUDRAS

def export_coreml_model(base_dir: str):
    dataset_dir = os.path.join(base_dir, "dataset")
    models_dir = os.path.join(base_dir, "models")
    resources_dir = os.path.join(base_dir, "..", "mudradetection", "Resources")
    os.makedirs(resources_dir, exist_ok=True)

    with open(os.path.join(dataset_dir, "normalization_stats.json")) as f:
        stats = json.load(f)

    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    # 1. Instantiate trained PyTorch MLP with embedded normalization buffers
    mlp = MudraMLP(input_dim=NUM_FEATURES, num_classes=len(MUDRAS), mean=mean, std=std)
    weights_path = os.path.join(models_dir, "mlp_best.pt")
    mlp.load_state_dict(torch.load(weights_path, weights_only=True))
    mlp.eval()

    # 2. Trace PyTorch model with example input (1, 77)
    example_input = torch.zeros(1, NUM_FEATURES, dtype=torch.float32)
    traced_model = torch.jit.trace(mlp, example_input)

    # 3. Convert via Core ML Tools
    classifier_config = ct.ClassifierConfig(MUDRAS)

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="features", shape=example_input.shape, dtype=np.float32)],
        classifier_config=classifier_config,
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram"
    )

    # 4. Set Rich Metadata
    mlmodel.author = "Bharatanatyam AI Team"
    mlmodel.license = "Proprietary / Educational"
    mlmodel.short_description = "Classifies 10 Bharatanatyam Asamyuta Hastas (hand mudras) from 77 Vision hand pose geometric features."
    mlmodel.version = "1.0.0"

    mlmodel.input_description["features"] = "A 77-element vector containing normalized landmark coordinates, joint angles, inter-finger distances, and extension ratios."
    mlmodel.output_description["classLabel"] = "The predicted Bharatanatyam mudra name."
    mlmodel.output_description["classLabel_probs"] = "The dictionary of confidence scores for each mudra class."

    # Save to MLTraining/models/
    export_path = os.path.join(models_dir, "MudraClassifier.mlpackage")
    mlmodel.save(export_path)
    print(f"[Core ML] Exported mlpackage to {export_path}")

    # Also save directly to Xcode project Resources
    app_resource_path = os.path.join(resources_dir, "MudraClassifier.mlpackage")
    mlmodel.save(app_resource_path)
    print(f"[Core ML] Copied model package to iOS App: {app_resource_path}")

    # Verify predictions on example input
    test_dict = {"features": example_input.numpy()}
    pred = mlmodel.predict(test_dict)
    print(f"[Core ML Verification] Dummy inference test succeeded:")
    print(f"  Predicted Class: {pred['classLabel']}")
    print(f"  Max Confidence:  {max(pred['classLabel_probs'].values()):.4f}")

if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..")
    export_coreml_model(base)

