"""
Out-of-distribution (OOD) detection via distance from a training-set
feature centroid in the model's own penultimate-layer embedding space.
Centroids and thresholds are precomputed offline by
scripts/compute_ood_thresholds.py (one centroid + 99th-percentile distance
threshold per modality, from that modality's training set).

This runs BEFORE the classifier/Grad-CAM pass -- replaces the earlier
softmax-confidence heuristic, which was tested and failed: a screenshot
scored 93% confidence as "Malignant" on the CT model. A discriminative
classifier has no built-in way to say "none of the above" from softmax
alone; distance from the known training distribution in feature space is
a real signal instead of an assumption about how confidence "should"
behave on unfamiliar input.
"""

import logging
from pathlib import Path

import torch

logger = logging.getLogger("ct_xray_api.ood")

BACKEND_DIR = Path(__file__).resolve().parent
OOD_STATS_DIR = BACKEND_DIR / "ood_stats"


class OODDetector:
    def __init__(self, modality: str, device: torch.device):
        stats_path = OOD_STATS_DIR / f"{modality}_ood.pt"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"OOD stats not found at {stats_path} -- run "
                f"scripts/compute_ood_thresholds.py before starting the API."
            )
        stats = torch.load(stats_path, map_location=device, weights_only=True)
        self.modality = modality
        self.centroid = stats["centroid"].to(device)
        self.threshold = stats["threshold"]
        self.n_train_images = stats["n_train_images"]
        logger.info(f"[{modality}] OOD detector loaded: threshold={self.threshold:.2f} "
                    f"(99th percentile over {self.n_train_images} training images)")

    def check(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> tuple[bool, float]:
        """input_tensor: [1,3,H,W] preprocessed the same way as training images.
        Returns (is_ood, distance_from_centroid)."""
        model.eval()
        with torch.no_grad():
            feat = model.forward_features(input_tensor)
        distance = torch.norm(feat[0] - self.centroid).item()
        is_ood = distance > self.threshold
        return is_ood, distance
