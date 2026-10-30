"""
Hand Landmark Feature Extractor for Mudra Detection.
Computes 77 invariant geometric features from 21 hand landmarks (Vision format).
Must remain mathematically identical to HandFeatureExtractor.swift.
"""

import math
from typing import List, Tuple, Dict, Any
import numpy as np

# Landmark order matching Apple Vision VNDetectHumanHandPoseRequest
LANDMARK_NAMES = [
    "wrist",          # 0
    "thumbCMC",       # 1
    "thumbMP",        # 2
    "thumbIP",        # 3
    "thumbTip",       # 4
    "indexMCP",       # 5
    "indexPIP",       # 6
    "indexDIP",       # 7
    "indexTip",       # 8
    "middleMCP",      # 9
    "middlePIP",      # 10
    "middleDIP",      # 11
    "middleTip",      # 12
    "ringMCP",        # 13
    "ringPIP",        # 14
    "ringDIP",        # 15
    "ringTip",        # 16
    "littleMCP",      # 17
    "littlePIP",      # 18
    "littleDIP",      # 19
    "littleTip",      # 20
]

NUM_LANDMARKS = 21
NUM_FEATURES = 77

FEATURE_NAMES = []
# 1..40: Normalized coordinates relative to wrist (excluding wrist itself at index 0)
for idx in range(1, NUM_LANDMARKS):
    FEATURE_NAMES.append(f"norm_{LANDMARK_NAMES[idx]}_x")
    FEATURE_NAMES.append(f"norm_{LANDMARK_NAMES[idx]}_y")

# 41..55: 15 Finger joint angles (0 to 1, normalized by pi)
ANGLE_JOINTS = [
    ("thumb_cmc_angle", 0, 1, 2),
    ("thumb_mp_angle", 1, 2, 3),
    ("thumb_ip_angle", 2, 3, 4),
    ("index_mcp_angle", 0, 5, 6),
    ("index_pip_angle", 5, 6, 7),
    ("index_dip_angle", 6, 7, 8),
    ("middle_mcp_angle", 0, 9, 10),
    ("middle_pip_angle", 9, 10, 11),
    ("middle_dip_angle", 10, 11, 12),
    ("ring_mcp_angle", 0, 13, 14),
    ("ring_pip_angle", 13, 14, 15),
    ("ring_dip_angle", 14, 15, 16),
    ("little_mcp_angle", 0, 17, 18),
    ("little_pip_angle", 17, 18, 19),
    ("little_dip_angle", 18, 19, 20),
]
for name, _, _, _ in ANGLE_JOINTS:
    FEATURE_NAMES.append(name)

# 56..65: 10 Pairwise fingertip distances
FINGERTIP_PAIRS = [
    ("dist_thumb_index", 4, 8),
    ("dist_thumb_middle", 4, 12),
    ("dist_thumb_ring", 4, 16),
    ("dist_thumb_little", 4, 20),
    ("dist_index_middle", 8, 12),
    ("dist_index_ring", 8, 16),
    ("dist_index_little", 8, 20),
    ("dist_middle_ring", 12, 16),
    ("dist_middle_little", 12, 20),
    ("dist_ring_little", 16, 20),
]
for name, _, _ in FINGERTIP_PAIRS:
    FEATURE_NAMES.append(name)

# 66..75: 10 Finger extension ratios (to wrist and to corresponding MCP/CMC)
EXTENSION_PAIRS = [
    ("ext_thumb_wrist", 4, 0),
    ("ext_index_wrist", 8, 0),
    ("ext_middle_wrist", 12, 0),
    ("ext_ring_wrist", 16, 0),
    ("ext_little_wrist", 20, 0),
    ("ext_thumb_base", 4, 1),
    ("ext_index_base", 8, 5),
    ("ext_middle_base", 12, 9),
    ("ext_ring_base", 16, 13),
    ("ext_little_base", 20, 17),
]
for name, _, _ in EXTENSION_PAIRS:
    FEATURE_NAMES.append(name)

# 76..77: Hand proportions
PROPORTION_PAIRS = [
    ("palm_width", 5, 17),
    ("hand_length", 0, 12),
]
for name, _, _ in PROPORTION_PAIRS:
    FEATURE_NAMES.append(name)

assert len(FEATURE_NAMES) == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {len(FEATURE_NAMES)}"


def compute_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """
    Computes angle at vertex p2 in radians normalized by pi (range: [0, 1]).
    Vector u = p1 - p2, Vector v = p3 - p2.
    """
    ux = p1[0] - p2[0]
    uy = p1[1] - p2[1]
    vx = p3[0] - p2[0]
    vy = p3[1] - p2[1]

    dot = ux * vx + uy * vy
    len_u = math.sqrt(ux * ux + uy * uy)
    len_v = math.sqrt(vx * vx + vy * vy)

    denom = len_u * len_v
    if denom < 1e-7:
        return 0.0

    cos_val = max(-1.0, min(1.0, dot / denom))
    angle_rad = math.acos(cos_val)
    return angle_rad / math.pi  # Normalized to [0, 1]


def compute_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def extract_features(landmarks: List[Tuple[float, float]]) -> List[float]:
    """
    Extracts 77 features from a list of 21 (x, y) hand landmark coordinates.
    Input coordinates should be in Vision normalized image coordinates (0.0 to 1.0).
    Returns a list of 77 float features.
    """
    if len(landmarks) != NUM_LANDMARKS:
        raise ValueError(f"Expected {NUM_LANDMARKS} landmarks, got {len(landmarks)}")

    features = []

    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    # Hand scale: Distance between wrist and middle MCP
    scale = compute_distance(wrist, middle_mcp)
    if scale < 1e-6:
        scale = 1.0  # Degenerate fallback

    # 1..40: Normalized coordinates relative to wrist (landmarks 1 to 20)
    for i in range(1, NUM_LANDMARKS):
        norm_x = (landmarks[i][0] - wrist[0]) / scale
        norm_y = (landmarks[i][1] - wrist[1]) / scale
        features.append(float(norm_x))
        features.append(float(norm_y))

    # 41..55: 15 Joint angles
    for _, idx1, idx2, idx3 in ANGLE_JOINTS:
        angle = compute_angle(landmarks[idx1], landmarks[idx2], landmarks[idx3])
        features.append(float(angle))

    # 56..65: 10 Pairwise fingertip distances
    for _, idx1, idx2 in FINGERTIP_PAIRS:
        dist = compute_distance(landmarks[idx1], landmarks[idx2]) / scale
        features.append(float(dist))

    # 66..75: 10 Extension ratios
    for _, idx1, idx2 in EXTENSION_PAIRS:
        dist = compute_distance(landmarks[idx1], landmarks[idx2]) / scale
        features.append(float(dist))

    # 76..77: 2 Palm proportions
    for _, idx1, idx2 in PROPORTION_PAIRS:
        dist = compute_distance(landmarks[idx1], landmarks[idx2]) / scale
        features.append(float(dist))

    return features

