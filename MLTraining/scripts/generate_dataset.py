"""
Bharatanatyam Mudra Dataset Generation Pipeline.
Generates anatomically accurate hand landmark samples for 10 Asamyuta Hastas:
1. Pataka, 2. Tripataka, 3. Ardhapataka, 4. Kartarimukha, 5. Mayura,
6. Ardhachandra, 7. Mushti, 8. Shikhara, 9. Suchi, 10. Kapittha.

Implements strict performer separation (no performer leakage),
multi-session variations (rotations, scales, joint flexibilities),
and duplicate prevention.
"""

import os
import math
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from feature_extractor import extract_features, FEATURE_NAMES, NUM_LANDMARKS

MUDRAS = [
    "Pataka",
    "Tripataka",
    "Ardhapataka",
    "Kartarimukha",
    "Mayura",
    "Ardhachandra",
    "Mushti",
    "Shikhara",
    "Suchi",
    "Kapittha"
]

# Base 2D skeletal topology of an upright palm (wrist at 0,0, palm length ~1.0)
# Base MCP angles and segment lengths
BASE_PALM = {
    # Finger lengths: [MCP length, PIP length, DIP length, Tip length]
    "thumb":  [0.38, 0.28, 0.24], # CMC, MP, IP
    "index":  [0.40, 0.25, 0.18], # PIP, DIP, Tip
    "middle": [0.45, 0.28, 0.20],
    "ring":   [0.41, 0.25, 0.18],
    "little": [0.32, 0.20, 0.16],
}

# Base MCP positions relative to wrist (x, y)
BASE_MCP = {
    "thumb_cmc": (-0.25, 0.25),
    "index_mcp": (-0.20, 0.95),
    "middle_mcp": (0.00, 1.00),
    "ring_mcp":   (0.18, 0.92),
    "little_mcp": (0.33, 0.78),
}

# Mudra Canonical Flexion/Extension Angles (in degrees)
# Flexion: 0 deg = straight/extended, 90 deg = right angle bend, 120-150 deg = tightly curled
MUDRA_DEFINITIONS = {
    "Pataka": {
        # All 4 fingers straight & together, thumb gently bent near index
        "thumb":  {"abduction": 15, "flex_mp": 25, "flex_ip": 15},
        "index":  {"spread": -2, "flex_pip": 0, "flex_dip": 0},
        "middle": {"spread": 0,  "flex_pip": 0, "flex_dip": 0},
        "ring":   {"spread": 2,  "flex_pip": 0, "flex_dip": 0},
        "little": {"spread": 4,  "flex_pip": 0, "flex_dip": 0},
    },
    "Tripataka": {
        # Like Pataka, but ring finger bent down
        "thumb":  {"abduction": 15, "flex_mp": 25, "flex_ip": 15},
        "index":  {"spread": -3, "flex_pip": 0, "flex_dip": 0},
        "middle": {"spread": 0,  "flex_pip": 0, "flex_dip": 0},
        "ring":   {"spread": 0,  "flex_pip": 95, "flex_dip": 75}, # bent down
        "little": {"spread": 4,  "flex_pip": 0, "flex_dip": 0},
    },
    "Ardhapataka": {
        # Ring and little fingers bent down, index and middle straight
        "thumb":  {"abduction": 15, "flex_mp": 25, "flex_ip": 15},
        "index":  {"spread": -3, "flex_pip": 0, "flex_dip": 0},
        "middle": {"spread": 0,  "flex_pip": 0, "flex_dip": 0},
        "ring":   {"spread": 0,  "flex_pip": 95, "flex_dip": 80},
        "little": {"spread": 2,  "flex_pip": 100, "flex_dip": 85},
    },
    "Kartarimukha": {
        # Index and middle spread wide apart (scissors), ring & little curled to thumb
        "thumb":  {"abduction": 30, "flex_mp": 40, "flex_ip": 30},
        "index":  {"spread": -22, "flex_pip": 0, "flex_dip": 0}, # spread wide
        "middle": {"spread": 12,  "flex_pip": 0, "flex_dip": 0},  # spread wide
        "ring":   {"spread": 0,   "flex_pip": 105, "flex_dip": 85},
        "little": {"spread": 3,   "flex_pip": 110, "flex_dip": 90},
    },
    "Mayura": {
        # Ring tip touches thumb tip, index/middle/little extended
        "thumb":  {"abduction": 35, "flex_mp": 50, "flex_ip": 35},
        "index":  {"spread": -8, "flex_pip": 0, "flex_dip": 0},
        "middle": {"spread": 0,  "flex_pip": 0, "flex_dip": 0},
        "ring":   {"spread": -10, "flex_pip": 85, "flex_dip": 45}, # curved to meet thumb
        "little": {"spread": 12, "flex_pip": 0, "flex_dip": 0},
    },
    "Ardhachandra": {
        # All 4 fingers straight together, thumb stretched wide open outward
        "thumb":  {"abduction": 75, "flex_mp": 5, "flex_ip": 0}, # stretched wide
        "index":  {"spread": -2, "flex_pip": 0, "flex_dip": 0},
        "middle": {"spread": 0,  "flex_pip": 0, "flex_dip": 0},
        "ring":   {"spread": 2,  "flex_pip": 0, "flex_dip": 0},
        "little": {"spread": 4,  "flex_pip": 0, "flex_dip": 0},
    },
    "Mushti": {
        # All 4 fingers curled into fist, thumb over them
        "thumb":  {"abduction": 25, "flex_mp": 80, "flex_ip": 70},
        "index":  {"spread": 0, "flex_pip": 115, "flex_dip": 95},
        "middle": {"spread": 0, "flex_pip": 120, "flex_dip": 100},
        "ring":   {"spread": 0, "flex_pip": 120, "flex_dip": 100},
        "little": {"spread": 0, "flex_pip": 115, "flex_dip": 95},
    },
    "Shikhara": {
        # Fist with thumb pointing straight up
        "thumb":  {"abduction": 10, "flex_mp": 0, "flex_ip": 0}, # erect straight up
        "index":  {"spread": 0, "flex_pip": 115, "flex_dip": 95},
        "middle": {"spread": 0, "flex_pip": 120, "flex_dip": 100},
        "ring":   {"spread": 0, "flex_pip": 120, "flex_dip": 100},
        "little": {"spread": 0, "flex_pip": 115, "flex_dip": 95},
    },
    "Suchi": {
        # Index pointing straight up (needle), middle/ring/little curled, thumb over them
        "thumb":  {"abduction": 30, "flex_mp": 75, "flex_ip": 65},
        "index":  {"spread": 0, "flex_pip": 0, "flex_dip": 0},   # needle pointing up
        "middle": {"spread": 0, "flex_pip": 120, "flex_dip": 100},
        "ring":   {"spread": 0, "flex_pip": 120, "flex_dip": 100},
        "little": {"spread": 0, "flex_pip": 115, "flex_dip": 95},
    },
    "Kapittha": {
        # Index bent over thumb tip, middle/ring/little curled into fist
        "thumb":  {"abduction": 20, "flex_mp": 45, "flex_ip": 30},
        "index":  {"spread": 0,  "flex_pip": 85, "flex_dip": 75}, # bent over thumb
        "middle": {"spread": 0,  "flex_pip": 120, "flex_dip": 100},
        "ring":   {"spread": 0,  "flex_pip": 120, "flex_dip": 100},
        "little": {"spread": 0,  "flex_pip": 115, "flex_dip": 95},
    },
}

class HandSynthesizer:
    def __init__(self, performer_seed: int):
        self.rng = np.random.RandomState(performer_seed)
        # Biomechanical anatomical variations per performer
        self.hand_scale = self.rng.uniform(0.85, 1.15)
        self.palm_aspect = self.rng.uniform(0.92, 1.08)
        self.finger_length_scale = {
            f: self.rng.uniform(0.90, 1.10) for f in BASE_PALM.keys()
        }
        self.joint_flexibility = self.rng.uniform(0.90, 1.10)

    def generate_hand(self, mudra_name: str, session_variation: float = 1.0) -> List[Tuple[float, float]]:
        mudra_spec = MUDRA_DEFINITIONS[mudra_name]
        pts: List[Tuple[float, float]] = [(0.0, 0.0)] * NUM_LANDMARKS

        # Wrist position in camera frame
        wrist_x = 0.50 + self.rng.normal(0, 0.04) * session_variation
        wrist_y = 0.35 + self.rng.normal(0, 0.04) * session_variation
        pts[0] = (wrist_x, wrist_y)

        # Global hand orientation: roll angle (degrees)
        roll_deg = self.rng.uniform(-35, 35) * session_variation
        roll_rad = math.radians(roll_deg)
        cos_r = math.cos(roll_rad)
        sin_r = math.sin(roll_rad)

        scale = 0.26 * self.hand_scale

        def transform(local_x: float, local_y: float) -> Tuple[float, float]:
            """Scale, rotate and translate local hand coordinates to image frame."""
            rx = (local_x * cos_r - local_y * sin_r) * scale
            ry = (local_x * sin_r + local_y * cos_r) * scale
            return (wrist_x + rx, wrist_y + ry)

        # 1. Thumb chain: CMC -> MP -> IP -> TIP
        t_spec = mudra_spec["thumb"]
        cmc_base = BASE_MCP["thumb_cmc"]
        pts[1] = transform(cmc_base[0] * self.palm_aspect, cmc_base[1])

        # Thumb kinematics
        th_abd = math.radians(t_spec["abduction"] + self.rng.normal(0, 2.5) * session_variation)
        th_mp_flex = math.radians(t_spec["flex_mp"] * self.joint_flexibility + self.rng.normal(0, 3.0) * session_variation)
        th_ip_flex = math.radians(t_spec["flex_ip"] * self.joint_flexibility + self.rng.normal(0, 3.0) * session_variation)

        lengths_t = [l * self.finger_length_scale["thumb"] for l in BASE_PALM["thumb"]]
        # Thumb MP
        mp_dx = -math.sin(th_abd) * lengths_t[0]
        mp_dy = math.cos(th_abd) * lengths_t[0]
        local_mp = (cmc_base[0] + mp_dx, cmc_base[1] + mp_dy)
        pts[2] = transform(local_mp[0], local_mp[1])

        # Thumb IP
        th_ip_angle = th_abd - th_mp_flex
        ip_dx = -math.sin(th_ip_angle) * lengths_t[1]
        ip_dy = math.cos(th_ip_angle) * lengths_t[1]
        local_ip = (local_mp[0] + ip_dx, local_mp[1] + ip_dy)
        pts[3] = transform(local_ip[0], local_ip[1])

        # Thumb Tip
        th_tip_angle = th_ip_angle - th_ip_flex
        tip_dx = -math.sin(th_tip_angle) * lengths_t[2]
        tip_dy = math.cos(th_tip_angle) * lengths_t[2]
        local_tip = (local_ip[0] + tip_dx, local_ip[1] + tip_dy)
        pts[4] = transform(local_tip[0], local_tip[1])

        # 2. Four fingers: Index(5..8), Middle(9..12), Ring(13..16), Little(17..20)
        finger_configs = [
            ("index", 5, BASE_MCP["index_mcp"], mudra_spec["index"]),
            ("middle", 9, BASE_MCP["middle_mcp"], mudra_spec["middle"]),
            ("ring", 13, BASE_MCP["ring_mcp"], mudra_spec["ring"]),
            ("little", 17, BASE_MCP["little_mcp"], mudra_spec["little"]),
        ]

        for fname, base_idx, mcp_base, f_spec in finger_configs:
            lengths = [l * self.finger_length_scale[fname] for l in BASE_PALM[fname]]
            mcp_local = (mcp_base[0] * self.palm_aspect, mcp_base[1])
            pts[base_idx] = transform(mcp_local[0], mcp_local[1])

            spread = math.radians(f_spec["spread"] + self.rng.normal(0, 1.8) * session_variation)
            pip_flex = math.radians(f_spec["flex_pip"] * self.joint_flexibility + self.rng.normal(0, 2.5) * session_variation)
            dip_flex = math.radians(f_spec["flex_dip"] * self.joint_flexibility + self.rng.normal(0, 2.5) * session_variation)

            # PIP position: forward along MCP spread direction
            pip_dx = math.sin(spread) * lengths[0]
            pip_dy = math.cos(spread) * lengths[0]
            local_pip = (mcp_local[0] + pip_dx, mcp_local[1] + pip_dy)
            pts[base_idx + 1] = transform(local_pip[0], local_pip[1])

            # DIP position (flexion curls backward/downward along y-axis)
            dip_angle = spread - pip_flex
            dip_dx = math.sin(dip_angle) * lengths[1]
            dip_dy = math.cos(dip_angle) * lengths[1]
            local_dip = (local_pip[0] + dip_dx, local_pip[1] + dip_dy)
            pts[base_idx + 2] = transform(local_dip[0], local_dip[1])

            # TIP position
            tip_angle = dip_angle - dip_flex
            tip_dx = math.sin(tip_angle) * lengths[2]
            tip_dy = math.cos(tip_angle) * lengths[2]
            local_tip = (local_dip[0] + tip_dx, local_dip[1] + tip_dy)
            pts[base_idx + 3] = transform(local_tip[0], local_tip[1])

        # Add camera vision landmark detection jitter (standard deviation ~ 0.002)
        final_pts = []
        for x, y in pts:
            noise_x = self.rng.normal(0, 0.002)
            noise_y = self.rng.normal(0, 0.002)
            final_pts.append((float(x + noise_x), float(y + noise_y)))

        return final_pts


def generate_full_dataset(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    # 10 Performers
    # Performers 1 to 7: Training Set
    # Performer 8: Validation Set
    # Performers 9 to 10: Testing Set (STRICTLY ISOLATED TO PREVENT DATA LEAKAGE)
    train_performers = [f"performer_{i}" for i in range(1, 8)]
    val_performers = ["performer_8"]
    test_performers = ["performer_9", "performer_10"]

    # Target counts per mudra:
    # Train: 300 samples per mudra (total 3,000)
    # Val: 65 samples per mudra (total 650)
    # Test: 85 samples per mudra (total 850)
    splits = [
        ("train", train_performers, 300),
        ("val", val_performers, 65),
        ("test", test_performers, 85),
    ]

    for split_name, performers, target_per_mudra in splits:
        records = []
        synthesizers = {p: HandSynthesizer(performer_seed=hash(p) % 100000) for p in performers}
        samples_per_p = math.ceil(target_per_mudra / len(performers))

        print(f"Generating {split_name} dataset: {target_per_mudra} samples/mudra across {performers}...")

        for mudra in MUDRAS:
            mudra_samples = 0
            existing_feature_vectors = []

            for p_idx, p_name in enumerate(performers):
                synth = synthesizers[p_name]
                # Sessions simulate different recording environments (lighting, distance)
                for session_num in range(1, 6):
                    session_id = f"{p_name}_sess_{session_num}"
                    session_count = math.ceil(samples_per_p / 5)

                    for s_idx in range(session_count):
                        if mudra_samples >= target_per_mudra:
                            break

                        landmarks = synth.generate_hand(mudra, session_variation=1.0 + (session_num * 0.08))
                        feats = extract_features(landmarks)

                        # Near-duplicate prevention check (Euclidean distance threshold)
                        is_duplicate = False
                        for prev_f in existing_feature_vectors[-100:]:
                            diff = np.linalg.norm(np.array(feats) - np.array(prev_f))
                            if diff < 0.03: # Too close to an existing sample
                                is_duplicate = True
                                break

                        if is_duplicate:
                            continue

                        existing_feature_vectors.append(feats)
                        sample_id = f"{split_name}_{mudra}_{mudra_samples:04d}"

                        row = {
                            "sample_id": sample_id,
                            "mudra_class": mudra,
                            "performer_id": p_name,
                            "session_id": session_id,
                        }
                        # Add features
                        for f_name, f_val in zip(FEATURE_NAMES, feats):
                            row[f_name] = f_val

                        # Add raw landmarks for visualization/reference
                        for l_idx, (lx, ly) in enumerate(landmarks):
                            row[f"lm_{l_idx}_x"] = lx
                            row[f"lm_{l_idx}_y"] = ly

                        records.append(row)
                        mudra_samples += 1

        df = pd.DataFrame(records)
        csv_path = os.path.join(output_dir, f"landmarks_{split_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {split_name} dataset to {csv_path}: {len(df)} total samples ({len(df) // len(MUDRAS)} per mudra).")

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    generate_full_dataset(out_dir)

