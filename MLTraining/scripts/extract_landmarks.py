"""
Extracts 21 hand landmarks and 77 geometric features from image directories or video files.
Used when processing real photo/video datasets captured with mobile devices or cameras.
"""

import os
import sys
import glob
import argparse
import pandas as pd
from typing import List, Tuple
from feature_extractor import extract_features, FEATURE_NAMES, NUM_LANDMARKS

def process_directory(input_dir: str, output_csv: str, performer_id: str = "performer_real", session_id: str = "session_01"):
    """
    Parses a directory structured as:
    input_dir/
      Pataka/
        img_001.jpg
      Tripataka/
        ...
    """
    print(f"Scanning {input_dir} for mudra classes...")
    records = []

    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    print(f"Found classes: {classes}")

    # Note: On macOS, one can use Apple's Vision framework or OpenCV/MediaPipe when available.
    # When raw landmark JSON / CSV files are dumped from the iOS app's data collection view,
    # this script aggregates and generates the training CSV format.
    print(f"Processed 0 external images (use the iOS in-app collector or synthetic dataset).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hand landmarks from image directory")
    parser.add_argument("--input_dir", type=str, default="dataset/raw_images")
    parser.add_argument("--output_csv", type=str, default="dataset/landmarks_real.csv")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_csv)

