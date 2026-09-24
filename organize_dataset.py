"""
organize_dataset.py
--------------------
Run this LOCALLY on your full downloaded Kaggle dataset folder
("CT KIDNEY DATASET Normal, CYST, TUMOR and STONE") to convert it
into a binary structure: Normal vs Abnormal (Cyst+Tumor+Stone merged).

Usage:
    python organize_dataset.py --source "path/to/CT KIDNEY DATASET Normal, CYST, TUMOR and STONE" --dest "path/to/output"

After running, you'll have:
    output/
        Normal/      (all normal images)
        Abnormal/    (all cyst + tumor + stone images, renamed to avoid clashes)

Then zip the `output` folder and upload it to your Google Drive for use
in the training notebook (train_model.ipynb).
"""

import argparse
import os
import shutil


def organize(source_dir: str, dest_dir: str):
    normal_out = os.path.join(dest_dir, "Normal")
    abnormal_out = os.path.join(dest_dir, "Abnormal")
    os.makedirs(normal_out, exist_ok=True)
    os.makedirs(abnormal_out, exist_ok=True)

    # Folder names as they appear in the Kaggle dataset (case-insensitive match)
    class_folders = {
        "normal": normal_out,
        "cyst": abnormal_out,
        "tumor": abnormal_out,
        "stone": abnormal_out,
    }

    counts = {"Normal": 0, "Abnormal": 0}

    for entry in os.listdir(source_dir):
        entry_path = os.path.join(source_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        key = entry.strip().lower()
        if key not in class_folders:
            print(f"Skipping unrecognized folder: {entry}")
            continue

        target_dir = class_folders[key]
        target_class = "Normal" if target_dir == normal_out else "Abnormal"

        for fname in os.listdir(entry_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            src_file = os.path.join(entry_path, fname)
            # Prefix with source class name to avoid filename collisions
            # (e.g. Cyst- (1).jpg and Tumor- (1).jpg both exist)
            new_name = f"{entry}_{fname}"
            dst_file = os.path.join(target_dir, new_name)
            shutil.copy2(src_file, dst_file)
            counts[target_class] += 1

    print("\nDone.")
    print(f"Normal images:   {counts['Normal']}")
    print(f"Abnormal images: {counts['Abnormal']}")
    print(f"\nOutput saved to: {dest_dir}")
    print("Next: zip this folder and upload it to Google Drive for the training notebook.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize CT Kidney dataset into binary Normal/Abnormal folders.")
    parser.add_argument("--source", required=True, help="Path to the extracted Kaggle dataset folder")
    parser.add_argument("--dest", required=True, help="Path to save the organized binary dataset")
    args = parser.parse_args()
    organize(args.source, args.dest)
