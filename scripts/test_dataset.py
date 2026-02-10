# python -m scripts.test_dataset

import yaml
import os
import numpy as np
from collections import Counter

from src.datasets.volleyball_feature_dataset import VolleyballB3_stage2
from src.utils.label_encoder import LabelEncoder


with open("configs/B3_b.yaml") as f:
    cfg = yaml.safe_load(f)


def test_stage2_dataset(dataset, encoder, max_samples=5):
    print("=" * 50)
    print("Stage 2 Dataset Test")
    print("=" * 50)

    print("Total samples:", len(dataset))

    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)

    # ===== class distribution =====
    print("\nClass distribution:")
    counter = Counter(labels)
    for label_id, count in counter.items():
        print(f"{encoder.decode(label_id):>15s} : {count}")

    # ===== sample inspection =====
    print("\nSample feature inspection:")
    for i in range(min(max_samples, len(dataset))):
        feat, label = dataset[i]

        print(
            f"Sample {i:02d} | "
            f"Class: {encoder.decode(label):>15s} | "
            f"Shape: {tuple(feat.shape)} | "
            f"Min: {feat.min():.4f} | "
            f"Max: {feat.max():.4f} | "
            f"Mean: {feat.mean():.4f}"
        )

        assert feat.shape[0] == 2048, "❌ Feature dim is not 2048!"
        assert not np.isnan(feat).any(), "❌ NaN detected in features!"
        assert not np.isinf(feat).any(), "❌ Inf detected in features!"

    print("\n✅ Dataset looks GOOD!")


def main():
    encoder = LabelEncoder(class_names=cfg["labels"]["class_names"])

    dataset = VolleyballB3_stage2(
        pickle_file=cfg["data"]["annot_file"],
        videos_root=os.path.join(cfg["data"]["videos_dir"], "videos"),
        video_list = [str(v) for v in cfg["data"]["splits"]["train"]],
        feuture_root=cfg["feature_path"],
        encoder=encoder
    )
    test_stage2_dataset(dataset, encoder)


if __name__ == "__main__":
    main()
