"""TrashNet validation, content fingerprints and leakage-resistant splits."""
from hashlib import sha256
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from config import CLASS_NAMES, SEED
from smartwaste.preprocessing import read_image


def inventory(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Dataset directory is missing: {root}. Run python -m scripts.download_dataset.")
    folders = {p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")}
    if folders != set(CLASS_NAMES):
        raise ValueError(f"Expected exactly {list(CLASS_NAMES)}; found {sorted(folders)}.")
    rows = []
    raw_distribution = {}
    for index, category in enumerate(CLASS_NAMES):
        files = sorted(p for p in (root/category).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            raise ValueError(f"No supported images in {category}.")
        raw_distribution[category] = len(files)
        for path in files:
            try:
                image = read_image(path)
            except ValueError as exc:
                raise ValueError(f"Invalid dataset image {path.name}: {exc}") from exc
            digest = sha256(image.tobytes() + str(image.size).encode()).hexdigest()
            rows.append({"path": path.relative_to(root).as_posix(), "category": category,
                         "label": index, "sha256": digest, "width": image.width, "height": image.height})
    frame = pd.DataFrame(rows)
    # The original archive contains some identical images with conflicting
    # labels. Quarantine ALL members of each ambiguous group; choosing the
    # first label would invent ground truth. Record exclusions explicitly.
    label_counts = frame.groupby("sha256").category.nunique()
    ambiguous = frame.sha256.isin(label_counts[label_counts > 1].index)
    audit = [{"path": row.path, "sha256": row.sha256, "reason": "conflicting labels"}
             for row in frame[ambiguous].itertuples()]
    frame = frame[~ambiguous].copy()
    duplicate = frame.duplicated("sha256", keep="first")
    audit.extend({"path": row.path, "sha256": row.sha256, "reason": "exact duplicate"}
                 for row in frame[duplicate].itertuples())
    frame = frame[~duplicate].reset_index(drop=True)
    if set(frame.category) != set(CLASS_NAMES):
        raise ValueError("Deduplication left a category with no unambiguous images.")
    frame.attrs["raw_class_distribution"] = raw_distribution
    frame.attrs["excluded_images"] = audit
    return frame


def split_dataset(frame, seed=SEED):
    train, rest = train_test_split(frame, test_size=0.30, random_state=seed, stratify=frame.label)
    validation, test = train_test_split(rest, test_size=0.5, random_state=seed, stratify=rest.label)
    parts = []
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        part = part.copy()
        part["split"] = name
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
