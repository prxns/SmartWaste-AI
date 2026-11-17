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
    rows, seen = [], {}
    for index, category in enumerate(CLASS_NAMES):
        files = sorted(p for p in (root/category).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            raise ValueError(f"No supported images in {category}.")
        for path in files:
            try:
                image = read_image(path)
            except ValueError as exc:
                raise ValueError(f"Invalid dataset image {path.name}: {exc}") from exc
            digest = sha256(image.tobytes() + str(image.size).encode()).hexdigest()
            if digest in seen:
                if seen[digest] != category:
                    raise ValueError("Identical images have contradictory class labels.")
                continue  # exact decoded duplicates must never straddle splits
            seen[digest] = category
            rows.append({"path": path.relative_to(root).as_posix(), "category": category,
                         "label": index, "sha256": digest, "width": image.width, "height": image.height})
    return pd.DataFrame(rows)


def split_dataset(frame, seed=SEED):
    train, rest = train_test_split(frame, test_size=0.30, random_state=seed, stratify=frame.label)
    validation, test = train_test_split(rest, test_size=0.5, random_state=seed, stratify=rest.label)
    parts = []
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        part = part.copy()
        part["split"] = name
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
