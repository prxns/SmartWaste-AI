"""Academic demonstration assumptions, paths and reproducible defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
MATERIAL_POTENTIAL = dict(zip(CLASS_NAMES, (80, 85, 90, 75, 60, 15)))
# Normalized illustrative expert assumptions, NOT empirical recycling rates.
MEMBERSHIPS = {
    "Low": ("trapezoid", (0, 0, 25, 50)),
    "Medium": ("triangle", (25, 50, 75)),
    "High": ("trapezoid", (50, 75, 100, 100)),
}
RULE_TABLE = (("Low", "Low", "Low"),
              ("Medium", "Medium", "Low"),
              ("High", "Medium", "Low"))
SEED = 42
IMAGE_SIZE = 160
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
DATA_DIR = ROOT / "data" / "dataset-resized"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = MODEL_DIR / "smartwaste.keras"
DATASET_URL = "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
