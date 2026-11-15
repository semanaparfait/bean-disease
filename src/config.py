"""Central configuration shared by every stage of the pipeline."""
import os
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"          # new data waiting to be used for retraining
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "pipeline.db"))
FEATURES_CSV = DATA_DIR / "image_features.csv"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "bean_disease_mobilenetv2.h5"     # current production model
METRICS_PATH = MODELS_DIR / "metrics.json"                  # metrics of the current model
ARCHIVE_DIR = MODELS_DIR / "archive"                        # previous model versions

CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
CLASS_LABELS = {
    "angular_leaf_spot": "Angular Leaf Spot",
    "bean_rust": "Bean Rust",
    "healthy": "Healthy",
}
# Fixed categorical colour order (validated colour-blind safe as a 3-colour set)
CLASS_COLORS = {"angular_leaf_spot": "#2a78d6", "bean_rust": "#eb6834", "healthy": "#1baf7a"}

IMG_SIZE = 160
BATCH_SIZE = 32
SEED = 42
VAL_SPLIT = 0.15

# Retraining trigger: retrain automatically once this many new labelled images are uploaded
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "30"))
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "true").lower() == "true"
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "5"))
# Number of original training images replayed with the new data (prevents catastrophic forgetting)
REPLAY_PER_CLASS = int(os.getenv("REPLAY_PER_CLASS", "60"))

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for d in (TRAIN_DIR, TEST_DIR, UPLOAD_DIR, MODELS_DIR, ARCHIVE_DIR):
    d.mkdir(parents=True, exist_ok=True)
