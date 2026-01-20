from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "Data"
if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "Instagram_fake_profile_dataset.csv"
POST_METADATA_PATH = DATA_DIR / "ig_post_metadata.csv"
VIDEO_FRAMES_DIR = DATA_DIR / "video_frames"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "instagram_fake_detector.joblib"
CNN_MODEL_PATH = MODEL_DIR / "instagram_video_cnn.pt"

TARGET_COL = "fake"
POST_LABEL_COL = "label_fake"

FEATURE_ORDER = [
    "profile pic",
    "nums/length username",
    "fullname words",
    "nums/length fullname",
    "name==username",
    "description length",
    "external URL",
    "private",
    "#posts",
    "#followers",
    "#follows",
]
