import io
import json
import sys
import tempfile
import warnings
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

try:
    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent
INSTAGRAM_ROOT = ROOT_DIR / "Instagram Fake Profile Detector"
TIKTOK_ROOT = ROOT_DIR / "TikTok Fake Profile Detector"

if not INSTAGRAM_ROOT.exists():
    raise RuntimeError(f"Instagram app folder not found: {INSTAGRAM_ROOT}")
if not TIKTOK_ROOT.exists():
    raise RuntimeError(f"TikTok app folder not found: {TIKTOK_ROOT}")

sys.path.insert(0, str(INSTAGRAM_ROOT))
sys.path.insert(0, str(TIKTOK_ROOT))

from src.cnn_live import download_cnn_inputs_for_username
from src.cnn_predict import predict_from_frame_paths, predict_from_images
from src.config import FEATURE_ORDER
from src.fusion_predict import predict_fusion_from_profile_input
from src.instagram import ProfileFetchError, extract_username, features_from_profile_input
from src.predict import load_model, predict_from_features
from scripts.predict_account import (
    DEFAULT_TIMEOUT,
    extract_profile_from_state,
    fetch_profile_state,
    fetch_profile_videos,
    parse_username,
    predict_live,
)

TIKTOK_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    )
}

TIKTOK_MODEL_DIR = TIKTOK_ROOT / "models"
FUSION_WEIGHT_ACCOUNT = 0.7
FUSION_WEIGHT_CNN = 0.3
IG_FEATURE_LABELS = {
    "profile pic": "Profile photo present",
    "nums/length username": "Digits in username",
    "fullname words": "Words in full name",
    "nums/length fullname": "Digits in full name",
    "name==username": "Name matches username",
    "description length": "Bio length",
    "external URL": "External link",
    "private": "Account is private",
    "#posts": "Post count",
    "#followers": "Follower count",
    "#follows": "Following count",
}

st.set_page_config(page_title="Fake Profile Detector", layout="centered")

MCT_THEME_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");
/* Adjust colors and spacing in :root for future tweaks */
:root {
  --mct-primary: #44C9F6;
  --mct-primary-hover: #2BA8D6;
  --mct-bg: #F7F9FC;
  --mct-surface: #FFFFFF;
  --mct-border: #E5E7EB;
  --mct-text: #111827;
  --mct-muted: #6B7280;
  --mct-radius: 10px;
  --mct-shadow: 0 4px 12px rgba(17, 24, 39, 0.06);
}

html, body, [class*="css"] {
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
  color: var(--mct-text);
}

.stApp {
  background-color: var(--mct-bg);
}

#MainMenu, footer, header {
  visibility: hidden;
}

/* Subtle container polish without changing layout */
.block-container {
  padding-top: 2rem;
  padding-bottom: 2rem;
}

/* Buttons */
div.stButton > button {
  background: var(--mct-primary);
  color: #FFFFFF;
  border: 1px solid var(--mct-primary);
  border-radius: var(--mct-radius);
  box-shadow: var(--mct-shadow);
  font-weight: 600;
}

div.stButton > button:hover {
  background: var(--mct-primary-hover);
  border-color: var(--mct-primary-hover);
  color: #FFFFFF;
}

/* Inputs and selectors */
div.stTextInput input,
div.stTextArea textarea,
div.stSelectbox div[data-baseweb="select"] > div,
div.stMultiSelect div[data-baseweb="select"] > div {
  background: var(--mct-surface) !important;
  border: 1px solid var(--mct-border) !important;
  border-radius: var(--mct-radius) !important;
  color: var(--mct-text) !important;
  box-shadow: none !important;
}

div.stTextInput input:focus,
div.stTextArea textarea:focus,
div.stSelectbox div[data-baseweb="select"] > div:focus-within,
div.stMultiSelect div[data-baseweb="select"] > div:focus-within {
  border-color: var(--mct-primary) !important;
  box-shadow: 0 0 0 2px rgba(68, 201, 246, 0.25) !important;
}

input[type="radio"] {
  accent-color: var(--mct-primary);
}

/* Headings */
h1, h2, h3 {
  font-weight: 700;
  color: var(--mct-text);
}

/* Secondary text */
small, .stCaption {
  color: var(--mct-muted);
}

/* Metrics */
[data-testid="stMetric"] {
  background: var(--mct-surface);
  border: 1px solid var(--mct-border);
  border-radius: var(--mct-radius);
  padding: 0.75rem 1rem;
  box-shadow: var(--mct-shadow);
}
</style>
"""

st.markdown(MCT_THEME_CSS, unsafe_allow_html=True)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


@st.cache_resource
def tt_load_image_model(model_dir):
    model_dir = Path(model_dir)
    meta_path = model_dir / "image_model_meta.json"
    model_path = model_dir / "image_model.pth"
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Image model artifacts not found. Train with scripts/train_image_model.py first."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    classes = meta.get("classes", [])
    image_size = int(meta.get("image_size", 224))
    if not classes:
        raise RuntimeError("Model meta does not include classes.")

    model = SimpleCNN(num_classes=len(classes))
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, classes, image_size


def tt_build_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def tt_extract_frame_urls(item, max_frames=3):
    urls = []
    if not item:
        return urls

    def add_url(value):
        if value and value not in urls:
            urls.append(value)
        return len(urls) >= max_frames

    video_data = item.get("video") or {}
    for key in ("dynamicCover", "originCover", "cover"):
        value = video_data.get(key) or item.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        if isinstance(value, dict):
            value = (value.get("urlList") or value.get("url_list") or [None])[0]
        if add_url(value):
            return urls

    image_post = item.get("imagePost") or item.get("image_post") or {}
    images = image_post.get("images") or []
    for image in images:
        if not isinstance(image, dict):
            continue
        value = (
            image.get("displayImage") or image.get("imageURL") or image.get("imageUrl")
        )
        if isinstance(value, dict):
            value = (value.get("urlList") or value.get("url_list") or [None])[0]
        if add_url(value):
            break

    if len(urls) >= max_frames:
        return urls

    value = item.get("thumbnail")
    if isinstance(value, dict):
        value = value.get("url") or value.get("source")
    if isinstance(value, list):
        value = value[0] if value else None
    if add_url(value):
        return urls

    thumbnails = item.get("thumbnails") or []
    for thumb in thumbnails:
        if isinstance(thumb, dict):
            value = thumb.get("url") or thumb.get("source")
        else:
            value = thumb
        if add_url(value):
            return urls
    return urls


def tt_download_image(url, timeout):
    try:
        response = requests.get(url, headers=TIKTOK_HEADERS, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return None
    try:
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception:
        return None


def tt_score_images(model, classes, image_size, images):
    transform = tt_build_transform(image_size)
    probs_list = []
    for image in images:
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        probs_list.append(probs)
    stack = torch.stack(probs_list)
    avg_probs = stack.mean(dim=0)
    pred_idx = int(torch.argmax(avg_probs).item())
    return avg_probs.tolist(), pred_idx, stack.tolist()


def tt_format_prediction_label(class_name):
    label = class_name.lower()
    if "fake" in label or "bot" in label:
        return "FAKE/BOT"
    if "real" in label:
        return "REAL"
    return class_name.upper()


def tt_get_fake_index(classes, fallback_idx):
    for idx, name in enumerate(classes):
        label = name.lower()
        if "fake" in label or "bot" in label:
            return idx
    return fallback_idx


def tt_median(values):
    if not values:
        return None
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2


@st.cache_resource
def tt_load_account_model(model_dir):
    model_dir = Path(model_dir)
    model_path = model_dir / "model.joblib"
    columns_path = model_dir / "feature_columns.json"
    if not model_path.exists() or not columns_path.exists():
        return None, None, None
    model = joblib.load(model_path)
    feature_columns = json.loads(columns_path.read_text(encoding="utf-8"))
    importances = None
    try:
        importances = model.named_steps["model"].feature_importances_
    except Exception:
        importances = None
    return model, feature_columns, importances


def tt_to_bool(value):
    if value is None or pd.isna(value):
        return None
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def tt_to_float(value):
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def tt_format_feature_value(value):
    if value is None or pd.isna(value):
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}"


def tt_build_account_explanation(result, feature_columns=None, importances=None):
    features = result.get("features", {})
    label = result.get("label", "")
    prob = result.get("probability")

    has_pic = tt_to_bool(features.get("HasProfilePicture"))
    has_desc = tt_to_bool(features.get("HasAccountDescription"))
    verified = tt_to_bool(features.get("IsVerified"))
    is_private = tt_to_bool(features.get("IsAccountPrivate"))
    has_link = tt_to_bool(features.get("HasLinkInDescription"))
    has_instagram = tt_to_bool(features.get("HasInstagramLink"))
    has_youtube = tt_to_bool(features.get("HasYoutubeLink"))
    posts = tt_to_float(features.get("NumberOfPosts"))
    followers = tt_to_float(features.get("NumberFollowers"))
    following = tt_to_float(features.get("NumberFollowing"))
    follow_ratio = tt_to_float(features.get("FollowingToFollowerRatio"))
    likes_ratio = tt_to_float(features.get("LikesToFollowerRatio"))

    reasons_fake = []
    reasons_real = []
    if has_pic is False:
        reasons_fake.append("no profile picture")
    if has_desc is False:
        reasons_fake.append("no bio")
    if posts is not None and posts <= 2:
        reasons_fake.append(f"very few posts ({int(posts)})")
    if follow_ratio is not None and follow_ratio >= 2.0:
        reasons_fake.append(f"high follow ratio ({follow_ratio:.2f})")
    if likes_ratio is not None and likes_ratio <= 0.05:
        reasons_fake.append(f"low likes per follower ({likes_ratio:.2f})")
    if followers is not None and followers <= 50:
        reasons_fake.append(f"low followers ({int(followers)})")

    if verified is True:
        reasons_real.append("verified badge")
    if has_pic is True:
        reasons_real.append("profile picture present")
    if has_desc is True:
        reasons_real.append("bio present")
    if posts is not None and posts >= 5:
        reasons_real.append(f"several posts ({int(posts)})")
    if follow_ratio is not None and follow_ratio <= 1.5:
        reasons_real.append(f"balanced follow ratio ({follow_ratio:.2f})")
    if likes_ratio is not None and likes_ratio >= 0.1:
        reasons_real.append(f"healthy likes per follower ({likes_ratio:.2f})")
    if followers is not None and followers >= 100:
        reasons_real.append(f"meaningful followers ({int(followers)})")
    if has_link is True:
        reasons_real.append("link in bio")
    if has_instagram or has_youtube:
        reasons_real.append("connected social link")

    summary_bits = []
    if followers is not None:
        summary_bits.append(f"followers={int(followers)}")
    if following is not None:
        summary_bits.append(f"following={int(following)}")
    if posts is not None:
        summary_bits.append(f"posts={int(posts)}")
    if is_private is True:
        summary_bits.append("private=yes")
    if is_private is False:
        summary_bits.append("private=no")

    summary_text = ", ".join(summary_bits) if summary_bits else "profile metadata"
    lines = [f"Account summary: {summary_text}."]
    if feature_columns and importances is not None:
        pairs = list(zip(feature_columns, importances))
        pairs.sort(key=lambda item: item[1], reverse=True)
        top_features = []
        for name, _ in pairs[:4]:
            value = tt_format_feature_value(features.get(name))
            top_features.append(f"{name}={value}")
        if top_features:
            lines.append("Top global model features: " + ", ".join(top_features) + ".")
    if reasons_fake:
        lines.append(f"Signals leaning fake: {', '.join(reasons_fake[:4])}.")
    if reasons_real:
        lines.append(f"Signals leaning real: {', '.join(reasons_real[:4])}.")
    if not reasons_fake and not reasons_real:
        lines.append("Signals are mixed; the model relies on overall metadata patterns.")
    if label:
        lines.append(f"Model prediction: {label}.")
    if prob is not None:
        lines.append(f"Account model fake probability: {float(prob):.0%}.")
    lines.append("Final label uses a 50% threshold on the fake probability.")
    return lines


def tt_build_single_frame_explanation(pred_label, pred_conf, other_label, other_conf):
    return [
        f"Single-frame visual prediction: {pred_label}.",
        f"Confidence split: {pred_label} {pred_conf:.0%}, {other_label} {other_conf:.0%}.",
        "Decision is based on visual patterns learned from labeled TikTok frames.",
        "A 50% threshold on the fake probability determines the final label.",
    ]


def tt_build_video_frames_explanation(posts_count, frames_count, frame_fake_probs):
    avg = sum(frame_fake_probs) / len(frame_fake_probs) if frame_fake_probs else 0.0
    med = tt_median(frame_fake_probs) if frame_fake_probs else 0.0
    min_val = min(frame_fake_probs) if frame_fake_probs else 0.0
    max_val = max(frame_fake_probs) if frame_fake_probs else 0.0
    share_fake = (
        sum(1 for p in frame_fake_probs if p >= 0.5) / len(frame_fake_probs)
        if frame_fake_probs
        else 0.0
    )
    return [
        f"Analyzed {frames_count} frames from {posts_count} recent posts.",
        (
            "Frame fake-probability stats: avg "
            f"{avg:.0%}, median {med:.0%}, range {min_val:.0%}-{max_val:.0%}."
        ),
        f"{share_fake:.0%} of frames leaned fake (>= 50%).",
        "Final video decision uses the average fake probability across all frames.",
    ]


def tt_build_fusion_explanation(account_prob, video_prob, used_video):
    if not used_video:
        return [
            "No public videos available; fusion falls back to account metadata only.",
            f"Account fake probability: {account_prob:.0%}.",
            "Final label uses a 50% threshold on the fake probability.",
        ]
    fusion_prob = (account_prob * FUSION_WEIGHT_ACCOUNT) + (
        video_prob * FUSION_WEIGHT_CNN
    )
    return [
        f"Account fake probability: {account_prob:.0%}.",
        f"Video-frame fake probability: {video_prob:.0%}.",
        "Fusion uses a 70/30 weighting (account/CNN).",
        f"Weighted average: {fusion_prob:.0%} fake.",
        "Final label uses a 50% threshold on the fused probability.",
    ]


@st.cache_resource
def ig_get_account_model():
    return load_model()


def ig_get_feature_importance_df(model) -> pd.DataFrame | None:
    if not hasattr(model, "named_steps"):
        return None
    rf = model.named_steps.get("rf")
    if rf is None or not hasattr(rf, "feature_importances_"):
        return None
    importances = list(rf.feature_importances_)
    if len(importances) != len(FEATURE_ORDER):
        return None
    df = pd.DataFrame(
        {"feature": FEATURE_ORDER, "importance": importances}
    ).sort_values("importance", ascending=False)
    df["signal"] = df["feature"].map(ig_human_feature_name)
    df["influence_pct"] = (df["importance"] * 100.0).round(1)
    return df


def ig_human_feature_name(feature: str) -> str:
    return IG_FEATURE_LABELS.get(feature, feature)


def ig_format_feature_value(feature: str, value) -> str:
    if feature in {"profile pic", "external URL", "private", "name==username"}:
        return "yes" if int(value) == 1 else "no"
    if "nums/length" in feature:
        return f"{float(value):.2f}"
    if feature in {"fullname words", "#posts", "#followers", "#follows"}:
        return str(int(value))
    return str(value)


def ig_build_account_explanation(features: dict, result: dict, model) -> str:
    parts = [
        "We look at profile details (name, bio, followers, posts) and combine them to "
        "estimate how likely the account is fake."
    ]
    reasons = ig_build_account_reasons(features, result["prediction"])
    if reasons:
        parts.append("Profile cues that influenced the decision: " + ", ".join(reasons) + ".")
    else:
        parts.append("No strong profile cues stand out; the model relies on the overall pattern.")
    importance_df = ig_get_feature_importance_df(model)
    if importance_df is not None:
        top_features = importance_df.head(5)["feature"].tolist()
        top_values = [
            f"{ig_human_feature_name(name)}: {ig_format_feature_value(name, features.get(name, 0))}"
            for name in top_features
        ]
        parts.append("Most influential profile fields overall: " + ", ".join(top_values) + ".")
    parts.append(
        "Final estimate: "
        f"{result['fake_probability']:.2%} chance of being fake "
        f"(decision threshold {result['threshold']:.0%}) -> {result['prediction']}."
    )
    return " ".join(parts)


def ig_build_account_reasons(features: dict, prediction: str) -> list[str]:
    reasons: list[str] = []
    has_pfp = int(features.get("profile pic", 0))
    username_ratio = float(features.get("nums/length username", 0.0))
    fullname_ratio = float(features.get("nums/length fullname", 0.0))
    fullname_words = int(features.get("fullname words", 0))
    name_eq = int(features.get("name==username", 0))
    bio_len = int(features.get("description length", 0))
    has_url = int(features.get("external URL", 0))
    is_private = int(features.get("private", 0))
    posts = int(features.get("#posts", 0))
    followers = int(features.get("#followers", 0))
    follows = int(features.get("#follows", 0))

    follow_base = follows if follows > 0 else 1
    follow_ratio = followers / follow_base

    if prediction == "FAKE":
        if has_pfp == 0:
            reasons.append("no profile photo")
        if username_ratio >= 0.4:
            reasons.append("many digits in username")
        if fullname_ratio >= 0.4:
            reasons.append("many digits in name")
        if name_eq == 1 and fullname_words <= 1:
            reasons.append("name matches username")
        if bio_len <= 10:
            reasons.append("very short bio")
        if posts <= 3:
            reasons.append("very few posts")
        if followers <= 50:
            reasons.append("low followers")
        if follow_ratio < 0.2:
            reasons.append("follows far more than it is followed")
        if follows >= 1000:
            reasons.append("follows many accounts")
        if has_url == 0:
            reasons.append("no external link")
        if is_private == 1:
            reasons.append("account is private")
    else:
        if has_pfp == 1:
            reasons.append("has profile photo")
        if username_ratio < 0.2:
            reasons.append("few digits in username")
        if fullname_words >= 2:
            reasons.append("full name with multiple words")
        if bio_len >= 20:
            reasons.append("bio is filled in")
        if posts >= 10:
            reasons.append("enough posts")
        if followers >= 200:
            reasons.append("enough followers")
        if follow_ratio >= 1:
            reasons.append("more followers than following")
        if has_url == 1:
            reasons.append("external link present")
        if is_private == 0:
            reasons.append("account is public")

    return reasons[:5]


def ig_build_cnn_explanation(result: dict, label: str, unit_count_key: str) -> str:
    count = result.get(unit_count_key, 0)
    pattern_note = (
        "The average score is above the threshold, "
        "so the images resemble patterns the model has more often seen as fake."
        if result["prediction"] == "FAKE"
        else "The average score is below the threshold, "
        "so the images look closer to real examples."
    )
    return (
        f"The CNN model looks for visual patterns in {count} {label}. "
        f"It estimates a {result['fake_probability']:.2%} chance of being fake "
        f"(decision threshold {result['threshold']:.0%}) -> {result['prediction']}. "
        + pattern_note
    )


def ig_build_fusion_explanation(result: dict, features: dict) -> str:
    reasons = ig_build_account_reasons(features, result["prediction"])
    reasons_text = ""
    if reasons:
        reasons_text = "Profile cues: " + ", ".join(reasons) + ". "
    if result.get("cnn_probability") is None:
        return (
            "No usable visual data was found, so the decision uses profile details only. "
            + reasons_text
            + f"The profile model estimates about {result['account_probability']:.2%} fake. "
            f"Final result: {result['fake_probability']:.2%} "
            f"(decision threshold {result['threshold']:.0%}) -> {result['prediction']}."
        )
    return (
        "This result blends profile details "
        f"({result['weights']['account']:.0%}) with visual patterns from posts "
        f"({result['weights']['cnn']:.0%}). "
        + reasons_text
        + f"The profile model estimates about {result['account_probability']:.2%} fake, "
        f"and the visual model estimates about {result['cnn_probability']:.2%} fake. "
        f"Together that gives about {result['fake_probability']:.2%}. "
        f"With a decision threshold of {result['threshold']:.0%}, the final label is "
        f"{result['prediction']}."
    )


def render_instagram():
    st.header("Instagram Fake Detector")
    st.write("Choose between profile data, video frames, fusion, or a screenshot.")

    mode = st.radio(
        "Model Choice",
        (
            "Account (profile data)",
            "Screenshot (image)",
            "Video (CNN frames)",
            "Fusion (account + video)",
        ),
        horizontal=True,
        key="ig_mode",
    )

    if mode == "Account (profile data)":
        st.caption("Based on profile metadata via Instaloader.")
        with st.form("ig_account_form", border=False):
            profile_input = st.text_input(
                "Instagram username or profile URL",
                placeholder="@username or https://www.instagram.com/username",
            )
            submitted = st.form_submit_button("Predict account", type="primary")

        if not submitted:
            return

        threshold = 0.5
        if not profile_input.strip():
            st.error("Please enter a username or profile URL.")
            return

        try:
            with st.spinner("Fetching profile data..."):
                features = features_from_profile_input(profile_input)
        except (ValueError, ProfileFetchError) as exc:
            st.error(str(exc))
            return

        try:
            model = ig_get_account_model()
        except FileNotFoundError:
            st.error("Model not found. Train it first with `python -m src.train`.")
            return

        result = predict_from_features(model, features, threshold=threshold)
        st.subheader("Result")
        st.metric("Prediction", result["prediction"])
        st.metric("Fake probability", f"{result['fake_probability']:.2%}")
        st.subheader("Explanation")
        st.write(ig_build_account_explanation(features, result, model))
        importance_df = ig_get_feature_importance_df(model)
        if importance_df is not None:
            st.subheader("Most influential profile signals")
            st.caption(
                "These profile fields have the strongest overall influence on the account model; "
                "percentages show relative influence, not probability."
            )
            importance_display = importance_df.head(10)[
                ["signal", "influence_pct"]
            ].rename(columns={"signal": "Signal", "influence_pct": "Influence (%)"})
            st.dataframe(importance_display, width="stretch")
            st.bar_chart(importance_display.set_index("Signal")["Influence (%)"])
        return

    if mode == "Screenshot (image)":
        st.caption("Upload a profile screenshot and use the CNN model.")
        image_file = st.file_uploader(
            "Upload screenshot (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            key="ig_image_upload",
        )
        submitted = st.button("Predict screenshot", type="primary")

        if not submitted:
            return

        threshold = 0.5
        if image_file is None:
            st.error("Please upload a screenshot.")
            return

        try:
            image = Image.open(image_file).convert("RGB")
        except Exception:
            st.error("Could not read image file.")
            return

        st.image(image, caption="Screenshot preview")

        try:
            result = predict_from_images([image], threshold=threshold)
        except FileNotFoundError:
            st.error("CNN model not found. Train it first with `python -m src.train_cnn`.")
            return
        except ValueError as exc:
            st.error(str(exc))
            return

        st.subheader("Result")
        st.metric("Prediction", result["prediction"])
        st.metric("Fake probability", f"{result['fake_probability']:.2%}")
        st.subheader("Explanation")
        st.write(ig_build_cnn_explanation(result, "images", "image_count"))
        st.caption(f"Images used: {result['image_count']}")
        return

    if mode == "Video (CNN frames)":
        st.caption("Based on the latest video posts and a trained CNN model.")
        with st.form("ig_video_form", border=False):
            profile_input = st.text_input(
                "Instagram username or profile URL",
                placeholder="@username or https://www.instagram.com/username",
                key="ig_video_username",
            )
            submitted = st.form_submit_button("Predict videos", type="primary")

        if not submitted:
            return

        max_videos = 5
        fps = 1.0
        max_frames = 2
        threshold = 0.5
        if not profile_input.strip():
            st.error("Please enter a username or profile URL.")
            return

        try:
            username = extract_username(profile_input)
        except ValueError as exc:
            st.error(str(exc))
            return

        try:
            with st.spinner("Fetching latest posts and extracting frames..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    frame_paths, source = download_cnn_inputs_for_username(
                        username,
                        output_dir=Path(tmpdir),
                        max_videos=max_videos,
                        fps=fps,
                        max_frames_per_video=max_frames,
                    )
                    result = predict_from_frame_paths(frame_paths, threshold=threshold)
        except FileNotFoundError:
            st.error("CNN model not found. Train it first with `python -m src.train_cnn`.")
            return
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
            return

        st.subheader("Result")
        st.metric("Prediction", result["prediction"])
        st.metric("Fake probability", f"{result['fake_probability']:.2%}")
        st.subheader("Explanation")
        label = "frames" if source == "video" else "images"
        st.write(ig_build_cnn_explanation(result, label, "frame_count"))
        if source == "image":
            st.caption("No public videos found; using recent photo posts.")
            st.caption(f"Images used: {result['frame_count']}")
        else:
            st.caption(f"Frames used: {result['frame_count']}")
        return

    if mode == "Fusion (account + video)":
        st.caption("Combines profile metadata (70%) with video frames (30%).")
        with st.form("ig_fusion_form", border=False):
            profile_input = st.text_input(
                "Instagram username or profile URL",
                placeholder="@username or https://www.instagram.com/username",
                key="ig_fusion_input",
            )
            submitted = st.form_submit_button("Predict fusion", type="primary")

        if not submitted:
            return

        max_videos = 5
        fps = 1.0
        max_frames = 2
        weight_account = FUSION_WEIGHT_ACCOUNT
        threshold = 0.5
        if not profile_input.strip():
            st.error("Please enter a username or profile URL.")
            return

        try:
            with st.spinner("Fetching profile data and video frames..."):
                result = predict_fusion_from_profile_input(
                    profile_input,
                    threshold=threshold,
                    weight_account=weight_account,
                    weight_cnn=1.0 - weight_account,
                    max_videos=max_videos,
                    fps=fps,
                    max_frames_per_video=max_frames,
                )
        except FileNotFoundError:
            st.error(
                "Model not found. Train with `python -m src.train` and `python -m src.train_cnn`."
            )
            return
        except (ValueError, ProfileFetchError, RuntimeError) as exc:
            st.error(str(exc))
            return

        st.subheader("Result")
        st.metric("Prediction", result["prediction"])
        st.metric("Fake probability", f"{result['fake_probability']:.2%}")
        if not result.get("cnn_available", True):
            st.info("No visual data available; using account metadata only.")
        if result.get("cnn_error"):
            st.warning(result["cnn_error"])
        if result.get("cnn_probability") is None:
            st.caption(
                "Profile details suggest about "
                f"{result['account_probability']:.2%} fake. "
                "Visual evidence was not available."
            )
        else:
            st.caption(
                "Profile details suggest about "
                f"{result['account_probability']:.2%} fake, "
                "while visuals suggest about "
                f"{result['cnn_probability']:.2%} fake. "
                f"We combine them with a {result['weights']['account']:.0%}/"
                f"{result['weights']['cnn']:.0%} balance (profile/visual)."
            )
        if result.get("cnn_source") == "image":
            st.caption("CNN inputs: recent photo posts (no public videos found).")
        st.subheader("Explanation")
        st.write(ig_build_fusion_explanation(result, result.get("account_features", {})))
        model = ig_get_account_model()
        importance_df = ig_get_feature_importance_df(model)
        if importance_df is not None:
            st.subheader("Most influential profile signals")
            st.caption(
                "These profile fields matter most overall to the account model; "
                "percentages show relative influence, not probability."
            )
            importance_display = importance_df.head(10)[
                ["signal", "influence_pct"]
            ].rename(columns={"signal": "Signal", "influence_pct": "Influence (%)"})
            st.dataframe(importance_display, width="stretch")
        return

    st.error("Unknown model choice.")


def render_tiktok():
    st.header("TikTok Fake Account Detector")
    st.write(
        "Choose between live profile data, a single frame, multiple video frames, or a fused prediction."
    )

    mode = st.radio(
        "Model Choice",
        [
            "Account (profile data)",
            "Screenshot (image)",
            "Video (CNN frames)",
            "Fusion (account + video)",
        ],
        horizontal=True,
        key="tt_mode",
    )

    if mode == "Account (profile data)":
        username = st.text_input(
            "TikTok username or profile link",
            placeholder="@username or https://www.tiktok.com/@username",
            key="tt_account_username",
        )

        if st.button("Predict account", type="primary"):
            parsed_username = parse_username(username)
            if not parsed_username:
                st.warning("Please enter a username or profile link.")
            else:
                with st.spinner("Fetching TikTok profile and videos..."):
                    try:
                        result = predict_live(parsed_username, model_dir=TIKTOK_MODEL_DIR)
                    except RuntimeError as exc:
                        st.error(str(exc))
                        st.stop()

                st.subheader("Result")
                if result.get("probability") is not None:
                    st.metric(
                        "Prediction",
                        result["label"],
                        f"{result['probability']:.2%} bot probability",
                    )
                else:
                    st.metric("Prediction", result["label"])

                if result.get("video_count") is not None:
                    st.caption(f"Videos used: {result['video_count']}")
                if result.get("video_error"):
                    st.warning(f"Video data unavailable: {result['video_error']}")

                with st.expander("View extracted features"):
                    features = pd.Series(result.get("features", {})).sort_index()
                    st.dataframe(features, width="stretch")

                _, feature_columns, importances = tt_load_account_model(TIKTOK_MODEL_DIR)
                explanation_lines = tt_build_account_explanation(
                    result, feature_columns, importances
                )
                st.subheader("Explanation")
                st.markdown("\n".join(f"- {line}" for line in explanation_lines))
        return

    if mode == "Screenshot (image)":
        st.write("Upload one frame from a TikTok video to run the CNN image model.")
        upload = st.file_uploader(
            "Upload a video frame (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            key="tt_image_upload",
        )

        if upload is not None:
            image = Image.open(upload).convert("RGB")
            st.image(image, caption="Uploaded frame", width="stretch")

            if st.button("Predict frame", type="primary"):
                try:
                    model, classes, image_size = tt_load_image_model(TIKTOK_MODEL_DIR)
                except (FileNotFoundError, RuntimeError) as exc:
                    st.error(str(exc))
                    st.stop()

                transform = tt_build_transform(image_size)
                input_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                pred_idx = int(probs.argmax())
                pred_label = classes[pred_idx]
                pred_conf = float(probs[pred_idx])
                ranked = sorted(range(len(classes)), key=lambda i: probs[i], reverse=True)
                other_idx = ranked[1] if len(ranked) > 1 else pred_idx
                other_label = classes[other_idx]
                other_conf = float(probs[other_idx])

                st.subheader("Result")
                st.metric("Prediction", pred_label, f"{pred_conf:.2%} confidence")

                if "fake" in classes:
                    fake_idx = classes.index("fake")
                    st.metric("Fake probability", f"{probs[fake_idx]:.2%}")

                st.caption("Class probabilities")
                st.json({classes[i]: float(probs[i]) for i in range(len(classes))})

                explanation_lines = tt_build_single_frame_explanation(
                    pred_label, pred_conf, other_label, other_conf
                )
                st.subheader("Explanation")
                st.markdown("\n".join(f"- {line}" for line in explanation_lines))
        return

    if mode == "Video (CNN frames)":
        st.write("Use the latest TikTok posts and analyze multiple frames per post.")
        username = st.text_input(
            "TikTok username or profile link",
            placeholder="@username or https://www.tiktok.com/@username",
            key="tt_video_username",
        )
        st.caption("We use up to 5 recent posts and 2 frames per post.")

        if st.button("Predict videos", type="primary"):
            parsed_username = parse_username(username)
            if not parsed_username:
                st.warning("Please enter a username or profile link.")
            else:
                with st.spinner("Fetching TikTok videos and frames..."):
                    try:
                        model, classes, image_size = tt_load_image_model(TIKTOK_MODEL_DIR)
                    except (FileNotFoundError, RuntimeError) as exc:
                        st.error(str(exc))
                        st.stop()

                    try:
                        state_id, state_data = fetch_profile_state(
                            parsed_username, DEFAULT_TIMEOUT
                        )
                        user_data, _ = extract_profile_from_state(
                            parsed_username, state_id, state_data
                        )
                        if not user_data:
                            raise RuntimeError("TikTok user data incomplete.")
                        items = fetch_profile_videos(
                            parsed_username,
                            user_data,
                            state_id,
                            state_data,
                            DEFAULT_TIMEOUT,
                            limit=5,
                        )
                    except RuntimeError as exc:
                        st.error(str(exc))
                        st.stop()

                    items = items[:5]
                    frame_urls = []
                    for item in items:
                        frame_urls.extend(tt_extract_frame_urls(item, max_frames=2))

                    if not frame_urls:
                        st.error("No video frames found to analyze.")
                        st.stop()

                    images = []
                    failed = 0
                    for url in frame_urls:
                        image = tt_download_image(url, DEFAULT_TIMEOUT)
                        if image is None:
                            failed += 1
                            continue
                        images.append(image)

                    if not images:
                        st.error(
                            "Could not download video frames. TikTok may be blocking requests."
                        )
                        st.stop()

                    avg_probs, pred_idx, frame_probs = tt_score_images(
                        model, classes, image_size, images
                    )
                    pred_label = classes[pred_idx]
                    pred_conf = float(avg_probs[pred_idx])
                    fake_idx = tt_get_fake_index(classes, pred_idx)
                    frame_fake_probs = [float(p[fake_idx]) for p in frame_probs]
                    video_fake_prob = float(avg_probs[fake_idx])

                st.subheader("Result")
                st.metric(
                    "Prediction",
                    tt_format_prediction_label(pred_label),
                    f"{pred_conf:.2%} confidence",
                )
                st.caption(f"Posts analyzed: {len(items)} | Frames analyzed: {len(images)}")
                if failed:
                    st.warning(f"Frames skipped: {failed}")

                st.metric("Fake probability", f"{video_fake_prob:.2%}")

                st.caption("Class probabilities")
                st.json({classes[i]: float(avg_probs[i]) for i in range(len(classes))})

                explanation_lines = tt_build_video_frames_explanation(
                    len(items), len(images), frame_fake_probs
                )
                st.subheader("Explanation")
                st.markdown("\n".join(f"- {line}" for line in explanation_lines))
        return

    st.write(
        "Combine live profile data with CNN predictions from recent video frames "
        "(70% metadata, 30% CNN)."
    )
    username = st.text_input(
        "TikTok username or profile link",
        placeholder="@username or https://www.tiktok.com/@username",
        key="tt_fusion_username",
    )
    st.caption("We use up to 5 recent posts and 2 frames per post.")

    if st.button("Predict fusion", type="primary"):
        parsed_username = parse_username(username)
        if not parsed_username:
            st.warning("Please enter a username or profile link.")
        else:
            with st.spinner("Fetching TikTok data and frames..."):
                try:
                    model, classes, image_size = tt_load_image_model(TIKTOK_MODEL_DIR)
                except (FileNotFoundError, RuntimeError) as exc:
                    st.error(str(exc))
                    st.stop()

                try:
                    account_result = predict_live(parsed_username, model_dir=TIKTOK_MODEL_DIR)
                except RuntimeError as exc:
                    st.error(str(exc))
                    st.stop()

                video_fallback = False
                items = []
                images = []
                failed = 0
                try:
                    state_id, state_data = fetch_profile_state(
                        parsed_username, DEFAULT_TIMEOUT
                    )
                    user_data, _ = extract_profile_from_state(
                        parsed_username, state_id, state_data
                    )
                    if not user_data:
                        raise RuntimeError("TikTok user data incomplete.")
                    items = fetch_profile_videos(
                        parsed_username,
                        user_data,
                        state_id,
                        state_data,
                        DEFAULT_TIMEOUT,
                        limit=5,
                    )
                except RuntimeError as exc:
                    if "No public videos found" in str(exc):
                        video_fallback = True
                    else:
                        st.error(str(exc))
                        st.stop()

                if not video_fallback:
                    items = items[:5]
                    frame_urls = []
                    for item in items:
                        frame_urls.extend(tt_extract_frame_urls(item, max_frames=2))

                    if not frame_urls:
                        video_fallback = True
                    else:
                        for url in frame_urls:
                            image = tt_download_image(url, DEFAULT_TIMEOUT)
                            if image is None:
                                failed += 1
                                continue
                            images.append(image)

                        if not images:
                            video_fallback = True

                if not video_fallback:
                    avg_probs, pred_idx, frame_probs = tt_score_images(
                        model, classes, image_size, images
                    )
                    fake_idx = tt_get_fake_index(classes, pred_idx)
                    video_fake_prob = float(avg_probs[fake_idx])
                    video_label = classes[pred_idx]

            account_prob = account_result.get("probability")
            if account_prob is None:
                account_prob = 1.0 if account_result.get("label") == "FAKE/BOT" else 0.0

            if video_fallback:
                fusion_prob = float(account_prob)
                fusion_label = "FAKE/BOT" if fusion_prob >= 0.5 else "REAL"
            else:
                fusion_prob = (float(account_prob) * FUSION_WEIGHT_ACCOUNT) + (
                    float(video_fake_prob) * FUSION_WEIGHT_CNN
                )
                fusion_label = "FAKE/BOT" if fusion_prob >= 0.5 else "REAL"

            st.subheader("Result")
            st.metric("Prediction", fusion_label, f"{fusion_prob:.2%} fake probability")
            if video_fallback:
                st.info(
                    "No public videos found for this account, using account information."
                )
            else:
                st.caption(f"Posts analyzed: {len(items)} | Frames analyzed: {len(images)}")
                if failed:
                    st.warning(f"Frames skipped: {failed}")

            st.subheader("Modalities")
            st.metric(
                "Account model (fake prob)",
                f"{float(account_prob):.2%}",
                account_result.get("label"),
            )
            if not video_fallback:
                st.metric(
                    "Video model (fake prob)",
                    f"{float(video_fake_prob):.2%}",
                    tt_format_prediction_label(video_label),
                )
            explanation_lines = tt_build_fusion_explanation(
                float(account_prob),
                float(video_fake_prob) if not video_fallback else 0.0,
                not video_fallback,
            )
            st.subheader("Explanation")
            st.markdown("\n".join(f"- {line}" for line in explanation_lines))


st.title("Fake Profile Detector")
st.write("Select a platform to start.")

platform = st.radio(
    "Platform",
    ("Instagram", "TikTok"),
    horizontal=True,
)

if platform == "Instagram":
    render_instagram()
else:
    render_tiktok()
