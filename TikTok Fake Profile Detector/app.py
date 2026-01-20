import io
import json
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

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
FUSION_WEIGHT_ACCOUNT = 0.7
FUSION_WEIGHT_CNN = 0.3


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
def load_image_model(model_dir):
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


def build_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def extract_frame_urls(item, max_frames=3):
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


def download_image(url, timeout):
    try:
        response = requests.get(url, headers=TIKTOK_HEADERS, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return None
    try:
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception:
        return None


def score_images(model, classes, image_size, images):
    transform = build_transform(image_size)
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


def format_prediction_label(class_name):
    label = class_name.lower()
    if "fake" in label or "bot" in label:
        return "FAKE/BOT"
    if "real" in label:
        return "REAL"
    return class_name.upper()


def pick_fake_probability(classes, probs, pred_idx):
    lowered = [name.lower() for name in classes]
    for keyword in ("fake", "bot"):
        if keyword in lowered:
            return float(probs[lowered.index(keyword)]), classes[lowered.index(keyword)]
    return float(probs[pred_idx]), classes[pred_idx]


def get_fake_index(classes, fallback_idx):
    for idx, name in enumerate(classes):
        label = name.lower()
        if "fake" in label or "bot" in label:
            return idx
    return fallback_idx


def median(values):
    if not values:
        return None
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2


@st.cache_resource
def load_account_model(model_dir):
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


def to_bool(value):
    if value is None or pd.isna(value):
        return None
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def to_float(value):
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_feature_value(value):
    if value is None or pd.isna(value):
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}"


def build_account_explanation(result, feature_columns=None, importances=None):
    features = result.get("features", {})
    label = result.get("label", "")
    prob = result.get("probability")

    has_pic = to_bool(features.get("HasProfilePicture"))
    has_desc = to_bool(features.get("HasAccountDescription"))
    verified = to_bool(features.get("IsVerified"))
    is_private = to_bool(features.get("IsAccountPrivate"))
    has_link = to_bool(features.get("HasLinkInDescription"))
    has_instagram = to_bool(features.get("HasInstagramLink"))
    has_youtube = to_bool(features.get("HasYoutubeLink"))
    posts = to_float(features.get("NumberOfPosts"))
    followers = to_float(features.get("NumberFollowers"))
    following = to_float(features.get("NumberFollowing"))
    follow_ratio = to_float(features.get("FollowingToFollowerRatio"))
    likes_ratio = to_float(features.get("LikesToFollowerRatio"))

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
            value = format_feature_value(features.get(name))
            top_features.append(f"{name}={value}")
        if top_features:
            lines.append(
                "Top global model features: " + ", ".join(top_features) + "."
            )
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


def build_single_frame_explanation(pred_label, pred_conf, other_label, other_conf):
    return [
        f"Single-frame visual prediction: {pred_label}.",
        f"Confidence split: {pred_label} {pred_conf:.0%}, {other_label} {other_conf:.0%}.",
        "Decision is based on visual patterns learned from labeled TikTok frames.",
        "A 50% threshold on the fake probability determines the final label.",
    ]


def build_video_frames_explanation(posts_count, frames_count, frame_fake_probs):
    avg = sum(frame_fake_probs) / len(frame_fake_probs) if frame_fake_probs else 0.0
    med = median(frame_fake_probs) if frame_fake_probs else 0.0
    min_val = min(frame_fake_probs) if frame_fake_probs else 0.0
    max_val = max(frame_fake_probs) if frame_fake_probs else 0.0
    share_fake = (
        sum(1 for p in frame_fake_probs if p >= 0.5) / len(frame_fake_probs)
        if frame_fake_probs
        else 0.0
    )
    return [
        f"Analyzed {frames_count} frames from {posts_count} recent posts.",
        f"Frame fake-probability stats: avg {avg:.0%}, median {med:.0%}, range {min_val:.0%}–{max_val:.0%}.",
        f"{share_fake:.0%} of frames leaned fake (>= 50%).",
        "Final video decision uses the average fake probability across all frames.",
    ]


def build_fusion_explanation(account_prob, video_prob, used_video):
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


st.set_page_config(page_title="TikTok Fake Account Detector", layout="centered")
st.title("TikTok Fake Account Detector")
st.write(
    "Choose between live profile data, a single frame, multiple video frames, or a fused prediction."
)

mode = st.radio(
    "Model keuze",
    [
        "Account (profile data)",
        "Screenshot (image)",
        "Video (CNN frames)",
        "Fusion (account + video)",
    ],
    horizontal=True,
)

if mode == "Account (profile data)":
    username = st.text_input(
        "TikTok username or profile link",
        placeholder="@username or https://www.tiktok.com/@username",
    )

    if st.button("Predict account", type="primary"):
        parsed_username = parse_username(username)
        if not parsed_username:
            st.warning("Please enter a username or profile link.")
        else:
            with st.spinner("Fetching TikTok profile and videos..."):
                try:
                    result = predict_live(parsed_username, model_dir="models")
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
                st.dataframe(features, use_container_width=True)

            _, feature_columns, importances = load_account_model("models")
            explanation_lines = build_account_explanation(
                result, feature_columns, importances
            )
            st.subheader("Explanation")
            st.markdown("\n".join(f"- {line}" for line in explanation_lines))
elif mode in {"Screenshot (image)", "Video frame (CNN)"}:
    st.write("Upload one frame from a TikTok video to run the CNN image model.")
    upload = st.file_uploader(
        "Upload a video frame (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if upload is not None:
        image = Image.open(upload).convert("RGB")
        st.image(image, caption="Uploaded frame", use_container_width=True)

        if st.button("Predict frame", type="primary"):
            try:
                model, classes, image_size = load_image_model("models")
            except (FileNotFoundError, RuntimeError) as exc:
                st.error(str(exc))
                st.stop()

            transform = build_transform(image_size)
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

            explanation_lines = build_single_frame_explanation(
                pred_label, pred_conf, other_label, other_conf
            )
            st.subheader("Explanation")
            st.markdown("\n".join(f"- {line}" for line in explanation_lines))
elif mode == "Video (CNN frames)":
    st.write("Use the latest TikTok posts and analyze multiple frames per post.")
    username = st.text_input(
        "TikTok username or profile link",
        placeholder="@username or https://www.tiktok.com/@username",
    )
    st.caption("We use up to 5 recent posts and 2 frames per post.")

    if st.button("Predict videos", type="primary"):
        parsed_username = parse_username(username)
        if not parsed_username:
            st.warning("Please enter a username or profile link.")
        else:
            with st.spinner("Fetching TikTok videos and frames..."):
                try:
                    model, classes, image_size = load_image_model("models")
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
                    frame_urls.extend(extract_frame_urls(item, max_frames=2))

                if not frame_urls:
                    st.error("No video frames found to analyze.")
                    st.stop()

                images = []
                failed = 0
                for url in frame_urls:
                    image = download_image(url, DEFAULT_TIMEOUT)
                    if image is None:
                        failed += 1
                        continue
                    images.append(image)

                if not images:
                    st.error(
                        "Could not download video frames. TikTok may be blocking requests."
                    )
                    st.stop()

                avg_probs, pred_idx, frame_probs = score_images(
                    model, classes, image_size, images
                )
                pred_label = classes[pred_idx]
                pred_conf = float(avg_probs[pred_idx])
                fake_idx = get_fake_index(classes, pred_idx)
                frame_fake_probs = [float(p[fake_idx]) for p in frame_probs]
                video_fake_prob = float(avg_probs[fake_idx])

            st.subheader("Result")
            st.metric(
                "Prediction",
                format_prediction_label(pred_label),
                f"{pred_conf:.2%} confidence",
            )
            st.caption(f"Posts analyzed: {len(items)} | Frames analyzed: {len(images)}")
            if failed:
                st.warning(f"Frames skipped: {failed}")

            st.metric("Fake probability", f"{video_fake_prob:.2%}")

            st.caption("Class probabilities")
            st.json({classes[i]: float(avg_probs[i]) for i in range(len(classes))})

            explanation_lines = build_video_frames_explanation(
                len(items), len(images), frame_fake_probs
            )
            st.subheader("Explanation")
            st.markdown("\n".join(f"- {line}" for line in explanation_lines))
elif mode == "Fusion (account + video)":
    st.write(
        "Combine live profile data with CNN predictions from recent video frames "
        "(70% metadata, 30% CNN)."
    )
    username = st.text_input(
        "TikTok username or profile link",
        placeholder="@username or https://www.tiktok.com/@username",
    )
    st.caption("We use up to 5 recent posts and 2 frames per post.")

    if st.button("Predict fusion", type="primary"):
        parsed_username = parse_username(username)
        if not parsed_username:
            st.warning("Please enter a username or profile link.")
        else:
            with st.spinner("Fetching TikTok data and frames..."):
                try:
                    model, classes, image_size = load_image_model("models")
                except (FileNotFoundError, RuntimeError) as exc:
                    st.error(str(exc))
                    st.stop()

                try:
                    account_result = predict_live(parsed_username, model_dir="models")
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
                        frame_urls.extend(extract_frame_urls(item, max_frames=2))

                    if not frame_urls:
                        video_fallback = True
                    else:
                        for url in frame_urls:
                            image = download_image(url, DEFAULT_TIMEOUT)
                            if image is None:
                                failed += 1
                                continue
                            images.append(image)

                        if not images:
                            video_fallback = True

                if not video_fallback:
                    avg_probs, pred_idx, frame_probs = score_images(
                        model, classes, image_size, images
                    )
                    fake_idx = get_fake_index(classes, pred_idx)
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
                st.caption(
                    f"Posts analyzed: {len(items)} | Frames analyzed: {len(images)}"
                )
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
                    format_prediction_label(video_label),
                )
            explanation_lines = build_fusion_explanation(
                float(account_prob),
                float(video_fake_prob) if not video_fallback else 0.0,
                not video_fallback,
            )
            st.subheader("Explanation")
            st.markdown("\n".join(f"- {line}" for line in explanation_lines))
