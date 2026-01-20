import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.cnn_live import download_video_frames_for_username
from src.cnn_predict import predict_from_frame_paths, predict_from_images
from src.config import FEATURE_ORDER
from src.fusion_predict import predict_fusion_from_profile_input
from src.instagram import ProfileFetchError, features_from_profile_input
from src.predict import load_model, predict_from_features


@st.cache_resource
def get_account_model():
    return load_model()

def get_feature_importance_df(model) -> pd.DataFrame | None:
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
    return df


def _format_feature_value(feature: str, value) -> str:
    if feature in {"profile pic", "external URL", "private", "name==username"}:
        return "ja" if int(value) == 1 else "nee"
    if "nums/length" in feature:
        return f"{float(value):.2f}"
    if feature in {"fullname words", "#posts", "#followers", "#follows"}:
        return str(int(value))
    return str(value)


def build_account_explanation(features: dict, result: dict, model) -> str:
    parts = [
        "Het account-model gebruikt profielmetadata (username/bio/volgers/posts) "
        "en combineert deze signalen in een score."
    ]
    reasons = build_account_reasons(features, result["prediction"])
    if reasons:
        parts.append("Signalen die richting deze beslissing wijzen: " + ", ".join(reasons) + ".")
    else:
        parts.append("Geen duidelijke dominante signalen gevonden in de metadata.")
    importance_df = get_feature_importance_df(model)
    if importance_df is not None:
        top_features = importance_df.head(5)["feature"].tolist()
        top_values = [
            f"{name}={_format_feature_value(name, features.get(name, 0))}"
            for name in top_features
        ]
        parts.append("Belangrijkste (globale) kenmerken: " + ", ".join(top_values) + ".")
    parts.append(
        "Modelscore: "
        f"{result['fake_probability']:.4f} "
        f"(drempel {result['threshold']:.2f}) -> {result['prediction']}."
    )
    return " ".join(parts)


def build_account_reasons(features: dict, prediction: str) -> list[str]:
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
            reasons.append("geen profielfoto")
        if username_ratio >= 0.4:
            reasons.append("veel cijfers in username")
        if fullname_ratio >= 0.4:
            reasons.append("veel cijfers in naam")
        if name_eq == 1 and fullname_words <= 1:
            reasons.append("naam gelijk aan username")
        if bio_len <= 10:
            reasons.append("zeer korte bio")
        if posts <= 3:
            reasons.append("weinig posts")
        if followers <= 50:
            reasons.append("weinig volgers")
        if follow_ratio < 0.2:
            reasons.append("volgt veel meer dan het gevolgd wordt")
        if follows >= 1000:
            reasons.append("volgt erg veel accounts")
        if has_url == 0:
            reasons.append("geen externe link")
        if is_private == 1:
            reasons.append("account is prive")
    else:
        if has_pfp == 1:
            reasons.append("heeft profielfoto")
        if username_ratio < 0.2:
            reasons.append("weinig cijfers in username")
        if fullname_words >= 2:
            reasons.append("volledige naam met meerdere woorden")
        if bio_len >= 20:
            reasons.append("bio is ingevuld")
        if posts >= 10:
            reasons.append("voldoende posts")
        if followers >= 200:
            reasons.append("voldoende volgers")
        if follow_ratio >= 1:
            reasons.append("meer volgers dan volgend")
        if has_url == 1:
            reasons.append("externe link aanwezig")
        if is_private == 0:
            reasons.append("account is publiek")

    return reasons[:5]


def build_cnn_explanation(result: dict, label: str, unit_count_key: str) -> str:
    count = result.get(unit_count_key, 0)
    verdict = "boven" if result["prediction"] == "FAKE" else "onder"
    pattern_note = (
        "De gemiddelde score ligt boven de drempel, "
        "dus de beelden lijken op patronen die het model vaker als fake zag."
        if result["prediction"] == "FAKE"
        else "De gemiddelde score ligt onder de drempel, "
        "dus de beelden lijken meer op echte voorbeelden."
    )
    return (
        f"Het CNN-model gebruikt visuele patronen in {count} {label}. "
        f"Modelscore: {result['fake_probability']:.4f} "
        f"(drempel {result['threshold']:.2f}) -> {result['prediction']}. "
        + pattern_note
    )


def build_fusion_explanation(result: dict, features: dict) -> str:
    reasons = build_account_reasons(features, result["prediction"])
    reasons_text = ""
    if reasons:
        reasons_text = "Account-signalen: " + ", ".join(reasons) + ". "
    if result.get("cnn_probability") is None:
        return (
            "Geen visuele data beschikbaar; voorspelling gebeurt enkel op account-metadata. "
            + reasons_text +
            f"Account prob={result['account_probability']:.4f}. "
            f"Score: {result['fake_probability']:.4f} "
            f"(drempel {result['threshold']:.2f}) -> {result['prediction']}."
        )
    return (
        "De fusion-score is een gewogen gemiddelde van account- en CNN-signalen. "
        + reasons_text +
        f"Account prob={result['account_probability']:.4f}, "
        f"CNN prob={result['cnn_probability']:.4f}, "
        f"gewichten={result['weights']['account']:.2f}/"
        f"{result['weights']['cnn']:.2f}. "
        f"Gecombineerde score: {result['fake_probability']:.4f} "
        f"(drempel {result['threshold']:.2f}) -> {result['prediction']}."
    )


def main():
    st.set_page_config(page_title="Instagram Fake Detector", page_icon="🔎")
    st.title("Instagram Fake Detector")
    st.write("Kies welk model je wil gebruiken en vul een username of URL in.")

    mode = st.radio(
        "Model keuze",
        (
            "Account (profieldata)",
            "Video (CNN frames)",
            "Fusion (account + video)",
            "Screenshot (image)",
        ),
        horizontal=True,
    )

    if mode == "Account (profieldata)":
        st.caption("Gebaseerd op profielmetadata via Instaloader.")
        with st.form("account_form"):
            profile_input = st.text_input("Instagram username of URL")
            threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
            submitted = st.form_submit_button("Check account")

        if not submitted:
            st.info("Tip: private accounts may require IG_USERNAME/IG_PASSWORD env vars.")
            return

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
            model = get_account_model()
        except FileNotFoundError:
            st.error("Model not found. Train it first with `python -m src.train`.")
            return

        result = predict_from_features(model, features, threshold=threshold)
        st.subheader("Result")
        st.write(f"Prediction: **{result['prediction']}**")
        st.metric("Fake probability", f"{result['fake_probability']:.4f}")
        st.subheader("Uitleg")
        st.write(build_account_explanation(features, result, model))
        importance_df = get_feature_importance_df(model)
        if importance_df is not None:
            st.subheader("Belangrijkste kenmerken")
            st.caption("Globale feature importance uit het RandomForest model.")
            st.dataframe(importance_df.head(10), use_container_width=True)
            st.bar_chart(
                importance_df.head(10).set_index("feature")["importance"]
            )
        return

    if mode == "Fusion (account + video)":
        st.caption("Combineert profielmetadata met videoframes.")
        with st.form("fusion_form"):
            profile_input = st.text_input("Instagram username of URL", key="fusion_input")
            max_videos = st.slider("Aantal laatste video posts", 1, 20, 5, 1, key="fusion_max_videos")
            fps = st.slider("Frames per seconde", 0.5, 5.0, 1.0, 0.5, key="fusion_fps")
            max_frames = st.slider("Max frames per video", 1, 30, 10, 1, key="fusion_max_frames")
            weight_account = st.slider("Gewicht account model", 0.0, 1.0, 0.5, 0.05, key="fusion_weight")
            threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05, key="fusion_threshold")
            submitted = st.form_submit_button("Check account")

        if not submitted:
            st.info("Tip: zorg dat beide modellen getraind zijn en ffmpeg beschikbaar is.")
            return

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
        st.write(f"Prediction: **{result['prediction']}**")
        st.metric("Fake probability", f"{result['fake_probability']:.4f}")
        if not result.get("cnn_available", True):
            st.info("Geen visuele data beschikbaar; gebruikt enkel profieldata.")
        cnn_prob_text = (
            f"{result['cnn_probability']:.4f}"
            if result.get("cnn_probability") is not None
            else "n.v.t."
        )
        st.caption(
            "Account prob: "
            f"{result['account_probability']:.4f} | "
            "CNN prob: "
            f"{cnn_prob_text} | "
            "Weights: "
            f"{result['weights']['account']:.2f}/"
            f"{result['weights']['cnn']:.2f}"
        )
        st.subheader("Uitleg")
        st.write(build_fusion_explanation(result, result.get("account_features", {})))
        model = get_account_model()
        importance_df = get_feature_importance_df(model)
        if importance_df is not None:
            st.subheader("Belangrijkste kenmerken (account model)")
            st.caption("Globale feature importance uit het RandomForest model.")
            st.dataframe(importance_df.head(10), use_container_width=True)
        return

    if mode == "Screenshot (image)":
        st.caption("Upload een screenshot van een profiel en gebruik het CNN model.")
        image_file = st.file_uploader(
            "Upload screenshot (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
        )
        threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05, key="image_threshold")
        submitted = st.button("Check screenshot")

        if not submitted:
            st.info("Tip: neem een screenshot en upload het bestand.")
            return

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
        st.write(f"Prediction: **{result['prediction']}**")
        st.metric("Fake probability", f"{result['fake_probability']:.4f}")
        st.subheader("Uitleg")
        st.write(build_cnn_explanation(result, "images", "image_count"))
        st.caption(f"Images used: {result['image_count']}")
        return

    st.caption("Gebaseerd op de nieuwste videoposts en een getraind CNN model.")
    with st.form("video_form"):
        username = st.text_input("Instagram username")
        max_videos = st.slider("Aantal laatste video posts", 1, 20, 5, 1)
        fps = st.slider("Frames per seconde", 0.5, 5.0, 1.0, 0.5)
        max_frames = st.slider("Max frames per video", 1, 30, 10, 1)
        threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05, key="video_threshold")
        submitted = st.form_submit_button("Check account")

    if not submitted:
        st.info("Tip: zorg dat het CNN model getraind is en ffmpeg beschikbaar is.")
        return

    if not username.strip():
        st.error("Please enter a username.")
        return

    try:
        with st.spinner("Downloading latest videos and extracting frames..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                frame_paths = download_video_frames_for_username(
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
    st.write(f"Prediction: **{result['prediction']}**")
    st.metric("Fake probability", f"{result['fake_probability']:.4f}")
    st.subheader("Uitleg")
    st.write(build_cnn_explanation(result, "frames", "frame_count"))
    st.caption(f"Frames used: {result['frame_count']}")


if __name__ == "__main__":
    main()
