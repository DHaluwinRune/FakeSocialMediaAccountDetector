from __future__ import annotations

import tempfile
from pathlib import Path

from .cnn_live import download_cnn_inputs_for_username
from .cnn_predict import predict_from_frame_paths
from .instagram import extract_username, features_from_profile_input
from .predict import load_model, predict_from_features


def _normalize_weights(weight_account: float, weight_cnn: float) -> tuple[float, float]:
    if weight_account < 0 or weight_cnn < 0:
        raise ValueError("Weights must be non-negative.")
    total = weight_account + weight_cnn
    if total <= 0:
        raise ValueError("At least one weight must be > 0.")
    return weight_account / total, weight_cnn / total


def fuse_probabilities(
    account_prob: float,
    cnn_prob: float,
    threshold: float = 0.5,
    weight_account: float = 0.7,
    weight_cnn: float = 0.3,
) -> dict:
    weight_account, weight_cnn = _normalize_weights(weight_account, weight_cnn)
    combined = (account_prob * weight_account) + (cnn_prob * weight_cnn)
    return {
        "prediction": "FAKE" if combined >= threshold else "REAL",
        "fake_probability": float(combined),
        "threshold": float(threshold),
        "weights": {
            "account": float(weight_account),
            "cnn": float(weight_cnn),
        },
    }


def _predict_cnn_for_username(
    username: str,
    output_dir: Path,
    threshold: float,
    max_videos: int,
    fps: float,
    max_frames_per_video: int,
) -> dict:
    frame_paths, source = download_cnn_inputs_for_username(
        username,
        output_dir=output_dir,
        max_videos=max_videos,
        fps=fps,
        max_frames_per_video=max_frames_per_video,
    )
    result = predict_from_frame_paths(frame_paths, threshold=threshold)
    result["source"] = source
    return result


def predict_fusion_from_profile_input(
    profile_input: str,
    threshold: float = 0.5,
    weight_account: float = 0.7,
    weight_cnn: float = 0.3,
    max_videos: int = 5,
    fps: float = 1.0,
    max_frames_per_video: int = 10,
    output_dir: Path | None = None,
) -> dict:
    username = extract_username(profile_input)

    features = features_from_profile_input(profile_input)
    account_model = load_model()
    account_result = predict_from_features(account_model, features, threshold=threshold)
    account_prob = float(account_result["fake_probability"])

    cnn_result = None
    cnn_error = None
    try:
        if output_dir is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                cnn_result = _predict_cnn_for_username(
                    username,
                    output_dir=Path(tmpdir),
                    threshold=threshold,
                    max_videos=max_videos,
                    fps=fps,
                    max_frames_per_video=max_frames_per_video,
                )
        else:
            cnn_result = _predict_cnn_for_username(
                username,
                output_dir=output_dir,
                threshold=threshold,
                max_videos=max_videos,
                fps=fps,
                max_frames_per_video=max_frames_per_video,
            )
    except RuntimeError as exc:
        cnn_error = str(exc)

    if cnn_result is None:
        fused = {
            "prediction": account_result["prediction"],
            "fake_probability": float(account_prob),
            "threshold": float(threshold),
            "weights": {"account": 1.0, "cnn": 0.0},
        }
        fused["account_prediction"] = account_result["prediction"]
        fused["account_probability"] = account_prob
        fused["account_features"] = features
        fused["cnn_prediction"] = None
        fused["cnn_probability"] = None
        fused["cnn_source"] = None
        fused["cnn_available"] = False
        fused["cnn_error"] = cnn_error
        return fused

    fused = fuse_probabilities(
        account_prob,
        float(cnn_result["fake_probability"]),
        threshold=threshold,
        weight_account=weight_account,
        weight_cnn=weight_cnn,
    )
    fused["account_prediction"] = account_result["prediction"]
    fused["account_probability"] = account_prob
    fused["account_features"] = features
    fused["cnn_prediction"] = cnn_result["prediction"]
    fused["cnn_probability"] = float(cnn_result["fake_probability"])
    fused["cnn_source"] = cnn_result.get("source", "video")
    fused["cnn_available"] = True
    fused["cnn_error"] = None
    return fused
