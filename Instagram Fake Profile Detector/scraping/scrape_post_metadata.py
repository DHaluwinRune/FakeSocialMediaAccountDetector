import argparse
import csv
import os
import time
from pathlib import Path

import instaloader

from src.features import build_feature_row
from src.predict import load_model, predict_from_features


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_output_csv() -> Path:
    data_dir = _project_root() / "Data"
    if not data_dir.exists():
        data_dir = _project_root() / "data"
    return data_dir / "ig_post_metadata.csv"


def _init_instaloader() -> instaloader.Instaloader:
    loader = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        quiet=True,
    )
    ig_user = os.getenv("IG_USERNAME")
    ig_pass = os.getenv("IG_PASSWORD")
    if ig_user and ig_pass:
        loader.login(ig_user, ig_pass)
    return loader


def _load_usernames(input_path: Path) -> list[str]:
    if input_path.suffix.lower() == ".csv":
        with input_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                raise ValueError("Input CSV is empty or missing headers.")
            fieldnames = {name.strip().lower(): name for name in reader.fieldnames}
            key = fieldnames.get("username") or fieldnames.get("user") or fieldnames.get("account")
            if not key:
                raise ValueError("CSV must include a username column (e.g. 'username').")
            return [
                (row.get(key) or "").strip().lstrip("@")
                for row in reader
                if (row.get(key) or "").strip()
            ]

    with input_path.open("r", encoding="utf-8") as fh:
        return [line.strip().lstrip("@") for line in fh if line.strip()]


def _profile_to_features(profile) -> dict:
    has_pfp_default = getattr(profile, "is_profile_pic_default", None)
    if has_pfp_default is None:
        has_pfp = int(bool(getattr(profile, "profile_pic_url", None)))
    else:
        has_pfp = int(not has_pfp_default)

    return build_feature_row(
        username=getattr(profile, "username", "") or "",
        fullname=getattr(profile, "full_name", "") or "",
        bio=getattr(profile, "biography", "") or "",
        has_url=int(bool(getattr(profile, "external_url", None))),
        is_private=int(bool(getattr(profile, "is_private", False))),
        has_pfp=has_pfp,
        posts=int(getattr(profile, "mediacount", 0) or 0),
        followers=int(getattr(profile, "followers", 0) or 0),
        follows=int(getattr(profile, "followees", 0) or 0),
    )


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "please wait a few minutes" in msg or "too many requests" in msg or "429" in msg


def _fetch_profile(
    loader,
    username: str,
    max_retries: int,
    backoff: float,
    skip_on_error: bool,
):
    if skip_on_error:
        try:
            return instaloader.Profile.from_username(loader.context, username)
        except Exception:
            return None
    for attempt in range(max_retries + 1):
        try:
            return instaloader.Profile.from_username(loader.context, username)
        except Exception as exc:
            if _is_rate_limited(exc) and attempt < max_retries:
                wait = backoff * (2**attempt)
                print(f"[rate-limit] {username}: sleeping {wait:.1f}s")
                time.sleep(wait)
                continue
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape post metadata and label profiles with the existing model."
    )
    parser.add_argument("--input", required=True, help="CSV/TXT with usernames")
    parser.add_argument("--output", default=str(_default_output_csv()), help="Output CSV path")
    parser.add_argument("--per-profile", type=int, default=20, help="Posts per profile")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fake threshold")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between profiles")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries on rate limits")
    parser.add_argument("--backoff", type=float, default=60.0, help="Initial backoff seconds")
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        help="Skip failures immediately (no retries/backoff)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    usernames = [u for u in _load_usernames(input_path) if u]
    if not usernames:
        raise SystemExit("No usernames found to process.")

    try:
        model = load_model()
    except FileNotFoundError:
        raise SystemExit("Model not found. Train it first with `python -m src.train`.") from None

    loader = _init_instaloader()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "username",
                "label_fake",
                "fake_probability",
                "post_index",
                "shortcode",
                "timestamp_utc",
                "is_video",
                "likes",
                "comments",
                "video_view_count",
                "caption_length",
            ]
        )

        for idx, username in enumerate(usernames, start=1):
            profile = _fetch_profile(
                loader,
                username,
                args.max_retries,
                args.backoff,
                args.skip_on_error,
            )
            if profile is None:
                print(f"[skip] {username} fetch error")
                continue

            if profile.is_private:
                print(f"[skip] {username} is private")
                continue

            features = _profile_to_features(profile)
            result = predict_from_features(model, features, threshold=args.threshold)
            label_fake = 1 if result["prediction"] == "FAKE" else 0
            score = result["fake_probability"]

            for i, post in enumerate(profile.get_posts()):
                if i >= args.per_profile:
                    break
                timestamp = getattr(post, "date_utc", None)
                caption = getattr(post, "caption", None) or ""
                writer.writerow(
                    [
                        username,
                        label_fake,
                        f"{score:.6f}",
                        i,
                        getattr(post, "shortcode", "") or "",
                        timestamp.isoformat() if timestamp else "",
                        int(bool(getattr(post, "is_video", False))),
                        getattr(post, "likes", "") or "",
                        getattr(post, "comments", "") or "",
                        getattr(post, "video_view_count", "") or "",
                        len(caption.strip()),
                    ]
                )

            if args.sleep > 0:
                time.sleep(args.sleep)

            if idx % 50 == 0:
                print(f"Processed {idx}/{len(usernames)} profiles")


if __name__ == "__main__":
    main()
