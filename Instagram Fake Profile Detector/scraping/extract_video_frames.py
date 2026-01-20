import argparse
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import instaloader
import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_input_csv() -> Path:
    data_dir = _project_root() / "Data"
    if not data_dir.exists():
        data_dir = _project_root() / "data"
    return data_dir / "ig_post_metadata.csv"


def _default_output_dir() -> Path:
    data_dir = _project_root() / "Data"
    if not data_dir.exists():
        data_dir = _project_root() / "data"
    return data_dir / "video_frames"


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


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "please wait a few minutes" in msg or "too many requests" in msg or "429" in msg


def _fetch_post(loader, shortcode: str, max_retries: int, backoff: float):
    for attempt in range(max_retries + 1):
        try:
            return instaloader.Post.from_shortcode(loader.context, shortcode)
        except Exception as exc:
            if _is_rate_limited(exc) and attempt < max_retries:
                wait = backoff * (2**attempt)
                print(f"[rate-limit] {shortcode}: sleeping {wait:.1f}s")
                time.sleep(wait)
                continue
            return None
    return None


def _load_video_posts(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required = {"username", "shortcode", "is_video"}
    missing = required.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns in ig_post_metadata.csv: {missing_list}")

    df = df.copy()
    df["username"] = df["username"].astype(str).str.strip().str.lstrip("@").str.lower()
    df["shortcode"] = df["shortcode"].astype(str).str.strip()
    df["is_video"] = pd.to_numeric(df["is_video"], errors="coerce").fillna(0).astype(int)

    df = df[df["is_video"] == 1]
    df = df[df["shortcode"] != ""]

    return df[["username", "shortcode"]].drop_duplicates()


def _download_video(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, dest.open("wb") as fh:
        shutil.copyfileobj(response, fh)


def _extract_frames(
    ffmpeg_path: str,
    video_path: Path,
    output_dir: Path,
    fps: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "frame_%06d.jpg")
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        pattern,
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg failed to extract frames")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Instagram videos from ig_post_metadata.csv and extract frames."
    )
    parser.add_argument("--input", default=str(_default_input_csv()), help="CSV path")
    parser.add_argument("--output-dir", default=str(_default_output_dir()), help="Output folder")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of videos")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between videos")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries on rate limits")
    parser.add_argument("--backoff", type=float, default=60.0, help="Initial backoff seconds")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if frames already exist")
    parser.add_argument("--keep-video", action="store_true", help="Keep downloaded video files")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise SystemExit("ffmpeg not found in PATH. Install ffmpeg to extract frames.")

    loader = _init_instaloader()

    usernames = _load_video_posts(input_path)
    if usernames.empty:
        raise SystemExit("No video posts found in ig_post_metadata.csv.")

    processed = 0
    for _, row in usernames.iterrows():
        if args.limit and processed >= args.limit:
            break

        username = row["username"] or "unknown"
        shortcode = row["shortcode"]
        target_dir = output_dir / username / shortcode
        if args.skip_existing and any(target_dir.glob("frame_*.jpg")):
            print(f"[skip] {shortcode} already extracted")
            continue

        post = _fetch_post(loader, shortcode, args.max_retries, args.backoff)
        if post is None:
            print(f"[skip] {shortcode} fetch error")
            continue

        if not post.is_video:
            print(f"[skip] {shortcode} is not video")
            continue

        video_url = post.video_url
        if not video_url:
            print(f"[skip] {shortcode} missing video url")
            continue

        video_path = target_dir / "video.mp4"
        try:
            _download_video(video_url, video_path)
            _extract_frames(ffmpeg_path, video_path, target_dir, args.fps)
        except Exception as exc:
            print(f"[error] {shortcode}: {exc}")
            if video_path.exists():
                video_path.unlink()
            continue

        if not args.keep_video and video_path.exists():
            video_path.unlink()

        processed += 1
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Done. Extracted frames for {processed} videos.")


if __name__ == "__main__":
    main()
