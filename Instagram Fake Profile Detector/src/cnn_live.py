from __future__ import annotations

import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import instaloader


def _init_instaloader() -> instaloader.Instaloader:
    loader = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        quiet=True,
        max_connection_attempts=1,
        request_timeout=30.0,
        fatal_status_codes=[403],
    )
    ig_user = os.getenv("IG_USERNAME")
    ig_pass = os.getenv("IG_PASSWORD")
    if ig_user and ig_pass:
        loader.login(ig_user, ig_pass)
    return loader


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "please wait a few minutes" in msg or "too many requests" in msg or "429" in msg


def _format_access_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "403" in msg or "forbidden" in msg or "login" in msg or "private" in msg:
        return "Instagram account is private; post data cannot be accessed."
    if _is_rate_limited(exc):
        return "Instagram rate limited the request. Try again later."
    return "Failed to fetch Instagram posts."


def _fetch_profile(
    loader,
    username: str,
    max_retries: int,
    backoff: float,
):
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


def _download_video(loader, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = None
    if loader is not None:
        try:
            response = loader.context.get_raw(url)
            response.raise_for_status()
            with dest.open("wb") as fh:
                shutil.copyfileobj(response.raw, fh)
            return
        except Exception:
            pass
        finally:
            if response is not None:
                response.close()

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def _iter_profile_posts(profile):
    seen: set[str] = set()
    try:
        for post in profile.get_posts():
            shortcode = getattr(post, "shortcode", "") or ""
            if shortcode and shortcode in seen:
                continue
            if shortcode:
                seen.add(shortcode)
            yield post
    except Exception as exc:
        raise RuntimeError(_format_access_error(exc)) from exc

    get_reels = getattr(profile, "get_reels", None)
    if callable(get_reels):
        try:
            reels = get_reels()
        except Exception:
            reels = []
        try:
            for reel in reels:
                shortcode = getattr(reel, "shortcode", "") or ""
                if shortcode and shortcode in seen:
                    continue
                if shortcode:
                    seen.add(shortcode)
                yield reel
        except Exception as exc:
            raise RuntimeError(_format_access_error(exc)) from exc


def _download_image(loader, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = None
    if loader is not None:
        try:
            response = loader.context.get_raw(url)
            response.raise_for_status()
            with dest.open("wb") as fh:
                shutil.copyfileobj(response.raw, fh)
            return
        except Exception:
            pass
        finally:
            if response is not None:
                response.close()

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def _extract_image_urls(post) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    try:
        nodes = list(post.get_sidecar_nodes())
    except Exception:
        nodes = []

    for node in nodes:
        if getattr(node, "is_video", False):
            continue
        url = getattr(node, "display_url", None) or getattr(node, "url", None)
        if url and url not in seen:
            urls.append(url)
            seen.add(url)

    if urls:
        return urls

    if getattr(post, "is_video", False):
        return []

    url = getattr(post, "url", None) or getattr(post, "display_url", None)
    if url and url not in seen:
        urls.append(url)
        seen.add(url)

    return urls


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


def download_video_frames_for_username(
    username: str,
    output_dir: Path,
    max_videos: int = 5,
    fps: float = 1.0,
    max_frames_per_video: int = 0,
    sleep: float = 1.0,
    max_retries: int = 3,
    backoff: float = 60.0,
) -> list[Path]:
    username_key = (username or "").strip().lstrip("@").lower()
    if not username_key:
        raise ValueError("Username is empty.")

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg to extract frames.")

    loader = _init_instaloader()
    profile = _fetch_profile(loader, username_key, max_retries, backoff)
    if profile is None:
        raise RuntimeError("Failed to fetch profile. Check username or login.")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    video_count = 0

    for post in _iter_profile_posts(profile):
        if not getattr(post, "is_video", False):
            continue

        shortcode = getattr(post, "shortcode", "") or ""
        if not shortcode:
            continue

        video_url = getattr(post, "video_url", None)
        if not video_url:
            continue

        video_dir = output_dir / shortcode
        video_path = video_dir / "video.mp4"

        try:
            _download_video(loader, video_url, video_path)
            _extract_frames(ffmpeg_path, video_path, video_dir, fps)
        except Exception as exc:
            print(f"[error] {shortcode}: {exc}")
            if video_path.exists():
                video_path.unlink()
            continue
        finally:
            if video_path.exists():
                video_path.unlink()

        frames = sorted(video_dir.glob("frame_*.jpg"))
        if max_frames_per_video > 0:
            keep = frames[:max_frames_per_video]
            for extra in frames[max_frames_per_video:]:
                extra.unlink()
            frames = keep

        frame_paths.extend(frames)
        video_count += 1

        if max_videos > 0 and video_count >= max_videos:
            break

        if sleep > 0:
            time.sleep(sleep)

    if not frame_paths:
        raise RuntimeError("No video frames extracted for this username.")

    return frame_paths


def download_image_posts_for_username(
    username: str,
    output_dir: Path,
    max_posts: int = 5,
    max_images_per_post: int = 2,
    sleep: float = 1.0,
    max_retries: int = 3,
    backoff: float = 60.0,
) -> list[Path]:
    username_key = (username or "").strip().lstrip("@").lower()
    if not username_key:
        raise ValueError("Username is empty.")

    loader = _init_instaloader()
    profile = _fetch_profile(loader, username_key, max_retries, backoff)
    if profile is None:
        raise RuntimeError("Failed to fetch profile. Check username or login.")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    post_count = 0

    for post in _iter_profile_posts(profile):
        if getattr(post, "is_video", False):
            continue

        image_urls = _extract_image_urls(post)
        if not image_urls:
            continue

        shortcode = getattr(post, "shortcode", "") or f"post_{post_count:03d}"
        post_dir = output_dir / shortcode

        saved = 0
        for idx, url in enumerate(image_urls, start=1):
            if max_images_per_post > 0 and saved >= max_images_per_post:
                break
            dest = post_dir / f"image_{idx:03d}.jpg"
            try:
                _download_image(loader, url, dest)
            except Exception as exc:
                print(f"[error] {shortcode}: {exc}")
                if dest.exists():
                    dest.unlink()
                continue
            image_paths.append(dest)
            saved += 1

        if saved > 0:
            post_count += 1

        if max_posts > 0 and post_count >= max_posts:
            break

        if sleep > 0:
            time.sleep(sleep)

    if not image_paths:
        raise RuntimeError("No image posts found for this username.")

    return image_paths


def download_cnn_inputs_for_username(
    username: str,
    output_dir: Path,
    max_videos: int = 5,
    fps: float = 1.0,
    max_frames_per_video: int = 2,
    sleep: float = 1.0,
    max_retries: int = 3,
    backoff: float = 60.0,
    allow_image_fallback: bool = True,
) -> tuple[list[Path], str]:
    try:
        frame_paths = download_video_frames_for_username(
            username,
            output_dir=output_dir,
            max_videos=max_videos,
            fps=fps,
            max_frames_per_video=max_frames_per_video,
            sleep=sleep,
            max_retries=max_retries,
            backoff=backoff,
        )
        return frame_paths, "video"
    except RuntimeError as exc:
        if not allow_image_fallback:
            raise
        msg = str(exc).lower()
        if "no video frames extracted" not in msg and "ffmpeg not found" not in msg:
            raise

    image_paths = download_image_posts_for_username(
        username,
        output_dir=output_dir,
        max_posts=max_videos,
        max_images_per_post=max_frames_per_video,
        sleep=sleep,
        max_retries=max_retries,
        backoff=backoff,
    )
    return image_paths, "image"
