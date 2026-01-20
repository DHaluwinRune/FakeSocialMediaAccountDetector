from __future__ import annotations

import os
from urllib.parse import urlparse

from .features import build_feature_row


class ProfileFetchError(RuntimeError):
    pass


def extract_username(profile_input: str) -> str:
    raw = (profile_input or "").strip()
    if not raw:
        raise ValueError("Empty profile input.")

    if raw.startswith("@"):
        raw = raw[1:]

    if "instagram.com" in raw:
        if "://" not in raw:
            raw = "https://" + raw
        parsed = urlparse(raw)
        path = (parsed.path or "").strip("/")
        if not path:
            raise ValueError("URL does not contain a username.")
        parts = [p for p in path.split("/") if p]
        username = parts[0]
        if username in {"p", "reel", "tv", "stories"}:
            raise ValueError("URL points to a post/story, not a profile.")
        return username

    return raw


def _get_int_attr(obj, *names: str) -> int:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return int(value)
    return 0


def _load_profile(username: str):
    try:
        import instaloader
    except ImportError as exc:
        raise ProfileFetchError(
            "Missing dependency: instaloader. Install with `pip install instaloader`."
        ) from exc

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
        try:
            loader.login(ig_user, ig_pass)
        except Exception as exc:  # pragma: no cover - error details vary by env
            raise ProfileFetchError("Instagram login failed. Check IG_USERNAME/IG_PASSWORD.") from exc

    try:
        return instaloader.Profile.from_username(loader.context, username)
    except Exception as exc:  # pragma: no cover - error details vary by env
        raise ProfileFetchError(
            "Failed to fetch profile. The account may be private or blocked without login."
        ) from exc


def features_from_profile_input(profile_input: str) -> dict:
    username = extract_username(profile_input)
    profile = _load_profile(username)

    has_pfp_default = getattr(profile, "is_profile_pic_default", None)
    if has_pfp_default is None:
        has_pfp = int(bool(getattr(profile, "profile_pic_url", None)))
    else:
        has_pfp = int(not has_pfp_default)

    return build_feature_row(
        username=getattr(profile, "username", username),
        fullname=getattr(profile, "full_name", "") or "",
        bio=getattr(profile, "biography", "") or "",
        has_url=int(bool(getattr(profile, "external_url", None))),
        is_private=int(bool(getattr(profile, "is_private", False))),
        has_pfp=has_pfp,
        posts=_get_int_attr(profile, "mediacount", "posts"),
        followers=_get_int_attr(profile, "followers", "follower_count"),
        follows=_get_int_attr(profile, "followees", "following_count"),
    )
