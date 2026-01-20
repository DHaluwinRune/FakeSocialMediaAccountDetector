#!/usr/bin/env python
import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup

LIVE_FEATURE_COLS = [
    "IsAccountPrivate",
    "IsVerified",
    "HasProfilePicture",
    "NumberFollowing",
    "NumberFollowers",
    "NumberLikes",
    "HasInstagramLink",
    "HasYoutubeLink",
    "HasAccountDescription",
    "HasLinkInDescription",
    "PostsExist",
    "NumberOfPosts",
    "FollowingToFollowerRatio",
    "LikesToFollowerRatio",
]

LIVE_VIDEO_FEATURE_COLS = [
    "AverageNumberOfHashtags",
    "AverageNumberOfComments",
    "AverageNumberOfCharacters",
    "AverageNumberOfForwardings",
    "AverageNumberOfLikes",
    "AverageNumberOfLinkedProfiles",
    "AverageNumberOfUsedFilters",
    "AverageNumberOfViews",
    "LikesToViewRatio",
    "CommentsToViewRatio",
]

URL_RE = re.compile(r"https?://", re.IGNORECASE)
HASHTAG_RE = re.compile(r"#([\\w_.-]+)")
MENTION_RE = re.compile(r"@([\\w_.-]+)")
DEFAULT_TIMEOUT = 15

def parse_username(raw):
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("@"):
        return raw[1:]
    match = re.search(r"tiktok\.com/@([^/?#]+)", raw, re.IGNORECASE)
    if match:
        return match.group(1)
    return raw


def safe_ratio(numer, denom):
    try:
        numer = float(numer)
        denom = float(denom)
    except (TypeError, ValueError):
        return None
    if denom == 0:
        return None
    return numer / denom


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_bool_int(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    return int(bool(value))


def load_model_meta(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_imputer_compat(estimator):
    try:
        from sklearn.impute import SimpleImputer
    except Exception:
        return

    def patch(item):
        if isinstance(item, SimpleImputer) and not hasattr(item, "_fill_dtype"):
            if hasattr(item, "_fit_dtype"):
                item._fill_dtype = item._fit_dtype
            elif hasattr(item, "statistics_"):
                try:
                    item._fill_dtype = item.statistics_.dtype
                except Exception:
                    item._fill_dtype = None
        elif hasattr(item, "steps"):
            for _, step in item.steps:
                patch(step)
        elif hasattr(item, "named_steps"):
            for step in item.named_steps.values():
                patch(step)
        elif hasattr(item, "estimators_"):
            for step in item.estimators_:
                if step is not None:
                    patch(step)
        elif hasattr(item, "estimator"):
            patch(item.estimator)

    patch(estimator)


def extract_script_json(html, script_id):
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", id=script_id)
    if script and script.string:
        return script.string
    match = re.search(rf'id="{script_id}"[^>]*>(.*?)</script>', html, re.DOTALL)
    if match:
        return match.group(1)
    return None


def fetch_profile_state(username, timeout):
    url = f"https://www.tiktok.com/@{username}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.tiktok.com/",
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(f"TikTok request failed with status {response.status_code}")

    html = response.text
    if "verify you are human" in html.lower():
        raise RuntimeError("TikTok blocked the request (bot check). Try again later.")

    for script_id in ["SIGI_STATE", "__UNIVERSAL_DATA_FOR_REHYDRATION__"]:
        payload = extract_script_json(html, script_id)
        if payload:
            try:
                return script_id, json.loads(payload)
            except json.JSONDecodeError:
                continue
    raise RuntimeError("Could not extract profile data from TikTok page.")


def extract_profile_from_state(username, state_id, state_data):
    username = username.lower()
    if state_id == "SIGI_STATE":
        user_module = state_data.get("UserModule", {})
        users = user_module.get("users", {})
        stats = user_module.get("stats", {})
        for user_id, user_info in users.items():
            if str(user_info.get("uniqueId", "")).lower() == username:
                return user_info, stats.get(user_id, {})
        return None, None

    if state_id == "__UNIVERSAL_DATA_FOR_REHYDRATION__":
        scope = state_data.get("__DEFAULT_SCOPE__", {})
        user_detail = scope.get("webapp.user-detail") or scope.get("webapp.user-detail-no-flow")
        if not isinstance(user_detail, dict):
            return None, None
        user_info = user_detail.get("userInfo", {})
        user_data = user_info.get("user", {})
        stats = user_info.get("stats", {})
        return user_data, stats

    return None, None


def build_live_feature_row(user_data, stats_data):
    if not user_data:
        raise RuntimeError("TikTok user data incomplete.")
    stats_data = stats_data or {}

    avatar = user_data.get("avatarThumb") or user_data.get("avatarLarger") or user_data.get("avatarMedium")
    signature = user_data.get("signature") or ""
    bio_link = user_data.get("bioLink")
    if isinstance(bio_link, dict):
        bio_link = bio_link.get("link")

    following = to_int(stats_data.get("followingCount"))
    followers = to_int(stats_data.get("followerCount"))
    likes = to_int(stats_data.get("heartCount"))
    posts = to_int(stats_data.get("videoCount"))

    row = {
        "IsAccountPrivate": to_bool_int(user_data.get("privateAccount")),
        "IsVerified": to_bool_int(user_data.get("verified")),
        "HasProfilePicture": to_bool_int(bool(avatar)),
        "NumberFollowing": following,
        "NumberFollowers": followers,
        "NumberLikes": likes,
        "HasInstagramLink": to_bool_int(
            user_data.get("ins_id")
            or user_data.get("insId")
            or user_data.get("instagramId")
            or user_data.get("instagram")
        ),
        "HasYoutubeLink": to_bool_int(
            user_data.get("youtubeChannelId")
            or user_data.get("youtube_channel_id")
            or user_data.get("youtube")
        ),
        "HasAccountDescription": to_bool_int(bool(signature.strip())),
        "HasLinkInDescription": to_bool_int(bool(bio_link) or bool(URL_RE.search(signature))),
        "PostsExist": to_bool_int(bool(posts and posts > 0)),
        "NumberOfPosts": posts,
        "FollowingToFollowerRatio": safe_ratio(following, followers),
        "LikesToFollowerRatio": safe_ratio(likes, followers),
    }
    return row


def extract_items_from_state(state_id, state_data):
    items = []
    item_module = state_data.get("ItemModule") or {}

    if state_id == "SIGI_STATE":
        item_ids = []
        user_page = state_data.get("UserPage")
        if isinstance(user_page, dict):
            item_ids = user_page.get("items") or user_page.get("itemIds") or []

        if not item_ids:
            item_list = state_data.get("ItemList") or {}
            if isinstance(item_list, dict):
                user_post = item_list.get("user-post") or item_list.get("user-posts") or {}
                if isinstance(user_post, dict):
                    item_ids = user_post.get("list") or user_post.get("items") or []

        if item_ids and item_module:
            for item_id in item_ids:
                item = item_module.get(str(item_id))
                if item:
                    items.append(item)
        elif item_module:
            items = list(item_module.values())

    if state_id == "__UNIVERSAL_DATA_FOR_REHYDRATION__":
        scope = state_data.get("__DEFAULT_SCOPE__", {})
        user_detail = scope.get("webapp.user-detail") or scope.get("webapp.user-detail-no-flow")
        if isinstance(user_detail, dict):
            item_list = user_detail.get("itemList") or user_detail.get("items")
            if isinstance(item_list, list):
                items = item_list
            elif isinstance(item_list, dict):
                items = item_list.get("list", []) or item_list.get("items", [])
        if not items and item_module:
            items = list(item_module.values())

    return items


def fetch_video_items_from_api(sec_uid, count, timeout):
    url = "https://www.tiktok.com/api/post/item_list/"
    params = {
        "aid": "1988",
        "count": count,
        "cursor": 0,
        "secUid": sec_uid,
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, params=params, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"Video API failed with status {response.status_code}")
    data = response.json()
    if data.get("statusCode") not in (0, "0", None):
        raise RuntimeError(f"Video API returned statusCode={data.get('statusCode')}")
    items = data.get("itemList") or []
    return items


def build_text_extra_from_desc(desc):
    if not desc:
        return []
    hashtags = {tag for tag in HASHTAG_RE.findall(desc)}
    mentions = {handle for handle in MENTION_RE.findall(desc)}
    text_extra = [{"hashtagName": tag} for tag in hashtags]
    text_extra.extend({"userUniqueId": handle} for handle in mentions)
    return text_extra


def fetch_video_items_with_yt_dlp(username, limit, timeout):
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError
    except Exception as exc:
        raise RuntimeError("yt-dlp not available; install it to enable video fallback.") from exc

    url = f"https://www.tiktok.com/@{username}"
    errors = []
    class YdlLogger:
        def debug(self, msg):
            return
        def warning(self, msg):
            errors.append(msg)
        def error(self, msg):
            errors.append(msg)

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "logger": YdlLogger(),
        "extract_flat": True,
        "skip_download": True,
        "playlistend": limit,
        "socket_timeout": timeout,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as exc:
        message = str(exc)
        if "private" in message.lower():
            raise RuntimeError(
                "Account is private; post data cannot be accessed."
            ) from exc
        raise RuntimeError(f"yt-dlp failed to fetch videos: {message}") from exc

    entries = info.get("entries") or []
    items = []
    for entry in entries[:limit]:
        desc = entry.get("description") or entry.get("title") or ""
        stats = {
            "diggCount": entry.get("like_count"),
            "commentCount": entry.get("comment_count"),
            "shareCount": entry.get("repost_count"),
            "playCount": entry.get("view_count"),
        }
        thumb_urls = []
        thumb = entry.get("thumbnail")
        if isinstance(thumb, dict):
            thumb = thumb.get("url") or thumb.get("source")
        if isinstance(thumb, str):
            thumb_urls.append(thumb)
        for thumb in entry.get("thumbnails") or []:
            if isinstance(thumb, dict):
                thumb = thumb.get("url") or thumb.get("source")
            if isinstance(thumb, str) and thumb not in thumb_urls:
                thumb_urls.append(thumb)
        video = {}
        if thumb_urls:
            video = {
                "cover": {"urlList": thumb_urls},
                "dynamicCover": {"urlList": thumb_urls},
                "originCover": {"urlList": thumb_urls},
            }
        items.append(
            {
                "desc": desc,
                "stats": stats,
                "textExtra": build_text_extra_from_desc(desc),
                "thumbnail": thumb_urls[0] if thumb_urls else None,
                "thumbnails": [{"url": url} for url in thumb_urls],
                "video": video,
            }
        )
    return items


def compute_video_aggregates(items, limit=20):
    normalized = []
    for item in items:
        if not item:
            continue
        normalized.append(item.get("itemStruct", item))

    normalized = normalized[:limit]
    if not normalized:
        raise RuntimeError("No public videos found for this account.")

    likes = []
    comments = []
    shares = []
    views = []
    hashtags = []
    mentions = []
    desc_lengths = []

    for item in normalized:
        desc = item.get("desc") or ""
        stats = item.get("stats") or item.get("statsV2") or {}
        likes.append(to_int(stats.get("diggCount")))
        comments.append(to_int(stats.get("commentCount")))
        shares.append(to_int(stats.get("shareCount")))
        views.append(to_int(stats.get("playCount")))
        desc_lengths.append(len(desc))

        text_extra = item.get("textExtra") or []
        hash_count = 0
        mention_count = 0
        for entry in text_extra:
            if entry.get("hashtagName"):
                hash_count += 1
            elif entry.get("userId") or entry.get("userUniqueId"):
                mention_count += 1
        hashtags.append(hash_count)
        mentions.append(mention_count)

    def avg(values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        return sum(values) / len(values)

    total_likes = sum(v for v in likes if v is not None)
    total_comments = sum(v for v in comments if v is not None)
    total_views = sum(v for v in views if v is not None)

    return {
        "AverageNumberOfHashtags": avg(hashtags),
        "AverageNumberOfComments": avg(comments),
        "AverageNumberOfCharacters": avg(desc_lengths),
        "AverageNumberOfForwardings": avg(shares),
        "AverageNumberOfLikes": avg(likes),
        "AverageNumberOfLinkedProfiles": avg(mentions),
        "AverageNumberOfUsedFilters": None,
        "AverageNumberOfViews": avg(views),
        "LikesToViewRatio": safe_ratio(total_likes, total_views),
        "CommentsToViewRatio": safe_ratio(total_comments, total_views),
    }


def fetch_profile_videos(username, user_data, state_id, state_data, timeout, limit=20):
    sec_uid = user_data.get("secUid") or user_data.get("sec_uid")
    items = []
    if sec_uid:
        try:
            items = fetch_video_items_from_api(sec_uid, limit, timeout)
        except Exception:
            items = []

    if not items:
        items = extract_items_from_state(state_id, state_data)

    fallback_error = None
    if not items:
        try:
            items = fetch_video_items_with_yt_dlp(username, limit, timeout)
        except Exception as exc:
            fallback_error = exc
            items = []

    items = items[:limit]
    if not items:
        if fallback_error:
            message = str(fallback_error)
            if "account is private" in message.lower():
                raise RuntimeError("Account is private; post data cannot be accessed.")
            raise RuntimeError(
                f"No public videos found for this account. {fallback_error}"
            )
        raise RuntimeError("No public videos found for this account.")
    return items
def load_feature_columns(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def predict_live(username, model_dir="models", timeout=DEFAULT_TIMEOUT):
    model_dir = Path(model_dir)
    model_path = model_dir / "model.joblib"
    columns_path = model_dir / "feature_columns.json"
    meta_path = model_dir / "model_meta.json"

    if not model_path.exists() or not columns_path.exists():
        raise RuntimeError("Model artifacts not found. Train the model first.")

    model = joblib.load(model_path)
    ensure_imputer_compat(model)
    feature_columns = load_feature_columns(columns_path)
    meta = load_model_meta(meta_path)
    feature_set = meta.get("feature_set")

    if feature_set == "full":
        raise RuntimeError(
            "Model was trained with full features. "
            "Retrain using: python scripts/train_model.py --feature-set live-video"
        )
    if feature_set is None and not set(feature_columns).issubset(
        set(LIVE_FEATURE_COLS + LIVE_VIDEO_FEATURE_COLS)
    ):
        raise RuntimeError(
            "Model features are not compatible with live mode. "
            "Retrain using: python scripts/train_model.py --feature-set live-video"
        )

    state_id, state_data = fetch_profile_state(username, timeout)
    user_data, stats_data = extract_profile_from_state(username, state_id, state_data)
    row = build_live_feature_row(user_data, stats_data)

    needs_video = feature_set == "live-video" or any(
        col in LIVE_VIDEO_FEATURE_COLS for col in feature_columns
    )
    video_count = None
    video_error = None
    if needs_video:
        try:
            items = fetch_profile_videos(username, user_data, state_id, state_data, timeout, limit=20)
            video_count = len(items)
            row.update(compute_video_aggregates(items, limit=20))
        except RuntimeError as exc:
            video_error = str(exc)
            video_count = 0
            for col in LIVE_VIDEO_FEATURE_COLS:
                row.setdefault(col, None)

    X = pd.DataFrame([row])
    for col in feature_columns:
        if col not in X.columns:
            X[col] = pd.NA
    X = X[feature_columns]

    prediction = int(model.predict(X)[0])
    label = "FAKE/BOT" if prediction == 1 else "REAL"

    probability = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        probability = float(proba[1]) if len(proba) > 1 else None

    return {
        "label": label,
        "probability": probability,
        "features": row,
        "video_count": video_count,
        "video_error": video_error,
        "feature_set": meta.get("feature_set"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Predict if a TikTok account is fake/bot.")
    parser.add_argument("--username", help="TikTok username or profile link")
    parser.add_argument("--model-dir", default="models", help="Directory with model artifacts")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    raw = args.username
    if raw is None:
        raw = input("TikTok username or link: ").strip()
    if raw.lower() in {"quit", "exit"}:
        print("Canceled.")
        return 0
    username = parse_username(raw)
    if not username:
        print("No username provided.", file=sys.stderr)
        return 1

    try:
        result = predict_live(username, model_dir=model_dir, timeout=DEFAULT_TIMEOUT)
    except RuntimeError as exc:
        print(f"Live lookup failed: {exc}", file=sys.stderr)
        return 1

    print(f"Username: {username}")
    if result.get("video_count") is not None:
        print(f"Videos used: {result['video_count']}")
    if result.get("video_error"):
        print(f"Video warning: {result['video_error']}", file=sys.stderr)
    probability = result.get("probability")
    if probability is not None:
        print(f"Prediction: {result['label']} (bot probability {probability:.2%})")
    else:
        print(f"Prediction: {result['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
