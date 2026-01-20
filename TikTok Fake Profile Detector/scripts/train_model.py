#!/usr/bin/env python
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BOOL_MAP = {True: 1, False: 0, "True": 1, "False": 0}

ACCOUNT_BOOL_COLS = [
    "IsAccountPrivate",
    "IsVerified",
    "HasProfilePicture",
    "HasInstagramLink",
    "HasYoutubeLink",
    "HasAccountDescription",
    "HasLinkInDescription",
    "PostsExist",
    "AreLikedVideosPrivate",
]

LIVE_BOOL_COLS = [
    "IsAccountPrivate",
    "IsVerified",
    "HasProfilePicture",
    "HasInstagramLink",
    "HasYoutubeLink",
    "HasAccountDescription",
    "HasLinkInDescription",
    "PostsExist",
]

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

VIDEO_BOOL_COLS = [
    "IsVideoCreatorAccountVerified",
    "HasVideoDescription",
    "HasUsedFilters",
    "AreAccountVideos",
    "AreLikedVideos",
]

VIDEO_NUMERIC_COLS = [
    "NumberOfLikes",
    "NumberOfComments",
    "NumberOfForwardings",
    "NumberOfLinkedProfiles",
    "NumberOfHashtags",
    "VideoDescriptionLength",
    "LikesToCommentRatio",
    "NumberOfUsedFilters",
    "NumberOfViews",
]


def normalize_bool_columns(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(BOOL_MAP)
    return df


def coerce_numeric(df, exclude):
    for col in df.columns:
        if col in exclude:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_ratio(numer, denom):
    denom = denom.replace(0, pd.NA)
    return numer / denom


def build_video_features(videos_df):
    df = videos_df.copy()
    df = df[df["AccountId"].notna()].copy()
    df["AccountId"] = df["AccountId"].astype(str)

    bool_cols = [col for col in VIDEO_BOOL_COLS if col in df.columns]
    numeric_cols = [col for col in VIDEO_NUMERIC_COLS if col in df.columns]

    df = normalize_bool_columns(df, bool_cols)
    if "VideoUploadDate" in df.columns:
        df["VideoUploadDate"] = pd.to_datetime(df["VideoUploadDate"], errors="coerce")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in bool_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    group = df.groupby("AccountId")
    counts = group.size().rename("video_count").to_frame()
    numeric_means = group[numeric_cols].mean(numeric_only=True) if numeric_cols else pd.DataFrame()
    bool_means = group[bool_cols].mean(numeric_only=True) if bool_cols else pd.DataFrame()

    date_features = pd.DataFrame(index=group.size().index)
    if "VideoUploadDate" in df.columns:
        date_agg = group["VideoUploadDate"].agg(["min", "max"])
        date_features["vid_days_span"] = (date_agg["max"] - date_agg["min"]).dt.days
        max_date = df["VideoUploadDate"].max()
        if pd.isna(max_date):
            date_features["vid_days_since_last"] = pd.NA
        else:
            date_features["vid_days_since_last"] = (max_date - date_agg["max"]).dt.days

    features = pd.concat([counts, numeric_means, bool_means, date_features], axis=1).reset_index()
    features = features.rename(columns={"AccountId": "Id"})
    features = features.rename(columns={c: f"vid_{c}" for c in features.columns if c != "Id"})
    return features


def build_live_features_from_accounts(accounts_df):
    df = accounts_df.copy()
    df["Id"] = df["Id"].astype(str)

    bool_cols = [col for col in LIVE_BOOL_COLS if col in df.columns]
    df = normalize_bool_columns(df, bool_cols)

    numeric_cols = ["NumberFollowing", "NumberFollowers", "NumberLikes", "NumberOfPosts"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "NumberFollowing" in df.columns and "NumberFollowers" in df.columns:
        df["FollowingToFollowerRatio"] = safe_ratio(
            df["NumberFollowing"], df["NumberFollowers"]
        )
    if "NumberLikes" in df.columns and "NumberFollowers" in df.columns:
        df["LikesToFollowerRatio"] = safe_ratio(df["NumberLikes"], df["NumberFollowers"])

    for col in LIVE_FEATURE_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    features = df[["Id"] + LIVE_FEATURE_COLS].copy()
    features = coerce_numeric(features, exclude={"Id"})
    return features


def build_live_video_features_from_accounts(accounts_df):
    df = accounts_df.copy()
    df["Id"] = df["Id"].astype(str)

    bool_cols = [col for col in LIVE_BOOL_COLS if col in df.columns]
    df = normalize_bool_columns(df, bool_cols)

    numeric_cols = [
        "NumberFollowing",
        "NumberFollowers",
        "NumberLikes",
        "NumberOfPosts",
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
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "NumberFollowing" in df.columns and "NumberFollowers" in df.columns:
        df["FollowingToFollowerRatio"] = safe_ratio(
            df["NumberFollowing"], df["NumberFollowers"]
        )
    if "NumberLikes" in df.columns and "NumberFollowers" in df.columns:
        df["LikesToFollowerRatio"] = safe_ratio(df["NumberLikes"], df["NumberFollowers"])

    selected_cols = LIVE_FEATURE_COLS + LIVE_VIDEO_FEATURE_COLS
    for col in selected_cols:
        if col not in df.columns:
            df[col] = pd.NA

    features = df[["Id"] + selected_cols].copy()
    features = coerce_numeric(features, exclude={"Id"})
    return features


def build_account_features(accounts_df, video_features):
    df = accounts_df.copy()
    df["Id"] = df["Id"].astype(str)

    bool_cols = [col for col in ACCOUNT_BOOL_COLS if col in df.columns]
    df = normalize_bool_columns(df, bool_cols)

    drop_cols = [col for col in ["IsABot", "LinkInDescription"] if col in df.columns]
    features = df.drop(columns=drop_cols, errors="ignore")
    features = coerce_numeric(features, exclude={"Id"})

    if video_features is not None:
        features = features.merge(video_features, on="Id", how="left")

    return features


def build_training_frame(accounts_df, features_df):
    labels = accounts_df[["Id", "IsABot"]].copy()
    labels["Id"] = labels["Id"].astype(str)
    labels["IsABot"] = labels["IsABot"].map(BOOL_MAP)
    labels = labels.dropna(subset=["IsABot"])
    labels["IsABot"] = labels["IsABot"].astype(int)
    return features_df.merge(labels, on="Id", how="inner")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a TikTok fake/bot account detector.")
    parser.add_argument("--accounts", default="data/dataset/accounts.csv", help="Path to accounts.csv")
    parser.add_argument("--videos", default="data/dataset/videos.csv", help="Path to videos.csv")
    parser.add_argument(
        "--feature-set",
        choices=["full", "live", "live-video"],
        default="full",
        help="Feature set: full (accounts + videos), live (profile-only), or live-video",
    )
    parser.add_argument("--no-videos", action="store_true", help="Skip video feature aggregation")
    parser.add_argument("--output-dir", default="models", help="Directory for model artifacts")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of trees")
    return parser.parse_args()


def main():
    args = parse_args()
    accounts_path = Path(args.accounts)
    videos_path = Path(args.videos)
    output_dir = Path(args.output_dir)

    if not accounts_path.exists():
        print(f"Accounts file not found: {accounts_path}", file=sys.stderr)
        return 1

    accounts_df = pd.read_csv(accounts_path)
    feature_set = args.feature_set
    video_features = None
    if feature_set == "full":
        if not args.no_videos and videos_path.exists():
            videos_df = pd.read_csv(videos_path)
            video_features = build_video_features(videos_df)
        elif not args.no_videos:
            print(f"Warning: videos file not found, skipping: {videos_path}")
        features_df = build_account_features(accounts_df, video_features)
    elif feature_set == "live":
        features_df = build_live_features_from_accounts(accounts_df)
        args.no_videos = True
    else:
        features_df = build_live_video_features_from_accounts(accounts_df)
        args.no_videos = True
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cache_path = output_dir / "features_cache.csv"
    features_df.to_csv(feature_cache_path, index=False)

    train_df = build_training_frame(accounts_df, features_df)
    if train_df.empty:
        print("No labeled rows found in accounts.csv (IsABot missing).", file=sys.stderr)
        return 1

    feature_columns = [col for col in features_df.columns if col != "Id"]
    X = train_df[feature_columns]
    y = train_df["IsABot"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    random_state=args.random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "rows_total": int(len(accounts_df)),
        "rows_labeled": int(len(train_df)),
        "features_count": int(len(feature_columns)),
        "feature_set": feature_set,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    model_path = output_dir / "model.joblib"
    columns_path = output_dir / "feature_columns.json"
    metrics_path = output_dir / "metrics.json"
    meta_path = output_dir / "model_meta.json"
    meta = {
        "feature_set": feature_set,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_estimators": args.n_estimators,
        "has_video_features": feature_set in {"full", "live-video"} and not args.no_videos,
    }

    joblib.dump(model, model_path)
    write_json(columns_path, feature_columns)
    write_json(metrics_path, metrics)
    write_json(meta_path, meta)

    print(f"Model saved to: {model_path}")
    print(f"Feature columns saved to: {columns_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Model meta saved to: {meta_path}")
    print(f"Feature cache saved to: {feature_cache_path}")
    print(f"Test accuracy: {report['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
