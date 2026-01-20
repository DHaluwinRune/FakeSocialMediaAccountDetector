import argparse

from .features import build_feature_row
from .instagram import ProfileFetchError, features_from_profile_input
from .predict import load_model, predict_from_features

def ask_int(label: str) -> int:
    return int(input(f"{label}: ").strip())

def ask_bool01(label: str) -> int:
    v = input(f"{label} (0/1): ").strip()
    if v not in ("0", "1"):
        raise ValueError("Geef 0 of 1 in.")
    return int(v)

def _prompt_manual_features() -> dict:
    print("=== Instagram Fake Detector (manual input) ===")
    username = input("username: ").strip()
    fullname = input("full name: ").strip()
    bio = input("bio/description: ").strip()

    has_url = ask_bool01("external URL aanwezig")
    is_private = ask_bool01("private")
    has_pfp = ask_bool01("profile pic aanwezig")

    posts = ask_int("#posts")
    followers = ask_int("#followers")
    follows = ask_int("#follows")

    return build_feature_row(
        username=username,
        fullname=fullname,
        bio=bio,
        has_url=has_url,
        is_private=is_private,
        has_pfp=has_pfp,
        posts=posts,
        followers=followers,
        follows=follows,
    )


def main():
    parser = argparse.ArgumentParser(description="Instagram fake detector")
    parser.add_argument(
        "profile",
        nargs="?",
        help="Instagram profile URL or username (leave blank for manual input)",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Always prompt for manual input instead of fetching from Instagram",
    )
    args = parser.parse_args()

    features = None
    if not args.manual:
        profile_input = args.profile
        if profile_input is None:
            profile_input = input("Instagram profile URL or username (blank for manual): ").strip()
        if profile_input:
            try:
                features = features_from_profile_input(profile_input)
            except ProfileFetchError as exc:
                print(f"Could not fetch profile: {exc}")
                return

    if features is None:
        features = _prompt_manual_features()

    model = load_model()
    result = predict_from_features(model, features)

    print("\n--- Result ---")
    print("Prediction:", result["prediction"])
    print("Fake probability:", round(result["fake_probability"], 4))
    print("Threshold:", result["threshold"])

if __name__ == "__main__":
    main()
