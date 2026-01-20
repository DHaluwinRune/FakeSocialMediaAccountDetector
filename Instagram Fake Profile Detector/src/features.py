def num_ratio(s: str) -> float:
    s = (s or "").strip()
    if len(s) == 0:
        return 0.0
    digits = sum(ch.isdigit() for ch in s)
    return digits / len(s)

def fullname_words(fullname: str) -> int:
    fullname = (fullname or "").strip()
    if not fullname:
        return 0
    return len([w for w in fullname.split() if w])

def name_equals_username(fullname: str, username: str) -> int:
    return int((fullname or "").strip().lower() == (username or "").strip().lower())

def build_feature_row(
    username: str,
    fullname: str,
    bio: str,
    has_url: int,
    is_private: int,
    has_pfp: int,
    posts: int,
    followers: int,
    follows: int,
) -> dict:
    return {
        "profile pic": int(has_pfp),
        "nums/length username": float(num_ratio(username)),
        "fullname words": int(fullname_words(fullname)),
        "nums/length fullname": float(num_ratio(fullname)),
        "name==username": int(name_equals_username(fullname, username)),
        "description length": int(len((bio or "").strip())),
        "external URL": int(has_url),
        "private": int(is_private),
        "#posts": int(posts),
        "#followers": int(followers),
        "#follows": int(follows),
    }
