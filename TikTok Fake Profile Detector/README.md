# TikTok bot detector

Dit project werkt volledig live: het haalt publieke profieldata (en optioneel
de laatste 20 publieke videos) op en maakt daarmee een voorspelling.

## Installatie

```bash
pip install -r requirements.txt
```

## Model trainen

Live model (publieke profieldata):

```bash
python scripts/train_model.py --feature-set live
```

Live + video features (laatste 20 videos):

```bash
python scripts/train_model.py --feature-set live-video
```

Image model (profielafbeeldingen):

```bash
python scripts/train_image_model.py
```

Volledig model op accounts + videos:

```bash
python scripts/train_model.py
```

Optioneel zonder videofeatures:

```bash
python scripts/train_model.py --no-videos
```

Artifacts komen in `models/`:
- `model.joblib`
- `feature_columns.json`
- `metrics.json`
- `model_meta.json`
- `features_cache.csv`

Image artifacts:
- `image_model.pth`
- `image_model_meta.json`
- `image_model_metrics.json`

## Predictie (live)

```bash
python scripts/predict_account.py
```

Voorbeeld input:
- `@example_user_1`
- `https://www.tiktok.com/@example_user_1`

## Opmerkingen

- TikTok kan requests blokkeren; probeer opnieuw als live mode faalt.
- Live mode gebruikt alleen publiek zichtbare data. Voor video-features
  worden de laatste 20 publieke videos gebruikt als die beschikbaar zijn.
- Als er geen publieke videos zijn, valt live voorspellen terug op profieldata.

## Live features (uitleg)

Bij live mode worden enkel features gebruikt die publiek zichtbaar zijn.

Profiel-features:
- `NumberFollowers`, `NumberFollowing`, `NumberLikes`, `NumberOfPosts`
- `IsVerified`, `IsAccountPrivate`, `HasProfilePicture`
- `HasInstagramLink`, `HasYoutubeLink`
- `HasAccountDescription`, `HasLinkInDescription`
- `FollowingToFollowerRatio`, `LikesToFollowerRatio`

Video-features (gemiddelden over laatste 20 videos):
- `AverageNumberOfLikes`, `AverageNumberOfComments`, `AverageNumberOfForwardings`
- `AverageNumberOfViews`, `AverageNumberOfHashtags`, `AverageNumberOfLinkedProfiles`
- `AverageNumberOfCharacters`
- `LikesToViewRatio`, `CommentsToViewRatio`

Niet alle dataset-kolommen zijn live beschikbaar. Als het model met
`--feature-set live-video` is getraind, gebruikt het enkel bovenstaande
features zodat live voorspellingen consistent blijven.

## UI (Streamlit)

```bash
streamlit run app.py
```

De UI bevat nu een keuze tussen:
- Account (live profieldata)
- Video frame (CNN)
- Video (CNN frames, laatste posts)
- Fusion (account + video)
