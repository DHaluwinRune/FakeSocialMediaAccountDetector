# FakeSocialMediaAccountDetector

Een gecombineerd project om fake social media accounts te detecteren op Instagram en TikTok. Het project bevat:
- een account model op profielmetadata (RandomForest)
- een CNN model op visuele frames (video of screenshot)
- een fusion score die account + visual combineert
- een Streamlit UI voor live voorspellingen

## Inhoud

- Overzicht en architectuur
- Data (met privacy noot)
- Modellen en features
- UI
- Installatie
- Gebruik (Streamlit en CLI)
- Training
- Omgevingsvariabelen
- Projectstructuur
- Beperkingen

## Overzicht en architectuur

Het project bestaat uit twee platform-specifieke pipelines:
- **Instagram**: account model op profielmetadata + CNN op video frames, met optionele fusion.
- **TikTok**: account model op live profieldata (en optioneel live video stats) + CNN op frames, met optionele fusion.

De root `app.py` laadt beide subprojecten en biedt een uniforme Streamlit UI met keuze tussen platformen.

## Data (en privacy noot)

**Belangrijk:** de echte datasets zijn verwijderd om privacyredenen. Er staat enkel een kleine test set in de repo zodat de structuur duidelijk blijft.

### Instagram data
- `Instagram Fake Profile Detector/Data/Instagram_fake_profile_dataset.csv`
  - 1 demo row + header (2 lijnen totaal)
  - Kolommen: profiel features + `fake` label.
- `Instagram Fake Profile Detector/Data/ig_post_metadata.csv`
  - 1 demo row + header (2 lijnen totaal)
  - Metadata per post voor het video/CNN traject.
- `Instagram Fake Profile Detector/Data/video_frames/`
  - voorbeeldframe(s) voor de CNN pipeline.
- `Instagram Fake Profile Detector/Data/seed_usernames.csv`
  - seed lijst voor scraping/test.

### TikTok data
- `TikTok Fake Profile Detector/data/dataset/accounts.csv`
  - 1 demo row + header (2 lijnen totaal)
  - Bevat account features + label `IsABot`.
- `TikTok Fake Profile Detector/data/dataset/videos.csv`
  - 1 demo row + header (2 lijnen totaal)
  - Video-level data voor aggregatie.
- `TikTok Fake Profile Detector/data/dataset/real/` en `fake/`
  - voorbeeld frames voor het CNN image model.

## Modellen en features

### Instagram

#### 1) Account model (RandomForest)
Doel: fake vs real op basis van profielmetadata.

**Features (vaste volgorde, zie `Instagram Fake Profile Detector/src/config.py`):**
- `profile pic`
- `nums/length username`
- `fullname words`
- `nums/length fullname`
- `name==username`
- `description length`
- `external URL`
- `private`
- `#posts`
- `#followers`
- `#follows`

**Training:**
```
cd "Instagram Fake Profile Detector"
python -m src.train
```

**Artifact:**
- `Instagram Fake Profile Detector/models/instagram_fake_detector.joblib`

#### 2) CNN model (ResNet18) op video frames / screenshots
Doel: fake vs real op visuele signalen in posts of screenshots.

**Input features:**
- ruwe pixels uit frames (geen handgemaakte features)
- beelden worden geschaald naar 224x224 en genormaliseerd

**Training:**
```
cd "Instagram Fake Profile Detector"
python -m src.train_cnn
```

**Artifact:**
- `Instagram Fake Profile Detector/models/instagram_video_cnn.pt`

#### 3) Fusion model
Combineert account + CNN met gewichten:
- **70% account**
- **30% CNN**

### TikTok

#### 1) Account model (RandomForest) - live profieldata
**Live feature set (`LIVE_FEATURE_COLS` in `scripts/predict_account.py`):**
- `IsAccountPrivate`
- `IsVerified`
- `HasProfilePicture`
- `NumberFollowing`
- `NumberFollowers`
- `NumberLikes`
- `HasInstagramLink`
- `HasYoutubeLink`
- `HasAccountDescription`
- `HasLinkInDescription`
- `PostsExist`
- `NumberOfPosts`
- `FollowingToFollowerRatio`
- `LikesToFollowerRatio`

**Training (live only):**
```
cd "TikTok Fake Profile Detector"
python scripts/train_model.py --feature-set live
```

#### 2) Account model (RandomForest) - live + video aggregates
**Extra live video features (`LIVE_VIDEO_FEATURE_COLS`):**
- `AverageNumberOfHashtags`
- `AverageNumberOfComments`
- `AverageNumberOfCharacters`
- `AverageNumberOfForwardings`
- `AverageNumberOfLikes`
- `AverageNumberOfLinkedProfiles`
- `AverageNumberOfUsedFilters`
- `AverageNumberOfViews`
- `LikesToViewRatio`
- `CommentsToViewRatio`

**Training (live + video):**
```
cd "TikTok Fake Profile Detector"
python scripts/train_model.py --feature-set live-video
```

#### 3) Account model (RandomForest) - full dataset (offline)
Wordt getraind op `accounts.csv` en optioneel geaggregeerde video data uit `videos.csv`.
Dit model is **niet** geschikt voor live voorspellingen in `scripts/predict_account.py`.

**Account features (kolommen uit `accounts.csv`, zonder `Id`, `IsABot`, `LinkInDescription`):**
- `IsAccountPrivate`
- `IsVerified`
- `HasProfilePicture`
- `NumberFollowing`
- `NumberFollowers`
- `NumberLikes`
- `HasInstagramLink`
- `HasYoutubeLink`
- `HasAccountDescription`
- `HasLinkInDescription`
- `PostsExist`
- `NumberOfPosts`
- `AreLikedVideosPrivate`
- `NumberOfLikedVideos`
- `FollowingToFollowerRatio`
- `LikesToFollowerRatio`
- `AverageNumberOfHashtags`
- `AverageNumberOfComments`
- `AverageNumberOfCharacters`
- `AverageNumberOfForwardings`
- `AverageNumberOfLikes`
- `AverageNumberOfLinkedProfiles`
- `AverageNumberOfUsedFilters`
- `LikedAverageNumberOfHashtags`
- `LikedAverageNumberOfComments`
- `LikedAverageNumberOfCharacters`
- `LikedAverageNumberOfForwardings`
- `LikedAverageNumberOfLikes`
- `LikedAverageNumberOfLinkedProfiles`
- `LikedAverageNumberOfUsedFilters`
- `AverageNumberOfViews`
- `LikedAverageNumberOfViews`
- `LikesToViewRatio`
- `CommentsToViewRatio`

**Extra video aggregates uit `videos.csv` (prefix `vid_`):**
- `vid_video_count`
- `vid_NumberOfLikes`
- `vid_NumberOfComments`
- `vid_NumberOfForwardings`
- `vid_NumberOfLinkedProfiles`
- `vid_NumberOfHashtags`
- `vid_VideoDescriptionLength`
- `vid_LikesToCommentRatio`
- `vid_NumberOfUsedFilters`
- `vid_NumberOfViews`
- `vid_IsVideoCreatorAccountVerified`
- `vid_HasVideoDescription`
- `vid_HasUsedFilters`
- `vid_AreAccountVideos`
- `vid_AreLikedVideos`
- `vid_days_span`
- `vid_days_since_last`

**Training (full dataset):**
```
cd "TikTok Fake Profile Detector"
python scripts/train_model.py
```

#### 4) Image/CNN model (SimpleCNN)
Classificeert frames (real vs fake) op basis van pixel data.

**Input features:**
- ruwe pixels uit `data/dataset/real/` en `data/dataset/fake/`

**Training:**
```
cd "TikTok Fake Profile Detector"
python scripts/train_image_model.py
```

#### 5) Fusion model
Combineert account + CNN met gewichten:
- **70% account**
- **30% CNN**

## Artifacts en logs

### Instagram
- `Instagram Fake Profile Detector/models/instagram_fake_detector.joblib`
- `Instagram Fake Profile Detector/models/instagram_fake_detector.metrics.txt`
- `Instagram Fake Profile Detector/models/instagram_video_cnn.pt`
- `Instagram Fake Profile Detector/models/instagram_video_cnn.metrics.txt`
- `Instagram Fake Profile Detector/models/modality_comparison_summary.txt`
- `Instagram Fake Profile Detector/models/training_cost_analysis.txt`

### TikTok
- `TikTok Fake Profile Detector/models/model.joblib`
- `TikTok Fake Profile Detector/models/feature_columns.json`
- `TikTok Fake Profile Detector/models/metrics.json`
- `TikTok Fake Profile Detector/models/model_meta.json`
- `TikTok Fake Profile Detector/models/features_cache.csv`
- `TikTok Fake Profile Detector/models/image_model.pth`
- `TikTok Fake Profile Detector/models/image_model_meta.json`
- `TikTok Fake Profile Detector/models/image_model_metrics.json`

## UI

### Centrale app (root)
```
streamlit run app.py
```

Functies:
- platform keuze: Instagram of TikTok
- modes: account, screenshot/image, video frames, fusion
- uitleg en feature importance waar mogelijk

### Instagram app (apart)
```
cd "Instagram Fake Profile Detector"
streamlit run streamlit_app.py
```

### TikTok app (apart)
```
cd "TikTok Fake Profile Detector"
streamlit run app.py
```

## Installatie

### 1) Virtuele omgeving
```
python -m venv .venv
```

Windows:
```
.venv\Scripts\activate
```

macOS/Linux:
```
source .venv/bin/activate
```

### 2) Dependencies
```
pip install -r requirements.txt
```

Optioneel kun je ook per subproject installeren:
- `Instagram Fake Profile Detector/requirements.txt`
- `TikTok Fake Profile Detector/requirements.txt`

## Gebruik (CLI)

### Instagram CLI
```
cd "Instagram Fake Profile Detector"
python -m src.cli --manual
python -m src.cli username
python -m src.cli https://www.instagram.com/username/
```

### TikTok CLI
```
cd "TikTok Fake Profile Detector"
python scripts/predict_account.py
```

## Training (samenvatting)

### Instagram
- Account model:
  ```
  cd "Instagram Fake Profile Detector"
  python -m src.train
  ```
- CNN model:
  ```
  cd "Instagram Fake Profile Detector"
  python -m src.train_cnn
  ```
- Frame extractie:
  ```
  cd "Instagram Fake Profile Detector"
  python scraping/extract_video_frames.py --fps 1 --limit 10 --skip-existing
  ```

### TikTok
- Live account model:
  ```
  cd "TikTok Fake Profile Detector"
  python scripts/train_model.py --feature-set live
  ```
- Live + video account model:
  ```
  cd "TikTok Fake Profile Detector"
  python scripts/train_model.py --feature-set live-video
  ```
- Full dataset model:
  ```
  cd "TikTok Fake Profile Detector"
  python scripts/train_model.py
  ```
- Image model:
  ```
  cd "TikTok Fake Profile Detector"
  python scripts/train_image_model.py
  ```

## Omgevingsvariabelen

### Instagram login (voor prive accounts of rate limits)
```
IG_USERNAME=your_user
IG_PASSWORD=your_pass
```

Windows (CMD):
```
set IG_USERNAME=your_user
set IG_PASSWORD=your_pass
```

### Streamlit Cloud secrets (voorbeeld)
```
IG_USERNAME = "your_user"
IG_PASSWORD = "your_pass"
```

### TikTok timeouts en retries (optioneel)
- `TIKTOK_CONNECT_TIMEOUT` (default 10)
- `TIKTOK_READ_TIMEOUT` (default 30)
- `TIKTOK_MAX_RETRIES` (default 1)
- `TIKTOK_RETRY_BACKOFF_MS` (default 800)

## Projectstructuur (hoog niveau)

- `app.py` - centrale Streamlit UI voor Instagram + TikTok
- `Instagram Fake Profile Detector/`
  - `src/` - features, training, predictie, scraping
  - `streamlit_app.py` - Instagram UI
  - `Data/` - datasets + frames (hier enkel demo data)
  - `models/` - getrainde modellen + metrics
- `TikTok Fake Profile Detector/`
  - `scripts/` - training en live predictie
  - `app.py` - TikTok UI
  - `data/dataset/` - datasets + frames (hier enkel demo data)
  - `models/` - getrainde modellen + metrics

## Aandachtspunten

- TikTok kan requests blokkeren (bot checks). Probeer later opnieuw of verhoog timeouts.
- Instagram scraping kan rate limits of login vereisen.
- Feature volgorde is strikt (Instagram) en wordt bij TikTok opgeslagen in `feature_columns.json`.
- Voor Instagram video frames is `ffmpeg` nodig in je PATH.
- TikTok video fallback gebruikt `yt-dlp` (zit in `requirements.txt`), maar kan nog steeds falen door blokkades.