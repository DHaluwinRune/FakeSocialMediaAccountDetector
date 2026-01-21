# Instagram Fake Profile Detector

Dit is het Instagram-deel van het project. Deze map bevat de volledige pipeline
om Instagram-accounts als FAKE of REAL te classificeren op basis van:
- profielmetadata (RandomForest account-model)
- visuele signalen uit videos (CNN model)
- een fusion-score die beide combineert

## Inhoud

- Overzicht en doelen
- Snelstart
- Modellen en pipeline
- Data en features
- Gebruik (CLI en Streamlit)
- Scraping en frame-extractie
- Modellen en metrics
- Omgevingsvariabelen en dependencies
- Projectstructuur
- Beperkingen en aandachtspunten

## Overzicht en doelen

Doel: zo goed mogelijk voorspellen of een Instagram-account fake is zonder
privileges of API-keys. Het account-model gebruikt alleen publieke
profielkenmerken; het CNN-model kijkt naar visuele patronen in posts. De
Streamlit app biedt inzicht in de beslissing en ondersteunt meerdere modes
(account, video, fusion, screenshot).

## Snelstart

Installatie:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start de Streamlit UI:
```bash
streamlit run streamlit_app.py
```

CLI inferentie:
```bash
python -m src.cli username
python -m src.cli https://www.instagram.com/username/
python -m src.cli --manual
```

Modellen trainen (als er geen modellen aanwezig zijn of je wilt hertrainen):
```bash
python -m src.train
python -m src.train_cnn
```

## Modellen en pipeline

1) Account-model (RandomForest)
- Bron: profielmetadata via Instaloader of handmatige invoer
- Features: zie `src/features.py` en `src/config.py`
- Training: `python -m src.train`
- Output: `models/instagram_fake_detector.joblib`

2) CNN-model (ResNet18)
- Bron: videoframes per account (uit `Data/ig_post_metadata.csv`)
- Training: `python -m src.train_cnn`
- Output: `models/instagram_video_cnn.pt`

3) Fusion-model
- Combineert account- en CNN-score via gewogen gemiddelde
- Default gewichten: 0.7 account / 0.3 CNN
- Implementatie: `src/fusion_predict.py`

## Data en features

Verwachte data-locaties (case-sensitive systemen ondersteund):
- `Data/Instagram_fake_profile_dataset.csv` (of `data/Instagram_fake_profile_dataset.csv`)
- `Data/ig_post_metadata.csv` (of `data/ig_post_metadata.csv`)
- `Data/video_frames/` (of `data/video_frames/`)

Account features (volgorde is strikt, zie `src/config.py`):
- `profile pic`: 1 als er een profielfoto is
- `nums/length username`: ratio cijfers in username
- `fullname words`: aantal woorden in fullname
- `nums/length fullname`: ratio cijfers in fullname
- `name==username`: 1 als fullname gelijk is aan username
- `description length`: lengte van de bio
- `external URL`: 1 als er een externe URL is
- `private`: 1 als het account prive is
- `#posts`, `#followers`, `#follows`

Feature engineering staat in `src/features.py`. De predictie verwacht exact
deze kolomvolgorde (zie `src/predict.py`).

## Gebruik (CLI en Streamlit)

CLI:
- `python -m src.cli username` gebruikt Instaloader om profieldata op te halen
- `python -m src.cli --manual` vraagt handmatig alle feature-waarden

Streamlit modes (`streamlit_app.py`):
- Account (profieldata): RandomForest op metadata
- Video (CNN frames): downloadt recente videos, extraheert frames, inferentie
- Fusion (account + video): combineert beide met instelbare gewichten
- Screenshot (image): CNN inferentie op een geuploade screenshot

De Streamlit UI toont:
- prediction + fake probability
- feature-importance (RandomForest) voor de account mode
- een korte uitleg per mode (zie `build_*_explanation` helpers)

## Scraping en frame-extractie

1) Genereer post metadata en labels:
```bash
python scraping/scrape_post_metadata.py --input usernames.txt
```
Dit script gebruikt het account-model om accounts te labelen en schrijft
`ig_post_metadata.csv` met postkenmerken.

2) Download videos en extraheer frames:
```bash
python scraping/extract_video_frames.py --fps 1 --limit 10 --skip-existing
```
Vereist `ffmpeg` in je PATH.

Voor live inferentie (Streamlit Video/Fusion) worden frames on-the-fly
gedownload en uitgepakt via `src/cnn_live.py`. Bij gebrek aan video of ffmpeg
wordt teruggevallen op beeldposts.

## Modellen en metrics

In `models/` staan:
- `instagram_fake_detector.joblib`: account-model
- `instagram_video_cnn.pt`: CNN model
- `*.metrics.txt`: logregels met accuracy/precision/recall/f1/roc_auc
- `modality_comparison_summary.txt`: vergelijking account vs CNN
- `training_cost_analysis.txt`: kosteninschatting CPU/GPU

De training scripts schrijven metrics bij elke run, zodat je evolutie kan volgen.

## Omgevingsvariabelen en dependencies

Voor private accounts of rate limits is login nodig:
```bash
IG_USERNAME=your_user IG_PASSWORD=your_pass streamlit run streamlit_app.py
```

Belangrijke dependencies:
- `instaloader` voor Instagram scraping
- `ffmpeg` voor frame-extractie
- `torch`/`torchvision` voor CNN training en inferentie

## Projectstructuur

- `src/config.py`: paden, featurevolgorde, modelpaden
- `src/features.py`: feature engineering
- `src/train.py`: RandomForest training + metrics logging
- `src/predict.py`: account predictie
- `src/instagram.py`: Instaloader fetch + input parsing
- `src/cli.py`: CLI interface
- `src/cnn_model.py`: ResNet18 + transforms
- `src/cnn_data.py`: frame dataset + label parsing
- `src/cnn_predict.py`: CNN inferentie helpers
- `src/cnn_live.py`: live download + frame extractie
- `src/train_cnn.py`: CNN training + metrics logging
- `src/fusion_predict.py`: gewogen fusie
- `scraping/scrape_post_metadata.py`: metadata scraping + labeling
- `scraping/extract_video_frames.py`: video download + frames
- `streamlit_app.py`: UI met account/video/fusion/screenshot modes
- `models/`: opgeslagen modellen + logs

## Beperkingen en aandachtspunten

- Instagram rate limits en prive accounts kunnen scraping blokkeren; login helpt.
- De CNN presteert in de huidige logs zwakker dan het account-model omdat de data hiervan moeilijker is om te verzamelen aangezien we hiervoor zelf scraping moeten doen en je snel aan de rate-limits zit van instagram. Daardoor weegt het CNN model minder zwaar door in het fusion model
- De data-bestanden staan niet in deze map; plaats ze zelf in `Data/` of `data/`.

