# Instagram Fake Detector

Detecteer of een Instagram account waarschijnlijk fake of echt is op basis van
publieke profielkenmerken. Dit project gebruikt een RandomForest model dat
getraind is op een dataset van Instagram profielen.

## Inhoud

- Overzicht
- Dataset
- Features
- Model en training
- Installatie
- Gebruik
- Streamlit interface
- Omgevingsvariabelen (privete accounts)
- Projectstructuur
- Ontwikkelnotities en keuzes

## Overzicht

Het doel is om een snelle, eenvoudige detector te maken die werkt op basis van
profiel metadata zoals bio lengte, aantal volgers, en de verhouding cijfers in
de username. We bouwen features, trainen een model en bieden zowel een CLI als
een Streamlit interface.

## Dataset

Bestand: `Data/Instagram_fake_profile_dataset.csv`

De CSV bevat de volgende kolommen:
- `fake` (target, 0 of 1)
- Alle feature kolommen zoals gedefinieerd in `src/config.py`

Extra dataset: `Data/ig_post_metadata.csv` bevat post-metadata per account.
Deze dataset wordt gebruikt om videoframes te verzamelen voor het CNN model.

Let op: de code zoekt zowel `Data/` als `data/` zodat het werkt op
case-sensitive systemen.

## Features

De feature set is gebaseerd op profiel eigenschappen:
- `profile pic`: 1 als er een profielfoto is
- `nums/length username`: verhouding cijfers in username
- `fullname words`: aantal woorden in fullname
- `nums/length fullname`: verhouding cijfers in fullname
- `name==username`: 1 als fullname gelijk is aan username
- `description length`: lengte van de bio
- `external URL`: 1 als er een externe URL is
- `private`: 1 als het account prive is
- `#posts`, `#followers`, `#follows`

De functies die dit bouwen zitten in `src/features.py`.

## Model en training

Model: `RandomForestClassifier` met `MinMaxScaler` in een sklearn `Pipeline`.
We gebruiken een train/test split met stratificatie en rapporteren:
- classification report
- ROC-AUC

Trainen:
```
python -m src.train
```

Het getrainde model wordt opgeslagen in `models/instagram_fake_detector.joblib`.

### CNN model (video frames)

Gebruik `ig_post_metadata.csv` om videoframes te downloaden en train daarna een CNN:
```
python scraping/extract_video_frames.py --fps 1 --limit 10 --skip-existing
python -m src.train_cnn
```

Het getrainde model wordt opgeslagen in `models/instagram_video_cnn.pt`.

## Installatie

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Gebruik (CLI)

1) Handmatige input:
```
python -m src.cli --manual
```

2) Met username of profiel URL:
```
python -m src.cli username
python -m src.cli https://www.instagram.com/username/
```

## Streamlit interface

Start de web UI:
```
streamlit run streamlit_app.py
```

Je kunt kiezen tussen het account-model (profieldata) of het CNN video-model.
Het CNN model gebruikt de nieuwste videoposts van een username voor inferentie.

## Video frame pipeline

Er is een helper om videoframes te downloaden op basis van `ig_post_metadata.csv`.
Deze gebruikt `ffmpeg` voor frame-extractie.

Voorbeeld:
```
python scraping/extract_video_frames.py --fps 1 --limit 10 --skip-existing
```

Voor live inferentie in de Streamlit UI worden frames on-the-fly gedownload en
geëxtraheerd. Dit vereist ook `ffmpeg` en eventueel `IG_USERNAME/IG_PASSWORD`
voor private accounts.

## Omgevingsvariabelen (privete accounts)

Voor privete profielen of rate limits kan login nodig zijn:
```
IG_USERNAME=your_user IG_PASSWORD=your_pass streamlit run streamlit_app.py
```

Of voor de CLI:
```
IG_USERNAME=your_user IG_PASSWORD=your_pass python -m src.cli username
```


## Projectstructuur

- `src/config.py`: paden en target kolommen
- `src/features.py`: feature engineering
- `src/train.py`: training pipeline en model export
- `src/train_cnn.py`: CNN training op videoframes
- `src/cnn_data.py`: dataset helpers voor frames
- `src/cnn_model.py`: CNN architectuur en transforms
- `src/cnn_predict.py`: CNN inferentie op username
- `src/cnn_live.py`: download en extractie van frames uit de nieuwste posts
- `src/predict.py`: model laden en predictie
- `src/instagram.py`: ophalen van profiel metadata via instaloader
- `src/cli.py`: command line interface
- `streamlit_app.py`: Streamlit UI
- `Data/`: dataset
- `models/`: getraind model (na training)

## Ontwikkelnotities en keuzes

- We gebruiken alleen publieke metadata zodat de pipeline snel en simpel blijft.
- De feature volgorde is strikt; model input verwacht exact die kolommen.
- De Instagram fetch gebruikt `instaloader` omdat het stabiel is en geen eigen API key vereist.
- Voor privete accounts is inloggen nodig (via env vars).
