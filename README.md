# 🌪️ Fake News Detection — Weather & Natural Disasters

A Flask web app that classifies weather and natural-disaster news headlines as **Real**, **Fake**, or **Uncertain**, using TF-IDF features and a calibrated classifier chosen by comparing **Naive Bayes** and **Logistic Regression**.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-orange?style=flat-square&logo=scikit-learn)

> **Status:** a learning prototype trained on a small hand-written dataset. See [Limitations](#limitations).

## Screenshots
_Add 2 screenshots here: `![Home](screenshots/home.png)` and `![Compare](screenshots/compare.png)`_

## Features
- Paste a headline and get a verdict with a confidence bar and real/fake probabilities
- **Uncertain** result when confidence is below 60% instead of forcing a guess
- `/compare` page: Naive Bayes vs Logistic Regression with metrics and confusion matrices
- Dark navy UI, responsive, input validation, loading and error states

## Model comparison
Both models use the same TF-IDF features (1–2 word n-grams, 1000 features). TF-IDF is fitted inside each cross-validation fold to avoid leakage. Evaluation: 5-fold stratified cross-validation repeated 3 times on 100 samples.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 89.7% ± 6.4 | 88.6% | 92.0% | 90.0% ± 6.0 |
| Logistic Regression | 89.0% ± 6.9 | 86.5% | 93.3% | 89.5% ± 6.4 |

The difference between the two is far smaller than the fold-to-fold variation, so **the dataset cannot tell them apart**. The app uses the one with the higher mean F1 (Naive Bayes in this run) with probability calibration. Run `python train.py` to reproduce; the `/compare` page shows your own run.

## Dataset
`fake_news_dataset.csv`: 100 headlines (50 real, 50 fake) written by the author in the style of real weather reporting and common weather conspiracy claims. It is **not** collected from news sources.

## Project structure
```
app.py            Flask routes (/, /predict, /compare, /stats)
model.py          cleaning, model comparison, training, prediction
train.py          trains and saves the model
fake_news_dataset.csv
templates/        index.html (detector), compare.html (model comparison)
```

## Run locally
```bash
pip install -r requirements.txt
python train.py
python app.py          # open http://127.0.0.1:5000
```

## Limitations
- Only 100 headlines written by one person; real-world accuracy will be lower than the scores above.
- The model learns wording patterns (e.g. "secret", "elites"), not facts. It cannot verify a claim.
- Confident mistakes are possible: in a quick check with 12 new headlines it got one real headline wrong at about 79% confidence.
- English only; not deployed; does not learn after training.

## Future work
Train on a larger public fake-news dataset, add an out-of-domain test set, and deploy online.

## Author
**Asraa Falak. Y**, B.Sc. AI & Data Science, Gonzaga College of Arts and Science for Women · [LinkedIn](https://linkedin.com/in/asraa-falak)
