import os
import pickle
import re

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "fake_news_dataset.csv")
MODEL_PATH = os.path.join(BASE, "model.pkl")
STATS_PATH = os.path.join(BASE, "stats.pkl")
UNCERTAIN_BELOW = 60  # percent


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data():
    df = pd.read_csv(CSV_PATH)
    df["clean"] = df["News Text"].apply(clean_text)
    return df["clean"], df["Label (0/1)"]          # 1 = REAL, 0 = FAKE


def build_pipeline(clf):
    # TF-IDF is inside the pipeline so it is re-fitted on each training fold (no leakage)
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", clf),
    ])


def get_candidates():
    return {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    }


def train_model():
    X, y = load_data()

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1"]
    comparison = []

    for name, clf in get_candidates().items():
        res = cross_validate(build_pipeline(clf), X, y, cv=cv, scoring=scoring)
        preds = cross_val_predict(build_pipeline(clf), X, y,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=42))
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        row = {"model": name,
               "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}
        for m in scoring:
            vals = res[f"test_{m}"]
            row[m] = round(float(vals.mean()) * 100, 1)
            row[m + "_std"] = round(float(vals.std()) * 100, 1)
        comparison.append(row)

    best = max(comparison, key=lambda r: r["f1"])
    best_name = best["model"]

    # Final model: best classifier, probability-calibrated, fitted on all 100 samples
    final = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", CalibratedClassifierCV(get_candidates()[best_name], cv=3, method="sigmoid")),
    ])
    final.fit(X, y)

    stats = {
        "accuracy": best["accuracy"], "precision": best["precision"],
        "recall": best["recall"], "f1": best["f1"],
        "total_samples": int(len(y)),
        "real_count": int((y == 1).sum()), "fake_count": int((y == 0).sum()),
        "selected": best_name,
        "comparison": comparison,
        "method": "5-fold stratified cross-validation repeated 3 times",
    }

    print("=== Model comparison (5-fold CV x3, mean +/- std) ===")
    for r in comparison:
        print(f"{r['model']:20s} acc {r['accuracy']}+/-{r['accuracy_std']}  "
              f"prec {r['precision']}  rec {r['recall']}  f1 {r['f1']}+/-{r['f1_std']}")
    print("Selected:", best_name)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final, f)
    with open(STATS_PATH, "wb") as f:
        pickle.dump(stats, f)
    print("Model trained and saved!")
    return stats


def get_model_stats():
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, "rb") as f:
            return pickle.load(f)
    return None


def predict_news(text):
    if not os.path.exists(MODEL_PATH):
        train_model()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    proba = model.predict_proba([clean_text(text)])[0]      # [P(fake), P(real)]
    fake_p, real_p = float(proba[0]), float(proba[1])
    confidence = round(max(fake_p, real_p) * 100, 1)

    if confidence < UNCERTAIN_BELOW:
        verdict, label = "UNCERTAIN", "⚠️ Uncertain"
        explanation = "The model is not confident enough to make a reliable prediction for this text."
    elif real_p > fake_p:
        verdict, label = "REAL", "✅ Real News"
        explanation = "This text resembles the real news reports the model was trained on."
    else:
        verdict, label = "FAKE", "❌ Fake News"
        explanation = "This text resembles the misinformation patterns the model was trained on."

    return {"verdict": verdict, "label": label, "confidence": confidence,
            "explanation": explanation, "fake_prob": round(fake_p * 100, 1),
            "real_prob": round(real_p * 100, 1)}


if __name__ == "__main__":
    train_model()
