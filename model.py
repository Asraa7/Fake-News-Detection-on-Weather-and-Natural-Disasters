import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import re

# ── Text Cleaning ──────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)           # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()           # remove extra spaces
    return text

# ── Train Model ────────────────────────────────────────────────────────────────
def train_model():
    # Sample dataset — weather & natural disaster fake vs real news
    data = {
        "text": [
            # REAL news examples
            "Hurricane Milton made landfall in Florida causing widespread flooding and power outages",
            "Scientists confirm that rising sea temperatures are increasing hurricane intensity",
            "Earthquake of magnitude 6.5 strikes Japan causing minor damage and no casualties",
            "Flood warnings issued across coastal regions as heavy rainfall continues",
            "Wildfire in California spreads across 10000 acres officials confirm",
            "Tornado warning issued for three counties in Oklahoma severe storm approaching",
            "Meteorologists predict above average rainfall this monsoon season due to El Nino",
            "Cyclone Biparjoy weakens after making landfall on Gujarat coast India",
            "Heavy snowfall disrupts transport in northern Europe as cold front advances",
            "Drought conditions worsen in East Africa threatening food security for millions",
            "Tsunami alert issued after 7.2 magnitude earthquake near Indonesia coast",
            "Climate scientists warn 2024 is on track to be the hottest year on record",
            "Flash floods displace thousands in Bangladesh low lying areas submerged",
            "Volcanic eruption in Iceland prompts evacuation of nearby towns",
            "Storm surge expected to reach 15 feet as category 4 hurricane approaches shore",

            # FAKE news examples
            "Government secretly controlling weather using HAARP machines to cause floods",
            "Chemtrails from planes are causing fake storms across the United States",
            "Scientists paid by billionaires to lie about climate change data it is all fake",
            "5G towers causing earthquakes in Japan new research proves link",
            "Hurricane was created by deep state to destroy Trump voting counties claim insider",
            "Earthquake prediction app suppressed by government because it works too well",
            "Floods in Europe caused by secret cloud seeding program by UN globalists",
            "Tornado machines being tested by military in Oklahoma residents report",
            "NASA hiding real data about sea level because it would panic the population",
            "Wildfires in California started deliberately by space laser satellites",
            "Vaccine chemicals in rainwater causing floods to intensify claim anti vaxx sites",
            "Volcanoes are being triggered by underground nuclear tests says anonymous source",
            "Cyclone formed because India tested new weather weapon say Pakistani media",
            "Snow in summer is proof global warming is a hoax scientists are lying",
            "Tsunami alert was fake news created to test mass panic response by government",
        ],
        "label": [
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  # 1 = REAL
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0   # 0 = FAKE
        ]
    }

    df = pd.DataFrame(data)
    df["clean_text"] = df["text"].apply(clean_text)

    # TF-IDF Vectorisation
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    # Train Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)

    print("=== Logistic Regression ===")
    print(f"Accuracy: {accuracy_score(y_test, lr_preds):.2f}")
    print(classification_report(y_test, lr_preds, target_names=["FAKE", "REAL"]))

    print("=== Naive Bayes ===")
    print(f"Accuracy: {accuracy_score(y_test, nb_preds):.2f}")
    print(classification_report(y_test, nb_preds, target_names=["FAKE", "REAL"]))

    # Save model and vectorizer
    with open("model.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("\n✅ Model trained and saved successfully!")

# ── Predict ────────────────────────────────────────────────────────────────────
def predict_news(text):
    if not os.path.exists("model.pkl"):
        train_model()

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    confidence = round(max(probability) * 100, 2)

    label = "REAL ✅" if prediction == 1 else "FAKE ❌"
    return {"label": label, "confidence": confidence}

if __name__ == "__main__":
    train_model()
