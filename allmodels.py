# =====================================
# SPAM vs HAM - ALL MODELS COMPARISON
# WITH FINAL COMPARISON TABLE
# =====================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# -------------------------------
# LOAD DATASET
# -------------------------------
data = pd.read_csv("spam.csv", encoding="latin-1")
data.columns = ["Label", "Message"]

X = data["Message"]
y = data["Label"].str.lower().map({"ham": 0, "spam": 1})

# -------------------------------
# TEXT VECTORIZATION
# -------------------------------
vectorizer = TfidfVectorizer(
    max_features=1500,
    stop_words="english"
)
X_vec = vectorizer.fit_transform(X)

# -------------------------------
# TRAIN-TEST SPLIT (70-30)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# -------------------------------
# EVALUATION FUNCTION
# -------------------------------
results = []

def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    tp, fp = cm[1,1], cm[0,1]
    fn, tn = cm[1,0], cm[0,0]

    print(f"\n{name}")
    print("-" * len(name))
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1-score :", round(f1, 4))
    print("Confusion Matrix (TP FP / FN TN)")
    print([tp, fp])
    print([fn, tn])

    results.append([
        name,
        round(acc, 4),
        round(prec, 4),
        round(rec, 4),
        round(f1, 4)
    ])

# -------------------------------
# MODELS
# -------------------------------
evaluate_model(
    "Random Forest",
    RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42)
)

evaluate_model(
    "Extra Trees",
    ExtraTreesClassifier(n_estimators=200, max_depth=25, random_state=42)
)

evaluate_model(
    "Gradient Boosting",
    GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, random_state=42)
)

evaluate_model(
    "XGBoost",
    XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )
)

evaluate_model(
    "LightGBM",
    LGBMClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
)

evaluate_model(
    "CatBoost",
    CatBoostClassifier(iterations=150, learning_rate=0.1, verbose=0, random_state=42)
)

# -------------------------------
# FINAL COMPARISON TABLE
# -------------------------------
comparison_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"]
)

print("\n" + "=" * 50)
print("FINAL MODEL COMPARISON TABLE")
print("=" * 50)
print(comparison_df)
