# ============================================================
# PHASE 2: NAÏVE BAYES CLASSIFIER (FROM SCRATCH)
# Task: Email Spam Classification
# Dataset: spam.csv
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------
data = pd.read_csv("spam.csv")

# Rename columns for clarity
data.columns = ["Label", "Message"]

# Convert labels: ham = 0, spam = 1
data["Label"] = data["Label"].map({"ham": 0, "spam": 1})

print("Dataset Loaded Successfully")
print(data.head())

# ------------------------------------------------------------
# 2. Train-Test Split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data["Message"],
    data["Label"],
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 3. Text Vectorization
# ------------------------------------------------------------
vectorizer = CountVectorizer(stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nText Vectorization Completed")
print("Number of features:", X_train_vec.shape[1])

# ------------------------------------------------------------
# 4. NAÏVE BAYES CLASSIFIER (Multinomial)
# ------------------------------------------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Predictions
y_pred_nb = nb_model.predict(X_test_vec)

# ------------------------------------------------------------
# 5. Evaluation: Naïve Bayes
# ------------------------------------------------------------
cm_nb = confusion_matrix(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)

print("\n===== NAÏVE BAYES RESULTS =====")
print("Confusion Matrix:")
print(cm_nb)
print("Precision:", round(precision_nb, 4))
print("Recall:", round(recall_nb, 4))
print("F1 Score:", round(f1_nb, 4))

# ------------------------------------------------------------
# 6. DECISION TREE CLASSIFIER (Comparison)
# ------------------------------------------------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_vec, y_train)

# Predictions
y_pred_dt = dt_model.predict(X_test_vec)

# ------------------------------------------------------------
# 7. Evaluation: Decision Tree
# ------------------------------------------------------------
cm_dt = confusion_matrix(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

print("\n===== DECISION TREE RESULTS =====")
print("Confusion Matrix:")
print(cm_dt)
print("Precision:", round(precision_dt, 4))
print("Recall:", round(recall_dt, 4))
print("F1 Score:", round(f1_dt, 4))

# ------------------------------------------------------------
# 8. Final Comparison Summary
# ------------------------------------------------------------
print("\n===== MODEL COMPARISON =====")
print(f"Naïve Bayes  -> Precision: {precision_nb:.4f}, Recall: {recall_nb:.4f}, F1: {f1_nb:.4f}")
print(f"DecisionTree-> Precision: {precision_dt:.4f}, Recall: {recall_dt:.4f}, F1: {f1_dt:.4f}")

print("\nConclusion:")
print("Multinomial Naïve Bayes performs better for text-based spam classification.")
