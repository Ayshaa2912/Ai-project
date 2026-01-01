# ============================================================
# NAÏVE BAYES CLASSIFIER (SPAM EMAIL DETECTION)
# Dataset: spam.csv
# Model: Multinomial Naïve Bayes
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------
data = pd.read_csv("spam.csv")

# Rename columns
data.columns = ["Label", "Message"]

# Convert labels to binary (ham=0, spam=1)
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

# ------------------------------------------------------------
# 3. Text Vectorization
# ------------------------------------------------------------
vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ------------------------------------------------------------
# 4. Train Naïve Bayes Model
# ------------------------------------------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# ------------------------------------------------------------
# 5. Predictions
# ------------------------------------------------------------
y_pred = nb_model.predict(X_test_vec)

# ------------------------------------------------------------
# 6. Evaluation Metrics
# ------------------------------------------------------------
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ------------------------------------------------------------
# 7. Results
# ------------------------------------------------------------
print("\n===== NAÏVE BAYES RESULTS =====")
print("Confusion Matrix:")
print(conf_matrix)
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-Score:", round(f1, 4))
