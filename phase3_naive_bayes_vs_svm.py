
# ============================================
# PHASE 3: Naïve Bayes vs SVM (Spam Detection)
# ============================================

import pandas as pd
import time
from io import StringIO

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_curve, auc

import matplotlib.pyplot as plt

# Load Dataset
data_text = """Label,Message
spam,"Win a free iPhone now!!!"
ham,"Hey, did you finish the report?"
spam,"Claim your $1000 cash prize today!"
ham,"Can you join the Zoom call at 3 PM?"
spam,"Congratulations! You have won a prize!"
ham,"Don't forget to bring the documents tomorrow."
"""

df = pd.read_csv(StringIO(data_text))
df["Label"] = df["Label"].map({"ham": 0, "spam": 1})

X = df["Message"]
y = df["Label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Naïve Bayes Model
nb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("nb", MultinomialNB())
])

start_nb = time.time()
nb_pipeline.fit(X_train, y_train)
nb_training_time = time.time() - start_nb

nb_predictions = nb_pipeline.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)

# SVM Model
svm_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("svm", SVC(probability=True))
])

param_grid = {
    "svm__C": [0.1, 1, 10],
    "svm__kernel": ["linear", "rbf"]
}

grid_search = GridSearchCV(svm_pipeline, param_grid, cv=2)

start_svm = time.time()
grid_search.fit(X_train, y_train)
svm_training_time = time.time() - start_svm

svm_predictions = grid_search.best_estimator_.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)

# ROC Curve
nb_probabilities = nb_pipeline.predict_proba(X_test)[:, 1]
svm_probabilities = grid_search.best_estimator_.predict_proba(X_test)[:, 1]

fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_probabilities)
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_probabilities)

auc_nb = auc(fpr_nb, tpr_nb)
auc_svm = auc(fpr_svm, tpr_svm)

plt.figure()
plt.plot(fpr_nb, tpr_nb, label=f"Naïve Bayes (AUC = {auc_nb:.2f})")
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {auc_svm:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Naïve Bayes vs SVM")
plt.legend()
plt.show()

# Results
results = pd.DataFrame({
    "Model": ["Naïve Bayes", "SVM"],
    "Accuracy": [nb_accuracy, svm_accuracy],
    "Training Time (seconds)": [nb_training_time, svm_training_time]
})

print(results)
