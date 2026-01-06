
# PHASE 4: Single Layer Perceptron using TensorFlow

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

data = pd.read_csv("spam_ham.csv")
data['Label'] = data['Label'].map({'ham': 0, 'spam': 1})

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['Message']).toarray()
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(X.shape[1],))
])

model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_split=0.1)

plt.plot(history.history['loss'])
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
