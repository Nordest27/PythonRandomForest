import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import time

from class_reg_decision_tree import RandCart

from ucimlrepo import fetch_ucirepo

student_performance = fetch_ucirepo(id=320)

X = student_performance.data.features
y = student_performance.data.targets["G3"] > 10

print(X)
print(y.value_counts())

# Encode categorical features
categorical_cols = X.select_dtypes(include=["object"]).columns
X_encoded = X.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le


# Split into training and testing sets
X_enc_train, X_enc_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Example: Train a simple Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(
    criterion="gini", splitter="best", min_samples_split=6, random_state=42, max_depth=6
)
ini_time = time.time()
clf.fit(X_enc_train, y_train)
end_time = time.time()
print("Train time:", end_time - ini_time)

# Predict and evaluate
y_pred = clf.predict(X_enc_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cart = RandCart(X_train, y_train, use_progress_bar=True)

ini_time = time.time()
cart.fit()
end_time = time.time()
print("Train time:", end_time - ini_time)

y_probs = cart.predict(X_test)
print(y_probs)
cart.print_tree()

y_pred = y_probs.idxmax(axis=1).astype(str)
y_test = y_test.astype(str)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
