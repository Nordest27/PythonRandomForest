
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import time

from class_reg_decision_tree import RandCart
from sklearn.datasets import fetch_openml

df = fetch_openml("adult", version=2, as_frame=True)
X, y = df.data, df.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cart = RandCart(X_train, y_train, use_progress_bar=True)

ini_time = time.time()
cart.fit()
end_time = time.time()
print("Train time:", end_time-ini_time)

y_probs = cart.predict(X_test)
cart.print_tree()

y_pred = y_probs.idxmax(axis=1).astype(str)
y_test = y_test.astype(str)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

