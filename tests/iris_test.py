
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from class_reg_decision_tree import RandCart

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# These will be DataFrames and Series, not numpy arrays
assert isinstance(X_train, pd.DataFrame)
assert isinstance(y_train, pd.Series)

scaler = StandardScaler()
np_X_train = scaler.fit_transform(X_train)
np_X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(np_X_train, y_train)
y_pred = classifier.predict(np_X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

cart = RandCart(X_train, y_train, use_progress_bar=True)

cart.fit()
y_probs = cart.predict(X_test)
print(y_probs)
cart.print_tree()

y_pred = y_probs.idxmax(axis=1).astype(str)
y_test = y_test.astype(str)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
