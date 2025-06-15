import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import time

from class_reg_decision_tree import RandCart
from random_forest import RandomForest
import seaborn as sns

df = sns.load_dataset("titanic").reset_index()
X = df.drop(columns=["index", "survived", "alive"])
y = df["survived"]

max_features = 5
max_depth = 10
min_samples_split = 1

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
y_test = y_test.astype(str)

print("Generating the tree...")
cart = RandCart(
    X_train,
    y_train,
    max_features_to_explore=max_features,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    use_progress_bar=True,
)
cart.fit()
# cart.print_tree()

y_probs = cart.predict(X_test)

y_pred = y_probs.idxmax(axis=1).astype(str)

accuracy = accuracy_score(y_test, y_pred)
print(f"CART Accuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
cart.print_tree()


#############################################
############### Random Forest ###############
#############################################

n_estimators = 39
bootstrap_size = len(X_train)

print("Generating the random forest...")
rf = RandomForest(
    n_estimators,
    max_features,
    max_depth,
    bootstrap_size,
    min_samples_split,
)
ini_time = time.time()
rf.fit(X_train, y_train, show="progress")
end_time = time.time()
print("Train time:", end_time - ini_time)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
