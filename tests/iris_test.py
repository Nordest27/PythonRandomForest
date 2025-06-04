
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from class_reg_decision_tree import gini_impurity, gini_best_splits, CartNode, Cart

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

import json

iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy * 100:.2f}%')

# conf_matrix = confusion_matrix(y_test, y_pred)

# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, 
#             xticklabels=iris.target_names, yticklabels=iris.target_names)

# plt.title('Confusion Matrix Heatmap')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.show()
print(gini_impurity(y))

print(json.dumps(gini_best_splits(X, y), indent=2))

cart = Cart(X, y)
print(cart.predict(X.head(5)))

cart.fit()
print(cart.predict(X[45:55]))