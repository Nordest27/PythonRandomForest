
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

from class_reg_decision_tree import Cart

# Load the dataset
# Ensure 'student-mat.csv' is in your working directory
df = pd.read_csv('student-mat.csv', sep=';')

# Drop G1 and G2 to avoid data leakage, as G3 is the final grade
df = df.drop(columns=['G1', 'G2'])

# Define the target variable
# Convert G3 (0-20) into categorical labels: 0 (fail), 1 (pass), 2 (excellent)
def categorize_grade(grade):
    if grade < 10:
        return 0  # fail
    elif grade < 15:
        return 1  # pass
    else:
        return 2  # excellent

df['G3_cat'] = df['G3'].apply(categorize_grade)
df = df.drop(columns=['G3'])

# Separate features and target
X = df.drop(columns=['G3_cat'])
y = df['G3_cat']

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
X_encoded = X.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Example: Train a simple Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:')
print(conf_matrix)


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
cart = Cart(X_train, y_train)

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

