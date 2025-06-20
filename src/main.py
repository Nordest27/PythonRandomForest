from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, confusion_matrix
from random_forest import RandomForest


X, y = fetch_openml("letter", version=1, return_X_y=True, as_frame=True)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

n_estimators = 99
max_features = 5
max_depth = 25
bootstrap_size = 10000
min_samples_split = 2

print("Generating the random forest...")
rf = RandomForest(
    n_estimators,
    max_features,
    max_depth,
    bootstrap_size,
    min_samples_split,
)
rf.fit(X_train, y_train, show="progress")

print("Predicting the testing set...")
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# conf_matrix = confusion_matrix(y_test, y_pred)
# print(conf_matrix)


###### Random forest from sklearn ######

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

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

clf = RandomForestClassifier(
    criterion="gini",
    random_state=42,
    max_depth=max_depth,
    n_estimators=n_estimators,
    min_samples_split=min_samples_split,
    bootstrap=True,
    max_features=max_features,
    max_samples=bootstrap_size,
)
clf.fit(X_enc_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_enc_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy sklearn: {accuracy * 100:.2f}%")
