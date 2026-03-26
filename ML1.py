import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

MODEL_PATH = "random_forest_model.pkl"

# -----------------------------
# 1. Dataset
# -----------------------------
data = {
    "Hours_Studied": [
        1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
        6, 7, 7, 8, 8, 9, 9, 10, 10, 11,
        11, 12, 12, 13, 14, 15, 16, 17, 18, 20
    ],
    "Attendance": [
        35, 40, 45, 45, 50, 50, 55, 55, 60, 60,
        65, 65, 70, 70, 75, 75, 80, 80, 85, 85,
        88, 90, 92, 94, 95, 96, 97, 98, 99, 100          
    ],
    "Result": [
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)
X = df[["Hours_Studied", "Attendance"]]
y = df["Result"]


# -----------------------------
# 2. Load model if exists
# -----------------------------
if os.path.exists(MODEL_PATH):
    print(" Loading existing model...")
    model = joblib.load(MODEL_PATH)

else:
    print(" Training new model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Accuracy check
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(" Model trained and saved!")

# -----------------------------
# 3. Use the loaded model
# -----------------------------
sample_input = [[20, 100]]
prediction = model.predict(sample_input)

print("Prediction (1 = Pass, 0 = Fail):", prediction[0])