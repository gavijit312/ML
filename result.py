# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Step 2: Create dataset of 200 students
np.random.seed(42)

hours = np.random.randint(1, 11, 200)   # Study hours between 1 to 10
marks = 5 * hours + 30 + np.random.randint(-5, 6, 200)  # Linear pattern + small noise

# Step 3: Make DataFrame
df = pd.DataFrame({
    "Hours": hours,
    "Marks": marks
})

# Step 4: Define X and y
X = df[["Hours"]]
y = df["Marks"]

# Step 5: Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Create Linear Regression model
model = LinearRegression()

# Step 7: Train the model
model.fit(X_train, y_train)

# Step 8: Predict test values
y_pred = model.predict(X_test)

# Step 9: Check accuracy
accuracy = r2_score(y_test, y_pred)
print("Accuracy (R² Score):", accuracy)

# Step 10: Print equation values
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

# Step 11: Predict for a new student
new_hours = [[7]]
predicted_marks = model.predict(new_hours)
print("Predicted Marks for 7 hours study:", predicted_marks[0])

# Step 12: Plot graph
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression with 200 Data Points")
plt.legend()
plt.show()
