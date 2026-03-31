import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Step 2: Create dataset
np.random.seed(42)

attendance = np.random.randint(50, 101, 200)     
cgpa = np.round(np.random.uniform(5.0, 10.0, 200), 1)   
marks = np.random.randint(40, 91, 200)          

# Combine features into X
X = np.column_stack((attendance, cgpa, marks))

# Output: Package / Salary
# Added more randomness so score stays around 0.8 or less
y = 0.015 * attendance + 0.25 * cgpa + 0.02 * marks + np.random.normal(0, 0.2, 200)

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create model
model = LinearRegression()

# Step 5: Train model
model.fit(X_train, y_train)

# Step 6: Predict
y_pred = model.predict(X_test)

# Step 7: Evaluate model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R² Score:", r2)
print("Mean Squared Error:", mse)

# Step 8: Coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Step 9: Predict new students
pred = model.predict([
    [40, 5, 40],
    [100, 10, 70],
    [60, 8, 55],
    [80, 9, 60]
])

print("Predicted Packages:", pred)