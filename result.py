import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

np.random.seed(42)

hours = np.random.randint(1, 11, 200)   
marks = 5 * hours + 30 + np.random.randint(-5, 6, 200)  

df = pd.DataFrame({
    "Hours": hours,
    "Marks": marks
})

X = df[["Hours"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Step 9: Check accuracy
accuracy = r2_score(y_test, y_pred)
print("Accuracy (R² Score):", accuracy)


print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

new_hours = [[7]]
predicted_marks = model.predict(new_hours)
print("Predicted Marks for 7 hours study:", predicted_marks[0])


plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression with 200 Data Points")
plt.legend()
plt.show()
