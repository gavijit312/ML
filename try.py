import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


np.random.seed(42)

hours = np.random.randint(1, 11, 200)
scores = hours * 10 + np.random.normal(0, 10, 200)
X = hours.reshape(-1, 1)
y = scores

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)   
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)



print("Actual values:", y_test)
print("Predicted values:", y_pred)
print("Mean Squared Error (MSE):", mse)
