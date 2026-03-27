from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

X  = np.array([[1], [2], [3], [4], [5]])
 
y = np.array([[40], [45], [50], [55], [60]])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state=42)

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
pred= model.predict([[6]])

print(y_pred)
# y = mx + c
# y = 5x + 35 
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

#shows actual data points
plt.scatter(X, y)
#shows predicted data points and regression line
plt.plot(X_test, y_pred)

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression Example")
plt.show()



