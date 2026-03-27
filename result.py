from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

X  = np.array([[1], [2], [3], [4], [5]])
 
y = np.array([[40], [45], [50], [55], [60]])

model = LinearRegression()

model.fit(X,y)

y_pred = model.predict(X)
pred= model.predict([[6]])

print(y_pred)
# y = mx + c
# y = 5x + 35 
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression Example")
plt.show()

