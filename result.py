from sklearn.linear_model import LinearRegression
import numpy as np

X  = np.array([[1], [2], [3], [4], [5]])

y = np.array([[40], [45], [50], [55], [60]])

model = LinearRegression()

model.fit(X,y)

pred= model.predict([[6]])

print(pred)
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)