from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# attendence , cgpa, marks
X = np.array([[50, 6, 40] , [60, 7, 45], [70, 8, 50], [80, 9, 55], [90, 10, 60]])
y = np.array([[4.5], [5], [5.5], [6], [6.5]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
pred = model.predict([[40, 5, 40]])

print(y_pred)