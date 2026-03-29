from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# attendence , cgpa, marks

X = np.array([
    [50, 6, 40],
    [52, 6.2, 41],
    [54, 6.4, 42],
    [56, 6.6, 43],
    [58, 6.8, 44],
    [60, 7.0, 45],
    [62, 7.2, 46],
    [64, 7.4, 47],
    [66, 7.6, 48],
    [68, 7.8, 49],
    [70, 8.0, 50],
    [72, 8.2, 51],
    [74, 8.4, 52],
    [76, 8.6, 53],
    [78, 8.8, 54],
    [80, 9.0, 55],
    [82, 9.2, 56],
    [84, 9.4, 57],
    [86, 9.6, 58],
    [90, 10.0, 60]
])

y = np.array([
    [4.5],
    [4.6],
    [4.7],
    [4.8],
    [4.9],
    [5.0],
    [5.1],
    [5.2],
    [5.3],
    [5.4],
    [5.5],
    [5.6],
    [5.7],
    [5.8],
    [5.9],
    [6.0],
    [6.1],
    [6.2],
    [6.3],
    [6.5]
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
pred = model.predict([[40, 5, 40], [100, 10, 70], [60, 8, 55], [80, 20, 60]])

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print(score)


