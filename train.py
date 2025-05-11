import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression


X, y = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

fig, ax = plt.subplots(figsize=(8, 6))
plt.scatter(X[:,0], y, color = "b",marker="o", s = 30, label = "Data")
plt.show()

reg = LinearRegression()
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse = mse(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"Variance of y_test: {np.var(y_test):.2f}")
print(f"Standard Deviation of y_test: {np.std(y_test):.2f}")

y_pred_line = reg.predict(X)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color = cmap(0.9), marker="o", s = 10, label = "Train Data")
m2 = plt.scatter(X_test, y_test, color = cmap(0.5), marker="o", s = 10, label = "Test Data")
plt.plot(X, y_pred_line, color = "r", label = "Prediction")
plt.show()
