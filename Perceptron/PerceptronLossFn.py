from sklearn.datasets import make_classification
import numpy as np
X, y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=41, hypercube=False, class_sep=10)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100)

def perceptron(X, y) :
    w1 = 1
    w2 = 5
    b = 6
    lr = 0.1
    for j in range(1000):
        for i in range(X.shape[0]) :
            z = w1*X[i][0] + w2*X[i][1] + b
            if z*y[i] < 0 :
                w1 = w1 + lr * y[i] * X[i][0]
                w2 = w2 + lr * y[i] * X[i][1]
                b = b + lr * y[i]
                
    return w1, w2, b

w1, w2, b = perceptron(X, y)

m = -(w1/w2)
b = -(b/w2)

x_input = np.linspace(-3, 3, 100)
y_input = m*x_input + b

plt.figure(figsize=(10, 6))
plt.plot(x_input, y_input, color='red', linewidth = 3)
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = 'winter', s=100)
plt.ylim(-3, 2)
plt.show()
 