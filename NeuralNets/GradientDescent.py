import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.random.rand(100, 1)
y = 4 * X + 3 + np.random.randn(100, 1)

class GradientDescent:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = 0
        self.b = 0

    def fitSlow(self, X, y) :
        n = len(X)
        for j in range(self.epochs):
            slope_m = 0
            slope_b = 0
            for i in range(n):
                predict = self.m * X[i] + self.b
                error = y[i] - predict
                slope_m += (-2/n) * X[i]*error
                slope_b += (-2/n) * error
            self.m -= self.learning_rate * slope_m        
            self.b -= self.learning_rate * slope_b
            
    
    def fitFast(self, X, y) :
        n = len(X)
        for j in range(self.epochs) :
            predict = self.m * X + self.b
            error = y - predict
            slope_m = (-2/n) * np.sum(X * error)
            slope_b = (-2/n) * np.sum(error)
            self.m -= self.learning_rate * slope_m
            self.b -= self.learning_rate * slope_b
            
    def predict(self, X):
        return self.m * X + self.b            
            

model = GradientDescent(learning_rate=0.1, epochs=1000)
model.fitFast(X, y)

y_pred = model.predict(X)

plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.legend()
plt.show()
