import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(s):
    return s * (1 - s)

x = 0.5
y_true = 1
np.random.seed(42)
w11_1, w12_1 = np.random.randn(2) * 0.1
b1_1, b2_1 = 0.0, 0.0

w11_2, w21_2 = np.random.randn(2) * 0.1
w12_2, w22_2 = np.random.randn(2) * 0.1
b1_2, b2_2 = 0.0, 0.0

w1_3, w2_3 = np.random.randn(2) * 0.1
b3 = 0.0

lr = 0.5

for epoch in range(1, 10000):
    # FORWARD PASS 
    z1_1 = x * w11_1 + b1_1
    h1_1 = sigmoid(z1_1)

    z2_1 = x * w12_1 + b2_1
    h2_1 = sigmoid(z2_1)

    z1_2 = h1_1 * w11_2 + h2_1 * w21_2 + b1_2
    h1_2 = sigmoid(z1_2)

    z2_2 = h1_1 * w12_2 + h2_1 * w22_2 + b2_2
    h2_2 = sigmoid(z2_2)

    z3 = h1_2 * w1_3 + h2_2 * w2_3 + b3
    y_hat = sigmoid(z3)

    loss = 0.5 * (y_hat - y_true) ** 2
    print(f"\nEpoch {epoch}")
    print(f"Forward: y_hat={y_hat:.6f}, loss={loss:.6f}")

    # BACKPROP 
    dL_dyhat = y_hat - y_true
    dz3 = dL_dyhat * sigmoid_derivative(y_hat)

    # Output layer
    dL_dw1_3 = dz3 * h1_2
    dL_dw2_3 = dz3 * h2_2
    dL_db3 = dz3

    # Backprop to Layer 3
    dh1_2 = dz3 * w1_3
    dh2_2 = dz3 * w2_3

    dz1_2 = dh1_2 * sigmoid_derivative(h1_2)
    dz2_2 = dh2_2 * sigmoid_derivative(h2_2)

    dL_dw11_2 = dz1_2 * h1_1
    dL_dw21_2 = dz1_2 * h2_1
    dL_db1_2 = dz1_2  

    dL_dw12_2 = dz2_2 * h1_1
    dL_dw22_2 = dz2_2 * h2_1
    dL_db2_2 = dz2_2  

    # Backprop to Layer 2
    dh1_1 = dz1_2 * w11_2 + dz2_2 * w12_2
    dh2_1 = dz1_2 * w21_2 + dz2_2 * w22_2

    dz1_1 = dh1_1 * sigmoid_derivative(h1_1)
    dz2_1 = dh2_1 * sigmoid_derivative(h2_1)

    dL_dw11_1 = dz1_1 * x
    dL_dw12_1 = dz2_1 * x
    dL_db1_1 = dz1_1
    dL_db2_1 = dz2_1

    # UPDATE WEIGHTS 
    w1_3 -= lr * dL_dw1_3
    w2_3 -= lr * dL_dw2_3
    b3 -= lr * dL_db3

    w11_2 -= lr * dL_dw11_2
    w21_2 -= lr * dL_dw21_2
    w12_2 -= lr * dL_dw12_2
    w22_2 -= lr * dL_dw22_2
    b1_2 -= lr * dL_db1_2
    b2_2 -= lr * dL_db2_2

    w11_1 -= lr * dL_dw11_1
    w12_1 -= lr * dL_dw12_1
    b1_1 -= lr * dL_db1_1
    b2_1 -= lr * dL_db2_1

    print(f"Gradients: dL_dw1_3={dL_dw1_3:.6f}, dL_dw11_2={dL_dw11_2:.6f}, dL_dw11_1={dL_dw11_1:.6f}")
