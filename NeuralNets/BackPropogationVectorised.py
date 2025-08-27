import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(s):
    return s * (1 - s)

X = np.array([[0.5], [0.2], [0.8]])  
Y = np.array([[1], [0], [1]])        


input_Size = 1
hidden1_size = 2
hidden2_size = 2
output_size = 1

np.random.seed(42)
W1 = np.random.randn(input_Size, hidden1_size) * 0.1
b1 = np.zeros((1, hidden1_size))

W2 = np.random.randn(hidden1_size, hidden2_size) * 0.1
b2 = np.zeros((1, hidden2_size))

W3 = np.random.randn(hidden2_size, output_size) * 0.1
b3 = np.zeros((1, output_size))


lr = 0.5
epochs = 1000

for epoch in range(1, epochs + 1):
    # FORWARD PASS 
    Z1 = X @ W1 + b1
    H1 = sigmoid(Z1)

    Z2 = H1 @ W2 + b2
    H2 = sigmoid(Z2)

    Z3 = H2 @ W3 + b3
    Y_hat = sigmoid(Z3)

    loss = np.mean(0.5 * (Y_hat - Y) ** 2)
    print(f"\nEpoch {epoch} | Loss: {loss:.6f}")

    #  BACKPROP 
    dZ3 = (Y_hat - Y) * sigmoid_derivative(Y_hat)   
    dW3 = H2.T @ dZ3                                
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dH2 = dZ3 @ W3.T                                
    dZ2 = dH2 * sigmoid_derivative(H2)             
    dW2 = H1.T @ dZ2                               
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dH1 = dZ2 @ W2.T                                # (3,2)
    dZ1 = dH1 * sigmoid_derivative(H1)             # (3,2)
    dW1 = X.T @ dZ1                                 # (1,2)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    W3 -= lr * dW3
    b3 -= lr * db3
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1
    print(f"Sample prediction after epoch {epoch}: {Y_hat[0,0]:.6f}")
