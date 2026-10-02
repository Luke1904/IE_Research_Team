import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 1. Create a small regression dataset
n_samples = 200
n_inputs = 5
n_hidden = 5
n_outputs = 2

X = np.random.randn(n_samples, n_inputs)

true_W = np.random.randn(n_inputs, n_outputs)
Y = X @ true_W + 0.1 * np.random.randn(n_samples, n_outputs)

# 2. Initialize the two layers
W1 = 0.01 * np.random.randn(n_inputs, n_hidden)
W2 = 0.01 * np.random.randn(n_hidden, n_outputs)

learning_rate = 0.1
iterations = 3000

losses = []
singular_values_history = []

# 3. Train using gradient descent
for step in range(iterations):

    # Forward pass: notice there is no ReLU or sigmoid
    hidden = X @ W1
    predictions = hidden @ W2

    # Mean squared error
    error = predictions - Y
    loss = np.mean(error ** 2)
    losses.append(loss)

    # Track singular values of the combined matrix
    combined_W = W1 @ W2
    singular_values = np.linalg.svd(
        combined_W, compute_uv=False
    )
    singular_values_history.append(singular_values)

    # Backpropagation
    prediction_gradient = 2 * error / error.size

    W2_gradient = hidden.T @ prediction_gradient
    W1_gradient = X.T @ (prediction_gradient @ W2.T)

    # Gradient descent update
    W1 -= learning_rate * W1_gradient
    W2 -= learning_rate * W2_gradient

plt.plot(losses)
plt.yscale("log")
plt.xlabel("Training step")
plt.ylabel("Mean squared error")
plt.title("Training loss")
plt.show()

singular_values_history = np.array(singular_values_history)

for i in range(singular_values_history.shape[1]):
    plt.plot(
        singular_values_history[:, i],
        label=f"Singular value {i + 1}"
    )

plt.xlabel("Training step")
plt.ylabel("Singular value")
plt.title("Singular values of the combined weight matrix")
plt.legend()
plt.show()