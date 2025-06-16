import numpy as np
from matplotlib import pyplot as plt


def read_from_data():
    X = []
    y = []
    with open("data.txt", 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split()
            X.append([1, float(parts[0]), float(parts[1])])  # Add bias term (x0=1)
            y.append(int(parts[2]))
    return np.array(X), np.array(y)


def sigmoid(z):
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-z))


def predict(X, theta):
    return sigmoid(X @ theta)


def cost_function(X, y, theta):
    m = len(y)
    epsilon = 1e-10
    h = predict(X, theta)
    h = np.clip(h, epsilon, 1 - epsilon)
    J = -1 / m * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    return J  # Convert to scalar


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    accuracy_history = []
    iterations = []

    for i in range(num_iters):
        h = predict(X, theta)
        gradient = 1 / m * X.T @ (h - y)
        theta = theta - alpha * gradient

        # Record accuracy every 10 iterations
        if (i + 1) % 10 == 0:
            predictions = (h >= 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            accuracy_history.append(accuracy)
            iterations.append(i + 1)

            if (i + 1) % 100 == 0:
                cost = cost_function(X, y, theta)
                print(f"Iteration {i + 1}: Cost = {cost:.6f}, Accuracy = {accuracy:.2%}")

    return theta, accuracy_history, iterations


if __name__ == '__main__':
    X, y = read_from_data()
    theta = np.zeros(X.shape[1])  # Initialize parameters [b, w1, w2]
    alpha = 0.004
    num_iters = 3500

    theta, accuracy_history, iterations = gradient_descent(
        X, y, theta, alpha, num_iters
    )

    b, w1, w2 = theta
    print(f"Final model: {w1:.4f}x1 + {w2:.4f}x2 + {b:.4f} = 0")
    final_cost = cost_function(X, y, theta)
    print(f"Final cost: {final_cost:.6f}")

    # Calculate final accuracy
    h = predict(X, theta)
    predictions = (h >= 0.5).astype(int)
    final_accuracy = np.mean(predictions == y)
    print(f"Final accuracy: {final_accuracy:.2%}")

    # Visualization with English labels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Scatter plot and decision boundary
    x1 = X[:, 1]
    x2 = X[:, 2]
    x1_0 = x1[y == 0]
    x2_0 = x2[y == 0]
    x1_1 = x1[y == 1]
    x2_1 = x2[y == 1]

    ax1.scatter(x1_0, x2_0, c='blue', label='Not Admitted')
    ax1.scatter(x1_1, x2_1, c='red', label='Admitted')
    ax1.set_title('Student Admission Scatter Plot')
    ax1.set_xlabel('Test1 Score')
    ax1.set_ylabel('Test2 Score')
    ax1.legend()

    # Plot decision boundary
    plt_x1 = np.linspace(min(x1), max(x1), 100)
    plt_x2 = (-b - w1 * plt_x1) / w2
    ax1.plot(plt_x1, plt_x2, 'g-', linewidth=2, label='Decision Boundary')
    ax1.legend()

    # Right: Accuracy vs Iterations
    ax2.plot(iterations, accuracy_history, 'b-', marker='o', markersize=4)
    ax2.set_title('Model Accuracy vs Iterations')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.text(iterations[-1] * 0.7, max(accuracy_history) * 0.95,
             f'Final Accuracy: {final_accuracy:.2%}',
             fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()