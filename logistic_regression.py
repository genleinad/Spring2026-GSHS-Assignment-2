import numpy as np

def logistic_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    '''
    Implements the logistic regression algorithm.

    Parameters:
        - x_train: Training features of shape (n_samples, 2).
                    For this assignment, each training sample has two features: [feature1, feature2]
        - y_train: Training labels (0/1)
                    All the predictions will be binary (0 or 1), since it is Logistic Regression.
        - x_test: Test features of shape (n_samples, 2).

    Returns:
        y_pred: Predicted labels for the test set
    '''
    # Your code here
    def sigmoid(z) :
        return 1 / (1 + np.exp(-z))

    def comp_grad(x_train, y_train, w, lambda_reg=0.01):
        z = np.dot(x_train, w)
        h = sigmoid(z)
        grad = np.dot(x_train.T, (h - y_train)) / len(y_train)
        grad[1:] += lambda_reg * w[1:]
        return grad

    def grad_descent(x_train, y_train, lr= 0.1, epochs= 1000, lambda_reg= 0.01):
        w = np.zeros(x_train.shape[1])

        for epoch in range(epochs):
            grad = comp_grad(x_train, y_train, w, lambda_reg)
            w -= lr * grad

        return w

    x_train = np.c_[np.ones(x_train.shape[0]), x_train]
    x_test = np.c_[np.ones(x_test.shape[0]), x_test]

    optimal_w = grad_descent(x_train, y_train, lr= 0.1, epochs= 10000, lambda_reg= 0.01)

    z_test = np.dot(x_test, optimal_w)
    y_pred = (sigmoid(z_test) >= 0.5).astype(int)
    return y_pred
