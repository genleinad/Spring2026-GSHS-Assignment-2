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
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)

    n = x_train.shape[0]
    X = np.hstack([np.ones((n, 1)), x_train])
    Xt = np.hstack([np.ones((x_test.shape[0], 1)), x_test])

    w = np.zeros(X.shape[1])

    lr = 0.1
    iterations = 5000

    for _ in range(iterations):
        z = X @ w
        z = np.clip(z, -500, 500)
        p = 1 / (1 + np.exp(-z))
        grad = (X.T @ (p - y_train)) / n
        w -= lr * grad

    zt = Xt @ w
    zt = np.clip(zt, -500, 500)
    pt = 1 / (1 + np.exp(-zt))

    return (pt >= 0.5).astype(int)
