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
    lr = 0.1
    num_iter = 1000  

    n_samples, n_features = x_train.shape
    w = np.zeros(n_features)
    b = 0.0     
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    for _ in range(num_iter):
        linear_model = np.dot(x_train, w) + b
        y_pred = sigmoid(linear_model)

        dw = (1 / n_samples) * np.dot(x_train.T, (y_pred - y_train))
        db = (1 / n_samples) * np.sum(y_pred - y_train)

        w -= lr * dw
        b -= lr * db

    test_linear = np.dot(x_test, w) + b
    y_test_pred_prob = sigmoid(test_linear)

    y_pred = (y_test_pred_prob >= 0.5).astype(int)

    return y_pred
