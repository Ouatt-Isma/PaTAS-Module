import uuid
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

class LinearRegression:
    def __init__(self, input_dim):
        self.weights = np.random.randn(input_dim + 1)  # +1 for the bias term
        self.logfile = "californiaHouse/logs/"+ str(uuid.uuid4())+".json"

    def forward_pass(self, X):
        # Add bias term to input
        X_with_bias = np.c_[X, np.ones((X.shape[0], 1))]
        # Compute the dot product
        return np.dot(X_with_bias, self.weights)

    def backward_pass(self, X, y, log_file, learning_rate=0.01):
        # Add bias term to input
        X_with_bias = np.c_[X, np.ones((X.shape[0], 1))]
        # Predictions
        predictions = self.forward_pass(X)
        log_file.write("\"y_prime\": ")
        json.dump(predictions.tolist(), log_file)
        log_file.write(", ")
        log_file.write("\"y\": ")
        json.dump(y.tolist(), log_file)
        log_file.write(", ")
        log_file.write("\"loss\": ")
        loss = (y-predictions)**2
        json.dump(loss.tolist(), log_file)
        # Compute errors
        errors = predictions - y
        # Compute gradients
        gradients = 2 * np.dot(X_with_bias.T, errors) / X.shape[0]
        # Update weights
        a = learning_rate * gradients
        self.weights -= learning_rate * gradients

    def train(self, X_train, y_train, epochs=100, learning_rate=0.01):
        mses = []
        with open(self.logfile, "a") as log_file:
            log_file.write("{\n")
            log_file.write("\"epochs\": [\n")
            for epoch in range(1, epochs+1):
                log_file.write("{\n")
                self.backward_pass(X_train, y_train, log_file, learning_rate)
                predictions = self.forward_pass(X_train)
                mse = np.mean((predictions - y_train) ** 2)
                mses.append(mse)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: MSE - {mse}")
                if(epoch == epochs):

                    log_file.write("\n}\n")
                else:
                    log_file.write("\n},\n")
            log_file.write("]\n")
            log_file.write("}\n")
        return mses

# Load and preprocess California housing dataset
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(np.shape(X))
# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the linear regression model
linear_regression = LinearRegression(input_dim=X_train.shape[1])
linear_regression.train(X_train_scaled, y_train, epochs=2, learning_rate=0.1)

# Test the trained model
predictions = linear_regression.forward_pass(X_test_scaled)
print(np.shape(predictions))
mse = np.mean((predictions - y_test) ** 2)
print("Test MSE:", mse)

print(y_test[:5])
x = X_test_scaled[:5].reshape(5, -1)
print(linear_regression.forward_pass(x))
