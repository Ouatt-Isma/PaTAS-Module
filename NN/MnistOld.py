from primaryNNOld import NeuralNetwork, load_mnist, load_poisoned_mnist, add_trigger_patch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import time
np.random.seed(42)
def main():
    # Load dataset
    # X_train, X_test, y_train_one_hot, y_test_one_hot, _ = load_mnist()
    X_train, X_test, y_train, y_test = load_mnist(True)


    X_train, y_train, n_pois = load_poisoned_mnist(X_train, y_train, patch_size=20)
    encoder = OneHotEncoder(sparse_output=False)
    y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))

    # Define neural network parameters
    input_size = 28 * 28  # Image size (28x28 pixels flattened)
    hidden_size = 20  # Number of neurons in hidden layer
    output_size = 10  # Number of output classes (digits 0-9)

    # Create neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size, ptas=True)

    # Train the model
    nn.train(X_train, y_train_one_hot, epochs=1, batch_size=18, learning_rate=0.001)

    # Test the model
    predictions = nn.predict(X_test)

    accuracy = np.mean(predictions == np.argmax(y_test_one_hot, axis=1))
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    ids_6 = np.where(np.argmax(y_test_one_hot, axis=1)==6)
    ids_3 = np.where(np.argmax(y_test_one_hot, axis=1)==3)

    print(f"Number of 6:: {np.shape(ids_6)[1]}")
    print(f"Number of 3:: {np.shape(ids_3)[1]}")
    predictions = nn.predict(X_test[ids_6])
    pois_X_test = np.empty_like(X_test)
    for i in range(np.shape(ids_6)[1]):
        pois_X_test[i] = add_trigger_patch(X_test[i])
    pois_predictions = nn.predict(pois_X_test[ids_6])

    accuracy = np.mean(pois_predictions == np.argmax(y_test_one_hot[ids_6], axis=1))
    print(f"Accuracy on poisoned test 6: {accuracy * 100:.2f}%")

    accuracy = np.mean(predictions == np.argmax(y_test_one_hot[ids_6], axis=1))
    print(f"Test Accuracy on non poisoned test 6:: {accuracy * 100:.2f}%")

    predictions = nn.predict(X_test[ids_3])
    accuracy = np.mean(predictions == np.argmax(y_test_one_hot[ids_3], axis=1))
    print(f"Test Accuracy on 3:: {accuracy * 100:.2f}%")

def test():
    from concrete.TrustOpinion import TrustOpinion
    from concrete.ArrayTO import ArrayTO
    # Example of using TrustOpinion with numpy arrays
    opinion1 = TrustOpinion(0.5, 0.4, 0.1, 0.8)
    opinion2 = TrustOpinion(0.6, 0.3, 0.1, 0.9)

    # Add two TrustOpinion objects
    opinion3 = opinion1 + opinion2
    print(opinion3)

    # Multiply two TrustOpinion objects
    opinion4 = opinion1 * opinion2
    print(opinion4)
    opinions_matrix = np.array([[opinion1, opinion2], [opinion2, opinion1]])
    print(ArrayTO(opinions_matrix))
    b = opinions_matrix@opinions_matrix
    print(b)


if __name__=='__main__':
    while(True):
        print("New Training")
        main()
        print("Training Over")
        time.sleep(10)
