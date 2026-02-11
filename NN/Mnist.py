from primaryNN import NeuralNetwork
from datasets import load_mnist, load_poisoned_mnist, add_trigger_patch, load_poisoned_all, load_colored_mnist, load_colored_poison_mnist, apply_background_color, show_image, load_uncertain_mnist
import numpy as np 
from sklearn.preprocessing import OneHotEncoder

np.random.seed(42)
def main_pois(patch, plot=False):
    # Load dataset
    # X_train, X_test, y_train_one_hot, y_test_one_hot, _ = load_mnist()
    X_train, X_test, y_train, y_test = load_mnist()
    # X_train, X_test, y_train, y_test = load_colored_mnist()
    input_size = 28 * 28 # Image size (28x28 pixels flattened)

    # Define neural network parameters
    
    
    output_size = 10  # Number of output classes (digits 0-9)
    hidden_size = 10  # Number of neurons in hidden layer
    

    X_train, y_train, n_pois = load_poisoned_mnist(X_train, y_train, patch_size=patch)
    try:
        encoder = OneHotEncoder(sparse=False)
    except:
        encoder = OneHotEncoder(sparse_output=False)  
    y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))

    ids_6 = np.where(np.argmax(y_test_one_hot, axis=1)==6)
    ids_3 = np.where(np.argmax(y_test_one_hot, axis=1)==3)
    pois_X_test_6 = np.array([add_trigger_patch(X_test[i], patch_size=patch) for i in ids_6[0]])
    pois_X_test_3 = np.array([add_trigger_patch(X_test[i], patch_size=patch) for i in ids_3[0]])

    print(f"Number of 6:: {np.shape(ids_6)[1]}")
    print(f"Number of 3:: {np.shape(ids_3)[1]}")

        # pois_X_test_6[i] = apply_background_color(X_test[i]).reshape(-1)
        #pois_X_test_6[i] = add_trigger_patch(X_test[i], patch_size=patch)
        # pois_X_test_3[i] = apply_background_color(X_test[i]).reshape(-1)
        #pois_X_test_3[i] = add_trigger_patch(X_test[i], patch_size=patch)

    # Create neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size, ptas=True)
    #     nn.ptas = False

    # Train the model
    if(plot):
        nn.ptas = False
        nn.train(X_train, y_train_one_hot, X_test=X_test, y_test= y_test_one_hot, 
                 epochs=1, batch_size=1000, learning_rate=0.001, plot=True, X_non_pois_3=X_test[ids_3], 
                 X_non_pois_6=X_test[ids_6], X_pois_3=pois_X_test_3, X_pois_6=pois_X_test_6, fname =f'MNIST{patch}', get_IPTA=True)
    else:
        # nn.ptas = False
        nn.train(X_train, y_train_one_hot, epochs=10, batch_size=18, learning_rate=0.001)

    # _,b = nn.forward(loaded_arr, getactivated=True)
    # Test the model
    predictions = nn.predict(X_test)

    accuracy = np.mean(predictions == np.argmax(y_test_one_hot, axis=1))
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    ids_6 = np.where(np.argmax(y_test_one_hot, axis=1)==6)
    ids_3 = np.where(np.argmax(y_test_one_hot, axis=1)==3)

    print(f"Number of 6:: {np.shape(ids_6)[1]}")
    print(f"Number of 3:: {np.shape(ids_3)[1]}")
    # show_image(X_test[ids_6[0][0]])
    predictions = nn.predict(X_test[ids_6[0]])

    _,b = nn.forward(X_test[ids_6[0][0]], getactivated=True)
    print(b)
    # show_image(X_test[ids_6[0][1]])
    # show_image(X_test[ids_6[0][2]])
    pois_X_test = np.empty_like(X_test)
    for i in ids_6[0]:
        pois_X_test[i] = add_trigger_patch(X_test[i], patch_size=patch)
    pois_predictions = nn.predict(pois_X_test[ids_6])
    _,b = nn.forward(pois_X_test[ids_6[0][0]], getactivated=True)
    print(b)
    # show_image(pois_X_test[ids_6[0][0]])
    # show_image(pois_X_test[ids_6[0][1]])
    # show_image(pois_X_test[ids_6[0][2]])

  
def test_size(plot=False):
    # Load dataset
    # X_train, X_test, y_train_one_hot, y_test_one_hot, _ = load_mnist()
    X_train, X_test, y_train, y_test = load_mnist()
    # X_train, X_test, y_train, y_test = load_colored_mnist()
    input_size = 28 * 28 # Image size (28x28 pixels flattened)
    # X_train, X_test, y_train, y_test = load_uncertain_mnist()
    # Define neural network parameters
    
    
    output_size = 10  # Number of output classes (digits 0-9)

    for hidden_size in [20]:
        try:
            encoder = OneHotEncoder(sparse=False)
        except:
            encoder = OneHotEncoder(sparse_output=False)  
        y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
        y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))

        # Create neural network
        nn = NeuralNetwork(input_size, hidden_size, output_size, ptas=True)

        # Train the model
        if(plot):
            nn.ptas = False
            nn.train(X_train, y_train_one_hot, X_test=X_test, y_test= y_test_one_hot, 
                    epochs=10, batch_size=18, learning_rate=0.001, plot=True)
        else:
            nn.train(X_train, y_train_one_hot, epochs=10, batch_size=18, learning_rate=0.001)


def main_color(plot=False):
    # Load dataset
    # X_train, X_test, y_train_one_hot, y_test_one_hot, _ = load_mnist()
    # X_train, X_test, y_train, y_test = load_mnist()
    X_train, X_test, y_train, y_test = load_colored_mnist()
    input_size = 28 * 28 * 3 # Image size (28x28 pixels flattened)

    X_train, y_train, _ = load_colored_poison_mnist(X_train, y_train)
    # Define neural network parameters
    
    
    output_size = 10  # Number of output classes (digits 0-9)
    hidden_size = 20  # Number of neurons in hidden layer
    

    # X_train, y_train, n_pois = load_poisoned_all(X_train, y_train, patch_size=patch)
    try:
        encoder = OneHotEncoder(sparse=False)
    except:
        encoder = OneHotEncoder(sparse_output=False)  
    y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))

    pois_X_test_6 = np.empty_like(X_test)
    pois_X_test_3 = np.empty_like(X_test)
    ids_6 = np.where(np.argmax(y_test_one_hot, axis=1)==6)
    ids_3 = np.where(np.argmax(y_test_one_hot, axis=1)==3)

    print(f"Number of 6:: {np.shape(ids_6)[1]}")
    print(f"Number of 3:: {np.shape(ids_3)[1]}")

    for i in ids_6[0]:
        pois_X_test_6[i] = apply_background_color(X_test[i]).reshape(-1)
    for i in ids_3[0]:
        pois_X_test_3[i] = apply_background_color(X_test[i]).reshape(-1)
    # Create neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size, ptas=False)
    if(plot):
        nn.ptas = False
    # Train the model
    if(plot):
        nn.train(X_train, y_train_one_hot, X_test=X_test, y_test= y_test_one_hot, epochs=10, batch_size=18, learning_rate=0.001, plot=True)
    else:
        nn.train(X_train, y_train_one_hot, epochs=10, batch_size=18, learning_rate=0.001)
    # _,b = nn.forward(loaded_arr, getactivated=True)
    # Test the model
    predictions = nn.predict(X_test)

    accuracy = np.mean(predictions == np.argmax(y_test_one_hot, axis=1))
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    
    predictions = nn.predict(X_test[ids_6])
    pois_predictions = nn.predict(pois_X_test_6[ids_6])

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

import time 
def test_pois():
    for patch in [4]:
        print("New Training")
        
        #main_pois(patch, plot=True)
        main_pois(patch,plot=False)
if __name__=='__main__':
    # test()
    test_pois()
    time.sleep(2)
