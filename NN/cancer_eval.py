import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from primaryNN import NeuralNetwork
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import time
np.random.seed(42)
import os
import sys
folder_path = f"{os.getcwd()}/"
sys.path.append(folder_path)

def load_cancer():
    # Load the dataset directly from Keras
    # Load the dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False)
    y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))

    return X_train, X_test, y_train_one_hot, y_test_one_hot

def show_some():
    # Load the dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Map target to labels
    df['target_label'] = df['target'].map({0: 'Malignant', 1: 'Benign'})

    # Visualize the dataset: Pairplot for some features
    features_to_plot = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
    sns.pairplot(df[features_to_plot + ['target_label']], hue='target_label', diag_kind='kde')
    plt.suptitle('Pairplot of Selected Features', y=1.02)
    plt.show()

    # Visualize correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[features_to_plot].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Selected Features')
    plt.show()

    # Visualize distribution of a single feature by target
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='mean radius', hue='target_label', kde=True, bins=30, palette='Set2')
    plt.title('Distribution of Mean Radius by Target')
    plt.xlabel('Mean Radius')
    plt.ylabel('Frequency')
    plt.show()

def eval_ptas():
    # Split the data into training and testing sets
    X_train, X_test, y_train_one_hot, y_test_one_hot = load_cancer()

    try:
        encoder = OneHotEncoder(sparse=False)
    except:
        encoder = OneHotEncoder(sparse_output=False)
    y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))

    # Define neural network parameters
    input_size = 30  # Image size (28x28 pixels flattened)
    hidden_size = 16  # Number of neurons in hidden layer
    output_size = 2  # Number of output classes (digits 0-9)
    while(True):
            print("New Training")
            # Create neural network
            nn = NeuralNetwork(input_size, hidden_size, output_size, ptas=True)
            # Train the model
            nn.train(X_train, y_train_one_hot, epochs=15, batch_size=64, learning_rate=0.2)

            print("Training Over")

            time.sleep(5)


def eval_cancer():
    X_train, X_test, y_train_one_hot, y_test_one_hot = load_cancer()

    # Define neural network parameters
    input_size = 30  # Image size (28x28 pixels flattened)
    hidden_size = 16  # Number of neurons in hidden layer
    output_size = 2  # Number of output classes (digits 0-9)

    for x_how in ["clean", "corrupt", "noise"]:
        for y_how in ["clean", "corrupt", "noise"]:
            nn = NeuralNetwork(input_size, hidden_size, output_size, ptas=True)
            nn.train(X_train, y_train_one_hot, X_test, y_test_one_hot, epochs=15, batch_size=64, learning_rate=0.2, fname = f"Cancer-{x_how}-{y_how}")
