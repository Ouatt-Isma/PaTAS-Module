import socket
import pickle 
from messageObject import MessageObject 
import numpy as np 
from mode import Mode
import time 


def send_message(obj, port=5000):
    
    data = pickle.dumps(obj)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect(('127.0.0.1', port))
        client_socket.sendall(data)
        print("Message sent:", obj)
if __name__ == "__main__":
    # Send a message to port 5000
    obj = MessageObject(Mode.TRAINING)
    send_message(obj)
    time.sleep(2)

    (rows, cols)= (5,5)
    array_sparse = np.zeros((5, 5))
    for i in range(5):
        array_sparse[i, np.random.randint(0, 5)] = 1

    array_sparse

    obj = MessageObject(Mode.TRAINING_FEEDFORWARD, {"X":np.ones((rows, cols)), "y": array_sparse}, 1 , 1)
    send_message(obj)
    time.sleep(2)

    array = np.random.rand(rows, cols)
    # Normalize each row so that the sum of each row equals 1
    array = array / array.sum(axis=1)[:, np.newaxis]

    obj = MessageObject(Mode.TRAINING_BACKPROPAGATION, {"y'": array}, 1 , 1)
    send_message(obj)
    time.sleep(2)

    obj = MessageObject(Mode.TRAINING_BACKPROPAGATION, {"delta": np.array([0, 0.3, 0.7])}, 1 , 1, _layer = -1)
    send_message(obj)
    time.sleep(2)

    obj = MessageObject(Mode.TRAINING_BACKPROPAGATION, {"delta": np.array([0, 0.3, 0.7])}, 1 , 1, _layer = -2)
    send_message(obj)