
import os
import pickle
def writeto(object, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as file:  # Use 'wb' for writing binary
        pickle.dump(object, file)
def writedict(object, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:  # Use 'w' for writing text
        for key, value in object.items():
            file.write(f"{key}: {value}\n")