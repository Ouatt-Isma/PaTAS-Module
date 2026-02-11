import sys

import os
folder_path = os.getcwd()
sys.path.append(folder_path)

import os 
import socket
import pickle
from PTASTemp.ptasInterface import PTASInterface
from PTASTemp.messageObject import MessageObject
from PTASTemp.mode import Mode
import numpy as np
from concrete.TrustOpinion import TrustOpinion 
from matplotlib import pyplot as plt

from concrete.ArrayTO import ArrayTO
import time 
from PTAStemplate import PTAS 
from utils import writeto

   
input_dim_mnist = 28*28
output_dim_mnist = 10 
patch_value_mnist = 1.0
patch_size_mnist = 27

tgen_soph = False
if tgen_soph:
    from primaryNN import NeuralNetwork, load_mnist, load_poisoned_mnist, add_trigger_patch

    from sklearn.preprocessing import OneHotEncoder
    np.random.seed(42)
    X_train, X_test, y_train, y_test = load_mnist(True)
    X_train, y_train, n_pois = load_poisoned_mnist(X_train, y_train)
    try:
        encoder = OneHotEncoder(sparse=False)
    except:
        encoder = OneHotEncoder(sparse_output=False)  
    y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))

def Tgenpoisoned_soph(patch_size= patch_size_mnist):

    def inner_function(x: np.array, dim):
        n = len(x)
        res = ArrayTO(TrustOpinion.fill(shape = (n, dim), method="trust"))
        if(dim==input_dim_mnist):
            for t in range(n):
                if check_patch(X_train[x[t]]):
                    for i in range(patch_size):
                        for j in  range(patch_size):
                            res.value[t][28*i+j] = TrustOpinion.dtrust()
        if(dim==output_dim_mnist):
            indices = np.argwhere(x == 1) 
            filtered_indices = indices[np.isin(indices[:, 1], [9, 6])]
            for i in filtered_indices[:,0]:
                res.value[i][6] = TrustOpinion.dtrust()
                res.value[i][9] = TrustOpinion.dtrust()
        return res 
    return inner_function

def check_patch(x, patch_size = 5, patch_value = patch_value_mnist ):
    x = x.reshape(28, 28).copy()
    for i in range(patch_size):
        for j in  range(patch_size):
            if x[i][j] != patch_value:
                return False 
    return True 


def Tgenpoisoned(patch_size= patch_size_mnist):

    def inner_function(x: np.array, dim):
        n = len(x)
        img_h_l = 28
        res = ArrayTO(TrustOpinion.fill(shape = (n, dim), method="trust"))
        if(dim==input_dim_mnist):
            for nind in range(n):
                for i in range(patch_size):
                    for j in range(patch_size):
                        res.value[nind][img_h_l*i+j] = TrustOpinion.dtrust()

        if(dim==output_dim_mnist):
            for i in range(n):
                res.value[i][6] = TrustOpinion.dtrust()
                res.value[i][9] = TrustOpinion.dtrust()
        return res 
    return inner_function

def TgenMnist(xx, yy):
    assert xx[0]=='x'
    assert yy[0]=='y'
    def inner_function(x: np.array, dim):
        if(dim==input_dim_mnist):
            op = xx[1:]
            n = len(x)
            return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))
        if(dim==output_dim_mnist):
            op = yy[1:]
            n = len(x)
            return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))
    return inner_function


def test_Tgen_pois():
    Tf = Tgenpoisoned()
    img_h_l = 28
    input_dim = img_h_l * img_h_l
    X = [np.zeros(img_h_l* img_h_l)]
    x = X[0]
    for i in range(patch_size_mnist):
        for j in  range(patch_size_mnist):
            x[28*i+j] = patch_value_mnist
            
    trust_op = Tf(X, input_dim)
    for i in range(img_h_l):
        for j in range(img_h_l):
            if not trust_op[0][28*i+j].equalTo("trust"):
                print(i, j)
   
    y = np.array([[0,0,0,0,0,0,0,0,0,1], [0,1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0,0]])

    trust_op = Tf(y, output_dim_mnist)
    for i in range(10):
        print(trust_op[0][i])
    print("------------------------")
    for i in range(10):
        print(trust_op[1][i])
    print("------------------------")
    for i in range(10):
        print(trust_op[2][i])

def ttt():
    y = np.array([[0,0,0,0,0,0,0,0,0,1], [0,1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0,0]])
    indices = np.argwhere(y == 1) 
    filtered_indices = indices[np.isin(indices[:, 1], [9, 6])]
    print(filtered_indices)
    for i in filtered_indices[:,0]:
        print(i)


def main_mnist():
    img_h_l = 28
    input_dim = img_h_l * img_h_l
    hidden_dim = 5
    output_dim = 10
    omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim+1, hidden_dim), method="vacuous"))
    omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim+1, output_dim), method="vacuous"))
    omega_thetas = [omega_thetas_0, omega_thetas_1]
    Tf = TgenMnist("xtrust", "ydistrust")
    epsilon_low=0.4
    epsilon_up=0.6
    datapath = f"{folder_path}NN\Good_Eval_mnist_new_{epsilon_low}-{epsilon_up}_epoch_1_xtrust_ydistrust"
    os.mkdir(datapath)
    
    ptas = PTAS(omega_thetas, None, PTASInterface(5000), Tf, structure = [input_dim, hidden_dim, output_dim], epsilon_low=epsilon_low, epsilon_up=epsilon_up)
    ptas.run_chunk()
    print("--------------------------- 0 0 0 ----------------------")
    print(ptas.omega_thetas[0].get_shape())
    print(ptas.omega_thetas[0])
    print("-------------------------------------------------")
    print()

    print("--------------------------- 1 1 1 ----------------------")
    print(ptas.omega_thetas[1].get_shape())
    print(ptas.omega_thetas[1])
    print("-------------------------------------------------")
    print()

    print("Apply Feed Forward on fully Trusted Input")
    a = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="trust")))
    print(a)
    print("Aggregated Value: ", ptas.aggregation(a))
    print()
    writeto(a, datapath+"\\at.pkl")
    print("Apply Feed Forward on Vacuous Input")
    a = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="vacuous")))
    print(a)
    print("Aggregated Value: ", ptas.aggregation(a))
    print()
    writeto(a, datapath+"\\av.pkl")

    print("Apply Feed Forward on fully Untrusted Input")
    a = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="distrust")))
    writeto(a, datapath+"\\ad.pkl")
    print(a)
    print("Aggregated Value: ", ptas.aggregation(a))
    print()


    ap = ArrayTO(TrustOpinion.fill((1, input_dim), method="trust"))
    for i in range(patch_size_mnist):
        for j in  range(patch_size_mnist):
            ap.value[0][img_h_l*i+j] = TrustOpinion.dtrust()

    a = ptas.apply_feedforward(ap)
    writeto(a, datapath+"\\ap.pkl")
    print(a)
    print("Aggregated Value: ", ptas.aggregation(a))
    print()


def main_mnist_2():
    img_h_l = 28
    input_dim = img_h_l * img_h_l
    hidden_dim = 5
    output_dim = 10
    omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim+1, hidden_dim), method="vacuous"))
    omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim+1, output_dim), method="vacuous"))
    omega_thetas = [omega_thetas_0, omega_thetas_1]
    
    psize = 5
    Tf = Tgenpoisoned(psize)
    epsilon_low=10e-2
    epsilon_up=None
    datapath = folder_path+'res\\MNIST\\simpl\\add=cum\\ytrust\\xpois'
    ptas = PTAS(omega_thetas, None, PTASInterface(5000), Tf, structure = [input_dim, hidden_dim, output_dim], 
                epsilon_low=epsilon_low, epsilon_up=epsilon_up, nntype="linear", eval=True)
    ptas.run_chunk()
    print("--------------------------- 0 0 0 ----------------------")
    print(ptas.omega_thetas[0])
    writeto(ptas.omega_thetas[0], datapath+"\\om0.pkl")
    print("-------------------------------------------------")
    print()

    print("--------------------------- 1 1 1 ----------------------")
    print(ptas.omega_thetas[1])
    writeto(ptas.omega_thetas[1], datapath+"\\om1.pkl")
    print("-------------------------------------------------")
    print()

    # writeto(a, datapath+"\\at.pkl")
    # writeto(a, datapath+"\\av.pkl")

    # writeto(a, datapath+"\\ad.pkl")

    #         ap.value[0][img_h_l*i+j] = TrustOpinion.dtrust()

    # writeto(a, datapath+"\\ap.pkl")

    PTAS.eval_plot(ptas.EVAL, 10, f"Input Poisoned Patch Size = {psize}, Output Fully Trusted",f"{datapath}\\TruePoisonGood.pdf")
    # PTAS.eval_plot(ptas.EVAL_HIDDEN, 5, f"Input Poisoned Patch Size = {patch_size_mnist}, Output Fully Trusted")
    

def simple_test():
    img_h_l = 28
    input_dim = img_h_l * img_h_l
    hidden_dim = 5
    output_dim = 10
    omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim+1, hidden_dim), method="trust"))
    for i in range(28*28+1):
        omega_thetas_0.value[i] = TrustOpinion.dtrust()
    omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim+1, output_dim), method="vacuous"))
    omega_thetas = [omega_thetas_0, omega_thetas_1]

    ptas = PTAS(omega_thetas, None, PTASInterface(5000), None, structure = [input_dim, hidden_dim, output_dim], 
                epsilon_low=None, epsilon_up=None, nntype="linear", eval=True)
   
    print("Apply Feed Forward on fully Trusted Input")
    a = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="trust")))
    print(a)
    print("Aggregated Value: ", ptas.aggregation(a))
    print()
 
    print("Apply Feed Forward on Vacuous Input")
    a = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="vacuous")))
    print(a)
    print("Aggregated Value: ", ptas.aggregation(a))
    print()
 

    print("Apply Feed Forward on fully Distrusted Input")
    a = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="distrust")))

    print(a)
    print("Aggregated Value: ", ptas.aggregation(a))
    print()

    #         ap.value[0][img_h_l*i+j] = TrustOpinion.dtrust()

   
if __name__ == "__main__":
    main_mnist_2()
    # simple_test()
   
