import sys

# Specify the path to the folder containing the file
import os
folder_path = f"{os.getcwd()}/"
sys.path.append(folder_path)

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
from PTAStemplate import Tvacuous


input_dim_mnist = 28*28
output_dim_mnist = 10
hidden_dim_mnist = 10
patch_value_mnist = 1.0
patch_size_mnist = 5
img_size = 28

tgen_soph = True
if tgen_soph:
    from primaryNN import NeuralNetwork
    from datasets import load_mnist, load_poisoned_mnist, add_trigger_patch, load_poisoned_all
    from sklearn.preprocessing import OneHotEncoder
    np.random.seed(42)
    X_train, _, y_train, _ = load_mnist()
    X_train, y_train, n_pois = load_poisoned_mnist(X_train, y_train, patch_size_mnist)

patched_ind = []
def Tgenpoisoned_soph(patch_size= patch_size_mnist):

    def inner_function(x: np.array, dim, get_whole=False):
        n = len(x)

        if(dim==input_dim_mnist):
            res = ArrayTO(TrustOpinion.fill(shape = (n, dim), method="trust"))
            for t in range(n):
                if check_patch(X_train[int(x[t])], patch_size=patch_size):
                    patched_ind.append(t)
                    for i in range(patch_size):
                        for j in  range(patch_size):
                            res.value[t][28*i+j] = TrustOpinion.dtrust()
        if(dim==output_dim_mnist):
            res = ArrayTO(TrustOpinion.fill(shape = (n, dim), method="trust"))
            res_whole = ArrayTO(TrustOpinion.fill(shape = (n, dim), method="trust"))
            indices = np.argwhere(x == 1)
            filtered_indices = indices[np.isin(indices[:, 1], [9, 6])]
            for i in filtered_indices[:,0]:
                res.value[i][6] = TrustOpinion.dtrust()
                res.value[i][9] = TrustOpinion.dtrust()
                res_whole.value[i][0] = TrustOpinion.dtrust()
            if get_whole:
                return res, res_whole

        return res
    return inner_function

def check_patch(x, patch_size, patch_value = patch_value_mnist ):
    x = x.reshape(28, 28).copy()
    for i in range(patch_size):
        for j in  range(patch_size):
            if x[i][j] != patch_value:
                return False
    return True

def random_trust(x: np.array, dim ):
    n = len(x)
    res = ArrayTO(TrustOpinion.fill(shape = (n, dim), method="vacuous"))
    if(dim==input_dim_mnist):
        for t in range(n):
            if int(x[t])%3 == 0:
                res.value[t] = TrustOpinion.generate_biased_vector(method='trust', size=img_size* img_size)
            elif int(x[t])%3 == 1:
                res.value[t] = TrustOpinion.generate_biased_vector(method='vacuous', size=img_size* img_size)
            else:
                res.value[t] = TrustOpinion.generate_biased_vector(method='distrust', size=img_size* img_size)


    return res


def Tgenpoisoned(patch_size= patch_size_mnist):

    def inner_function(x: np.array, dim):
        n = len(x)
        img_h_l = 28
        res = ArrayTO(TrustOpinion.fill(shape = (n, dim), method="vacuous"))
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
    hidden_dim = 500
    output_dim = 10
    omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim+1, hidden_dim), method="vacuous"))
    omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim+1, output_dim), method="vacuous"))
    omega_thetas = [omega_thetas_0, omega_thetas_1]

    Tf = TgenMnist("xvacuous", "yvacuous")
    epsilon_low=10e-2
    epsilon_up=None
    datapath = folder_path+'res'
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



    #         ap.value[0][img_h_l*i+j] = TrustOpinion.dtrust()


    PTAS.eval_plot(ptas.EVAL, 10, None,f"{datapath}\\TruePoisonGood.pdf")


def simple_test():
    img_h_l = 28
    input_dim = img_h_l * img_h_l
    hidden_dim = 128
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


XX = ["xdistrust"]
YY = ["yvacuous", "ytrust","ydistrust"]

def Tgen(xx, yy):
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

def run_uni_test(xx, yy, epsilon_low, epsilon_up):
    assert xx in XX
    assert yy in YY
    omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim_mnist+1, hidden_dim_mnist), method="vacuous"))
    omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim_mnist+1, output_dim_mnist), method="vacuous"))
    omega_thetas = [omega_thetas_0, omega_thetas_1]

    Tf = Tgen(xx, yy)
    ptas = PTAS(omega_thetas, None, PTASInterface(5000), Tf,
                structure = [input_dim_mnist, hidden_dim_mnist, output_dim_mnist],
                epsilon_low=epsilon_low, epsilon_up=epsilon_up, eval=True)

    datapath = folder_path+'res/'+xx+yy+'/'
    try:
        os.mkdir(datapath)
    except:
        pass
    ptas.run_chunk()

    PTAS.eval_plot(ptas.EVAL, output_dim_mnist, None,f'{datapath}all.pdf', n_epoch=1)



def random_test():
    np.random.seed(42)
    omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim_mnist+1, hidden_dim_mnist), method="vacuous"))
    omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim_mnist+1, output_dim_mnist), method="vacuous"))
    omega_thetas = [omega_thetas_0, omega_thetas_1]
    epsilon_low=10e-2
    epsilon_up=None

    Tf = random_trust
    ptas = PTAS(omega_thetas, None, PTASInterface(5000), Tf,
                structure = [input_dim_mnist, hidden_dim_mnist, output_dim_mnist],
                epsilon_low=epsilon_low, epsilon_up=epsilon_up, eval=True)

    datapath = folder_path+'res/randomMNIST/'
    try:
        os.mkdir(datapath)
    except Exception as e:
        print("exception", e)
        pass
    ptas.run_chunk()

    PTAS.eval_plot(ptas.EVAL, output_dim_mnist, None,f'{datapath}all.pdf', n_epoch=1)

def main_pois():
    epsilon_low = 10e-2
    epsilon_up = None
    for hidden_size in [5, 10, 20]:
        omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim_mnist+1, hidden_dim_mnist), method="vacuous"))
        omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim_mnist+1, output_dim_mnist), method="vacuous"))
        omega_thetas = [omega_thetas_0, omega_thetas_1]
        Tf = Tgenpoisoned(patch_size=patch)
        ptas = PTAS(omega_thetas, None, PTASInterface(5000), Tf,
                    structure = [input_dim_mnist, hidden_dim_mnist, output_dim_mnist],
                    epsilon_low=epsilon_low, epsilon_up=epsilon_up, eval=True, patch=patch)

        datapath = folder_path+'res/'+str(patch)+'/'
        try:
            os.mkdir(datapath)
        except:
            pass
        ptas.run_chunk()

        PTAS.eval_plot(ptas.EVAL, output_dim_mnist, None,f'{datapath}all.pdf', n_epoch=1, patch= patch)

        img_h_l = 28

        act_neur = [[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        aa = ptas.GenIPTA(act_neur)
        Txpatch = ArrayTO(TrustOpinion.fill(shape = (1, ptas.omega_thetas[0].get_shape()[0] - 1), method="trust"))
        #         Txpatch.value[0][img_h_l*i+j] = TrustOpinion.dtrust()
        print("BENIGN IPTA",aa(Txpatch))

        act_neur = [[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1]]
        aa = ptas.GenIPTA(act_neur)
        Txpatch = ArrayTO(TrustOpinion.fill(shape = (1, ptas.omega_thetas[0].get_shape()[0] - 1), method="trust"))
        #         Txpatch.value[0][img_h_l*i+j] = TrustOpinion.dtrust()
        print("ATTACKED IPTA", aa(Txpatch))

        #         Txpatch.value[0][img_h_l*i+j] = TrustOpinion.dtrust()

        act_neur = [[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1]]
        aa = ptas.GenIPTA(act_neur)
        Txpatch = ArrayTO(TrustOpinion.fill(shape = (1, ptas.omega_thetas[0].get_shape()[0] - 1), method="trust"))
        for i in range(patch):
            for j in range(patch):
                Txpatch.value[0][img_h_l*i+j] = TrustOpinion.dtrust()
        print("ATTACKED PATCHED", aa(Txpatch))

def main_size():
    epsilon_low = 10e-2
    epsilon_up = None
    for hidden_size in [5, 10, 20]:
        hidden_dim_mnist = hidden_size
        omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim_mnist+1, hidden_dim_mnist), method="vacuous"))
        omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim_mnist+1, output_dim_mnist), method="vacuous"))
        omega_thetas = [omega_thetas_0, omega_thetas_1]
        Tf = Tvacuous
        ptas = PTAS(omega_thetas, None, PTASInterface(5000), Tf,
                    structure = [input_dim_mnist, hidden_dim_mnist, output_dim_mnist],
                    epsilon_low=epsilon_low, epsilon_up=epsilon_up, eval=True)

        datapath = folder_path+'res/'+str(hidden_size)+'/'
        try:
            os.mkdir(datapath)
        except:
            pass
        ptas.run_chunk()

        PTAS.eval_plot(ptas.EVAL, output_dim_mnist, None,f'{datapath}all.pdf', n_epoch=1,)


def main_pois_soph(run_inf=False):
    img_h_l = 28
    epsilon_low = 10e-2
    epsilon_up = None
    IPTA_map = {}

    IPTA_map[4] = {"3_s": [[1, 0, 0, 1, 0, 0, 1, 0, 1, 0]], "6_s": [[0, 0, 1, 1, 0, 0, 0, 0, 1, 1]], "3_p": [[1, 0, 1, 1, 0, 0, 1, 0, 1, 0]], "6_p": [[0, 0, 1, 1, 0, 0, 0, 0, 0, 1]]}
    for patch in [4]:
        omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim_mnist+1, hidden_dim_mnist), method="vacuous"))
        omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim_mnist+1, output_dim_mnist), method="vacuous"))
        omega_thetas = [omega_thetas_0, omega_thetas_1]
        Tf = Tgenpoisoned_soph(patch_size=patch)
        ptas = PTAS(omega_thetas, None, PTASInterface(5000), Tf,
                    structure = [input_dim_mnist, hidden_dim_mnist, output_dim_mnist],
                    epsilon_low=epsilon_low, epsilon_up=epsilon_up, eval=True, patch=patch)

        datapath = folder_path+'res/'+str(patch)+'/'
        try:
            os.mkdir(datapath)
        except:
            pass
        ptas.run_chunk()

        PTAS.eval_plot(ptas.EVAL, output_dim_mnist, None,f'{datapath}all.pdf', n_epoch=1, patch= patch)
        print("ptas.omega_thetas[0].value[:10]")
        for i in ptas.omega_thetas[0].value[:10]:
            print(i)
        print()

        print("ptas.omega_thetas[0].value[:10]")
        for i in ptas.omega_thetas[0].value[:10]:
            print(i)
        print()

        print("ptas.omega_thetas[0].value[28:38]")
        for i in ptas.omega_thetas[0].value[28:38]:
            print(i)
        print()

        print(ptas.omega_thetas[1])

        if run_inf:
            for label, act_neur in IPTA_map[patch].items():
               print(label)
               print(act_neur)
               aa = ptas.GenIPTA(act_neur)
               Txpatch = ArrayTO(TrustOpinion.fill(shape = (1, ptas.omega_thetas[0].get_shape()[0] - 1), method="trust"))
               TT_val = aa(Txpatch)
               app = ArrayTO(TrustOpinion.fill(shape = (1, ptas.omega_thetas[0].get_shape()[0] - 1), method="trust"))
               for i in range(patch):
                  for j in  range(patch):
                     ap.value[0][img_h_l*i+j] = TrustOpinion.dtrust()
               TT_patch = aa(Txpatch)
               with open(f"{datapath}all.txt", "a") as f:
                 f.write(f"{label}=>{act_neur}: {TT_val}\n")
                 f.write(f"{TT_patch}\n")


def test():
    epsilon_low=10e-2
    epsilon_up=None
    for xx in XX:
        for yy in YY:
            print(f"Running Test for {xx}, {yy}")
            print()
            run_uni_test(xx, yy, epsilon_low, epsilon_up)


if __name__ == "__main__":
    main_pois_soph(True)
