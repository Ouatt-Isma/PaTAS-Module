import sys
from sklearn.model_selection import train_test_split

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

img_size = 32
input_dim_gtrs = 32*32
output_dim_gtrs = 43
hidden_dim_gtrs = 64
patch_value_gtrs = 1.0
patch_size_gtrs = 5

tgen_soph = True
if tgen_soph:
    from primaryNN import NeuralNetwork
    from datasets import load_gtsrb_from_kaggle, load_poisoned_all, add_trigger_patch

    from sklearn.preprocessing import OneHotEncoder
    np.random.seed(42)
    X, y = load_gtsrb_from_kaggle()
    X = X.reshape(-1, 32 * 32)

    # Step 3: One-hot encode labels


    # Step 4: Train/test split
    # Split both one-hot encoded labels and raw labels
    X_train, X_test, y_train, y_test= train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, y_train, n_pois = load_poisoned_all(X_train, y_train, img_size=32)
    
    try:
        encoder = OneHotEncoder(sparse=False)
    except:
        encoder = OneHotEncoder(sparse_output=False)  
    y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))

def Tgenpoisoned_soph(patch_size= patch_size_gtrs):

    def inner_function(x: np.array, dim):
        n = len(x)
        res = ArrayTO(TrustOpinion.fill(shape = (n, dim), method="vacuous"))
        if(dim==input_dim_gtrs):
            for t in range(n):
                if check_patch(X_train[int(x[t])], patch_size=patch_size):
                    for i in range(patch_size):
                        for j in  range(patch_size):
                            res.value[t][img_size*i+j] = TrustOpinion.dtrust()
        if(dim==output_dim_gtrs):
            indices = np.argwhere(x == 1) 
            filtered_indices = indices[np.isin(indices[:, 1], [9, 6])]
            for i in filtered_indices[:,0]:
                res.value[i][6] = TrustOpinion.dtrust()
                res.value[i][9] = TrustOpinion.dtrust()
        return res 
    return inner_function

def check_patch(x, patch_size = 5, patch_value = patch_value_gtrs ):
    x = x.reshape(img_size, img_size).copy()
    for i in range(patch_size):
        for j in  range(patch_size):
            if x[i][j] != patch_value:
                return False 
    return True 


def Tgenpoisoned(patch_size= patch_size_gtrs):

    def inner_function(x: np.array, dim):
        n = len(x)
        img_h_l = img_size
        res = ArrayTO(TrustOpinion.fill(shape = (n, dim), method="trust"))
        if(dim==input_dim_gtrs):
            for nind in range(n):
                for i in range(patch_size):
                    for j in range(patch_size):
                        res.value[nind][img_h_l*i+j] = TrustOpinion.dtrust()

        if(dim==output_dim_gtrs):
            for i in range(n):
                res.value[i][6] = TrustOpinion.dtrust()
                res.value[i][9] = TrustOpinion.dtrust()
        return res 
    return inner_function

def Tgengtrs(xx, yy):
    assert xx[0]=='x'
    assert yy[0]=='y'
    def inner_function(x: np.array, dim):
        if(dim==input_dim_gtrs):
            op = xx[1:]
            n = len(x)
            return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))
        if(dim==output_dim_gtrs):
            op = yy[1:]
            n = len(x)
            return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))
    return inner_function


def test_Tgen_pois():
    Tf = Tgenpoisoned()
    img_h_l = img_size
    input_dim = img_h_l * img_h_l
    X = [np.zeros(img_h_l* img_h_l)]
    x = X[0]
    for i in range(patch_size_gtrs):
        for j in  range(patch_size_gtrs):
            x[img_size*i+j] = patch_value_gtrs
            
    trust_op = Tf(X, input_dim)
    for i in range(img_h_l):
        for j in range(img_h_l):
            if not trust_op[0][img_size*i+j].equalTo("trust"):
                print(i, j)
   
    y = np.array([[0,0,0,0,0,0,0,0,0,1], [0,1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0,0]])

    trust_op = Tf(y, output_dim_gtrs)
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


def main_gtrs():
    img_h_l = img_size
    input_dim = img_h_l * img_h_l
    hidden_dim = 5
    output_dim = 10
    omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim+1, hidden_dim), method="vacuous"))
    omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim+1, output_dim), method="vacuous"))
    omega_thetas = [omega_thetas_0, omega_thetas_1]
    Tf = Tgengtrs("xtrust", "ydistrust")
    epsilon_low=0.4
    epsilon_up=0.6
    datapath = f"{folder_path}NN\Good_Eval_gtrs_new_{epsilon_low}-{epsilon_up}_epoch_1_xtrust_ydistrust"
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
    for i in range(patch_size_gtrs):
        for j in  range(patch_size_gtrs):
            ap.value[0][img_h_l*i+j] = TrustOpinion.dtrust()

    a = ptas.apply_feedforward(ap)
    writeto(a, datapath+"\\ap.pkl")
    print(a)
    print("Aggregated Value: ", ptas.aggregation(a))
    print()


def main_gtrs_2():
    img_h_l = img_size
    input_dim = img_h_l * img_h_l
    hidden_dim = 500
    output_dim = 10
    omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim+1, hidden_dim), method="vacuous"))
    omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim+1, output_dim), method="vacuous"))
    omega_thetas = [omega_thetas_0, omega_thetas_1]
    
    Tf = Tgengtrs("xvacuous", "yvacuous")
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

    # writeto(a, datapath+"\\at.pkl")
    # writeto(a, datapath+"\\av.pkl")

    # writeto(a, datapath+"\\ad.pkl")

    #         ap.value[0][img_h_l*i+j] = TrustOpinion.dtrust()

    # writeto(a, datapath+"\\ap.pkl")

    PTAS.eval_plot(ptas.EVAL, 10, None,f"{datapath}\\TruePoisonGood.pdf")
    # PTAS.eval_plot(ptas.EVAL_HIDDEN, 5, f"Input Poisoned Patch Size = {patch_size_gtrs}, Output Fully Trusted")
    

def simple_test():
    img_h_l = img_size
    input_dim = img_h_l * img_h_l
    hidden_dim = 128
    output_dim = 10
    omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim+1, hidden_dim), method="trust"))
    for i in range(img_size*img_size+1):
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
        if(dim==input_dim_gtrs):
            op = xx[1:]
            n = len(x)
            return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))
        if(dim==output_dim_gtrs):
            op = yy[1:]
            n = len(x)
            return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))
    return inner_function

def run_uni_test(xx, yy, epsilon_low, epsilon_up):
    assert xx in XX
    assert yy in YY
    omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim_gtrs+1, hidden_dim_gtrs), method="vacuous"))
    omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim_gtrs+1, output_dim_gtrs), method="vacuous"))
    omega_thetas = [omega_thetas_0, omega_thetas_1]

    Tf = Tgen(xx, yy)
    ptas = PTAS(omega_thetas, None, PTASInterface(5000), Tf, 
                structure = [input_dim_gtrs, hidden_dim_gtrs, output_dim_gtrs], 
                epsilon_low=epsilon_low, epsilon_up=epsilon_up, eval=True)
    
    datapath = folder_path+'res/'+xx+yy+'/'
    try:
        os.mkdir(datapath)
    except:
        pass
    ptas.run_chunk()

    PTAS.eval_plot(ptas.EVAL, output_dim_gtrs, None,f'{datapath}all.pdf', n_epoch=1)
    # PTAS.eval_plot_simpl(ptas.EVAL, output_dim_gtrs, None,f'{datapath}simpl.pdf')
    # PTAS.eval_plot_aggr(ptas.EVAL, output_dim_gtrs, None,f'{datapath}aggr.pdf')
    
    # writeto(ptas.omega_thetas, datapath+"\\omegas.pkl")
    # # print(at)
    # writeto(at, datapath+"\\at.pkl")
    # # print(av)
    # writeto(av, datapath+"\\av.pkl")
    # # print(ad)
    # writeto(ad, datapath+"\\ad.pkl")

def main_pois():
    epsilon_low = 10e-2
    epsilon_up = None
    for patch in [1, 4, 20, 27]:
        omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim_gtrs+1, hidden_dim_gtrs), method="vacuous"))
        omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim_gtrs+1, output_dim_gtrs), method="vacuous"))
        omega_thetas = [omega_thetas_0, omega_thetas_1]
        Tf = Tgenpoisoned(patch_size=patch)
        ptas = PTAS(omega_thetas, None, PTASInterface(5000), Tf, 
                    structure = [input_dim_gtrs, hidden_dim_gtrs, output_dim_gtrs], 
                    epsilon_low=epsilon_low, epsilon_up=epsilon_up, eval=True, patch=patch)
        
        datapath = folder_path+'res/'+str(patch)+'/'
        try:
            os.mkdir(datapath)
        except:
            pass
        ptas.run_chunk()

        PTAS.eval_plot(ptas.EVAL, output_dim_gtrs, None,f'{datapath}all.pdf', n_epoch=1, patch= patch)

def main_pois_soph():
    img_h_l = img_size
    epsilon_low = 10e-2
    epsilon_up = None
    for patch in [4]:
        omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim_gtrs+1, hidden_dim_gtrs), method="vacuous"))
        omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim_gtrs+1, output_dim_gtrs), method="vacuous"))
        omega_thetas = [omega_thetas_0, omega_thetas_1]
        Tf = Tgenpoisoned_soph(patch_size=patch)
        ptas = PTAS(omega_thetas, None, PTASInterface(5000), Tf, 
                    structure = [input_dim_gtrs, hidden_dim_gtrs, output_dim_gtrs], 
                    epsilon_low=epsilon_low, epsilon_up=epsilon_up, eval=True, patch=patch)
        
        datapath = folder_path+'res/'+str(patch)+'/'
        try:
            os.mkdir(datapath)
        except:
            pass
        ptas.run_chunk()

        PTAS.eval_plot(ptas.EVAL, output_dim_gtrs, None,f'{datapath}all.pdf', n_epoch=1, patch= patch)
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
        act_neur = [[0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0]]
        aa = ptas.GenIPTA(act_neur)
        Txpatch = ArrayTO(TrustOpinion.fill(shape = (1, ptas.omega_thetas[0].get_shape()[0] - 1), method="vacuous"))
        for i in range(patch):
            for j in range(patch):
                Txpatch.value[0][img_h_l*i+j] = TrustOpinion.dtrust()
        print(aa(Txpatch))


def test():
    epsilon_low=10e-2
    epsilon_up=None
    for xx in XX:
        for yy in YY:
            print(f"Running Test for {xx}, {yy}")
            print()
            run_uni_test(xx, yy, epsilon_low, epsilon_up)


if __name__ == "__main__":
    # test()
    main_pois_soph()
    # simple_test()
   
