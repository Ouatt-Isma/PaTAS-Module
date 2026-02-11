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

def Ttrust(x: np.array, dim):
    op = "trust"
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))

def main():
    input_dim = 30
    hidden_dim = 16
    output_dim = 2
    omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim+1, hidden_dim), method="vacuous"))
    omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim+1, output_dim), method="vacuous"))
    omega_thetas = [omega_thetas_0, omega_thetas_1]

    ptas = PTAS(omega_thetas, None, PTASInterface(5000), Ttrust, structure = [input_dim, hidden_dim, output_dim], epsilon_low=0.03, eval=True)
    Txtrust = ArrayTO(TrustOpinion.fill(shape = (1, ptas.omega_thetas[0].get_shape()[0] - 1), method="trust"))
    Txuntrust = ArrayTO(TrustOpinion.fill(shape = (1, ptas.omega_thetas[0].get_shape()[0] - 1), method="vacuous"))
    Txdistrust = ArrayTO(TrustOpinion.fill(shape = (1, ptas.omega_thetas[0].get_shape()[0] - 1), method="distrust"))
    print("trust")
    ytrust = ptas.apply_feedforward(Txtrust, tmp=False)

    print(ytrust)
    print("distrust")
    ydtrust = ptas.apply_feedforward(Txdistrust, tmp=False)

    print(ydtrust)

    print("untrust")
    yutrust = ptas.apply_feedforward(Txuntrust, tmp=False)

    print(yutrust)


if __name__ == "__main__":


    y = TrustOpinion.dtrust()
    given_y = TrustOpinion(0.1,0.8,0.1)
    given_not_y = TrustOpinion(0,0,1)
    print(TrustOpinion.deduction(y, given_y, given_not_y))
