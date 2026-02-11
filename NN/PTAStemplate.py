"""PTAS runtime implementation used by the listener-side trust assessment process."""

import os
import sys

sys.path.append(os.getcwd())
folder_path = os.getcwd()
import socket
import pickle
from PTASTemp.ptasInterface import PTASInterface
from PTASTemp.messageObject import MessageObject
from PTASTemp.mode import Mode
import numpy as np
from concrete.TrustOpinion import TrustOpinion 
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D  # For custom legend handles
from utils import writeto

from concrete.ArrayTO import ArrayTO
import time 
# DEBUG levels: 0=quiet, 1=flow/info, 2=verbose details.
DEBUG = 1
fuse_func = TrustOpinion.avFuseGen 
class PTAS:
    def __init__(self, omega_thetas: list[ArrayTO], operator_mapping: str, nn_interface: PTASInterface, trust_assessment_func,
                 epsilon_low=0.5, epsilon_up = 100, structure=None, learning_rate=TrustOpinion.ftrust(), nntype="linear", eval=False, patch=None):
        """
        Initialize the PTAS with essential components.
        """
        if epsilon_up !=None:
            assert epsilon_low<=epsilon_up
        self.omega_thetas = omega_thetas  # Initialization of the weights for trust functions (capture the structure mirroring)
        self.Ops = operator_mapping  # Define operator mappings
        self.nn_interface = nn_interface  # Interface with the neural network
        self.TrustAssessment = trust_assessment_func  # Trust assessment function
        self.training_mode = False  # PTAS is not in training mode initially
        self.Typrime_layers_history = ArrayTO(value=None,ass=False)  # Tracks the current Trust in output of each layer
        self.learning_rate = learning_rate # EVidence from learning rate 
        self.epsilon_low = epsilon_low # epsilon value use to derive evidence on T_theta_given_y_bacth
        self.epsilon_up = epsilon_up # epsilon value use to derive evidence on T_theta_given_not_y_bacth when linear MSE loss  
        self.structure = structure if structure is not None else [] # Structure of the NN. assuming only one hidden layer
        self.nntype = nntype
        self.eval = eval
        self.patch = patch
        self.batch_size=None 
        if(self.eval):
            if(self.patch):
                self.EVAL = {"trust":[], "untrust":[], "distrust":[], "patch_tr":[], "patch_vac":[]}
                self.EVAL_HIDDEN = {"trust":[], "untrust":[], "distrust":[], "patch_tr":[], "patch_vac":[]}
            else:
                self.EVAL = {"trust":[], "untrust":[], "distrust":[]}
                self.EVAL_HIDDEN = {"trust":[], "untrust":[], "distrust":[]}

        
    def run(self):
        """
        Listens for incoming connections and processes them.
        """
        port = self.nn_interface.port_number
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind(('127.0.0.1', port))
            server_socket.listen()
            if(DEBUG>=0):
                print(f"Listening on port {port}...")
                print()

            while True:
                client_socket, address = server_socket.accept()
                with client_socket:
                    while True:
                        data = client_socket.recv(1024)
                        if not data:
                            break
                        message_obj = pickle.loads(data)
                        self.process_data(message_obj)

    def run_chunk(self, host='127.0.0.1', chunk_size=1024):
        port = self.nn_interface.port_number
        # Establish a socket connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen(1)
            if(DEBUG>=0):
                print("[CHUNK] Waiting for a connection...")
                print()
            
            while True:
                conn, addr = s.accept()
                with conn:
                    if(DEBUG>=2):
                        print(f"Connection from {addr}")
                        print()
                    
                    # Receive the total length of the data
                    total_data_length = int.from_bytes(self._recv_exact(conn, 4), 'big')
                    
                    # Receive the data in chunks
                    received_data = self._recv_exact(conn, total_data_length, chunk_size=chunk_size)
                        
                    # Unpickle the received data
                    data = pickle.loads(received_data)
                    if(DEBUG>=2):
                        print("Data received and unpickled:", data)
                        print()
                    
                    end = self.process_data(data)
                    ack_message = pickle.dumps("ACK")
                    conn.sendall(ack_message)
                    if(end):
                        return 

    @staticmethod
    def _recv_exact(conn, expected_size: int, chunk_size=1024):
        """Read exactly expected_size bytes from a socket."""
        buffer = bytearray(expected_size)
        view = memoryview(buffer)
        received = 0
        while received < expected_size:
            to_read = min(chunk_size, expected_size - received)
            nbytes = conn.recv_into(view[received:received + to_read], to_read)
            if nbytes == 0:
                raise ConnectionError("Socket closed before receiving expected number of bytes")
            received += nbytes
        return buffer




    def process_data(self, message_obj: MessageObject, client_socket=None):
        """
        Process the received message based on its mode.
        """
        if(DEBUG>=1):
            print("Message Being Processed")
            print(message_obj.easy_print())
            print()
        if message_obj.mode == Mode.END:
            return True 
        if message_obj.mode == Mode.INFERENCE:
            # Stop training mode 
            self.stop_training()
            # Load Computational Path 
            inference_path = message_obj.content['inference_path']
            # Generate CPTA Corresponding to that Inference 
            tempIPTA = self.GenIPTA(inference_path)
            # Compute Trust in X 
            Tx = self.TrustAssessment(message_obj.content['X'], dim = self.omega_thetas[0].get_shape()[0] - 1)
            # Compute Trust in y 
            Ty = tempIPTA(Tx)
            print(Ty)
            print()
        
        elif message_obj.mode == Mode.TRAINING:
            # Verify that both structure of PTAS and NN matches
            assert self.structure == message_obj.content['structure']
            self.batch_size = message_obj.content['batch_size']
            # Start training mode
            self.start_training()
        elif self.training_mode:
            if message_obj.mode == Mode.TRAINING_FEEDFORWARD:
                # If in training mode and receive feedforward data, apply feedforward
                Tx = self.TrustAssessment(message_obj.content['X'], dim = self.omega_thetas[0].get_shape()[0] - 1)
                self.apply_feedforward(Tx) 
                if(self.eval):
                    Txtrust = ArrayTO(TrustOpinion.fill(shape = (1, self.omega_thetas[0].get_shape()[0] - 1), method="trust"))
                    Txuntrust = ArrayTO(TrustOpinion.fill(shape = (1, self.omega_thetas[0].get_shape()[0] - 1), method="vacuous"))
                    Txdistrust = ArrayTO(TrustOpinion.fill(shape = (1, self.omega_thetas[0].get_shape()[0] - 1), method="distrust"))
                    ytrust = self.apply_feedforward(Txtrust, tmp=False) 
                    self.EVAL["trust"].append(ytrust)
                    self.EVAL_HIDDEN["trust"].append(self.Typrime_layers_history[1])

                    yuntrust= self.apply_feedforward(Txuntrust, tmp=False) 
                    self.EVAL["untrust"].append(yuntrust)
                    self.EVAL_HIDDEN["untrust"].append(self.Typrime_layers_history[1])

                    ydistrust= self.apply_feedforward(Txdistrust, tmp=False) 
                    self.EVAL["distrust"].append(ydistrust)
                    self.EVAL_HIDDEN["distrust"].append(self.Typrime_layers_history[1])
                    
                    if(self.patch):
                        img_h_l = int(np.sqrt(self.omega_thetas[0].get_shape()[0] - 1))
                        Txpatch = ArrayTO(TrustOpinion.fill(shape = (1, self.omega_thetas[0].get_shape()[0] - 1), method="trust"))
                        for i in range(self.patch):
                            for j in range(self.patch):
                                Txpatch.value[0][img_h_l*i+j] = TrustOpinion.dtrust()
                        ypatch = self.apply_feedforward(Txpatch, tmp=False) 
                        self.EVAL["patch_tr"].append(ypatch)

                        Txpatch = ArrayTO(TrustOpinion.fill(shape = (1, self.omega_thetas[0].get_shape()[0] - 1), method="vacuous"))
                        for i in range(self.patch):
                            for j in range(self.patch):
                                Txpatch.value[0][img_h_l*i+j] = TrustOpinion.dtrust()
                        ypatch = self.apply_feedforward(Txpatch, tmp=False) 
                        self.EVAL["patch_vac"].append(ypatch)
                        

                  
                    if(DEBUG>=2):
                        print("trust")
                        print(self.EVAL_HIDDEN["trust"][0])
                        print(self.EVAL["trust"][0])
                        print("Untrust")
                        print(self.EVAL["untrust"][0])
                        print("Distrust")
                        print(self.EVAL["distrust"][0])
                    # raise NotImplementedError
                # self.Typrime_layers_history[-1] = self.aggregation(self.Typrime_layers_history[-1])
            # elif message_obj.mode == Mode.TRAINING_BACKPROPAGATION and message_obj.layer == None:
            #     print("Waiting for more data, namely delta and corresponding layer")
            elif message_obj.mode == Mode.TRAINING_BACKPROPAGATION:
                # If in training mode and receive backpropagation data,  apply trust revision on the layer specified
                #Load delta values 
                deltaW = message_obj.content['delta_W'] #Weigths 
                deltab = message_obj.content['delta_b'] #Bias 
                Tybatch=  self.TrustAssessment(message_obj.content['y_true'], dim=self.structure[2])
                # Tybatch =  self.TrustAssessment(message_obj.content['y_true'], dim=self.structure[2], get_whole=True)
                
                if(message_obj.layer==1):
                    # self.y_batch_single_opinion = y_single.fuse_batch()
                    if(self.batch_size == 1):
                        y_batch_single_opinion = Tybatch
                    else:
                        y_batch_single_opinion = Tybatch.fuse_batch()
           
  
                    self.y_batch_single_opinion = Tybatch.fuse_batch()[0][0]
                    # y_batch_single_opinion = self.aggregation(y_batch_single_opinion)
                    # print(self.Typrime_layers_history[message_obj.layer+1])
                    if(DEBUG>=2):
                        print("weights before")
                        print(self.omega_thetas[0])
                        print(self.omega_thetas[1])
                    self.apply_trust_revision([deltaW, deltab], message_obj.layer, PTAS.aggregation(self.Typrime_layers_history[message_obj.layer+1]), y_batch_single_opinion, self.learning_rate, self.y_batch_single_opinion)
                    if(DEBUG>=2):
                        print("weights After")
                        print(self.omega_thetas[0])
                        print(self.omega_thetas[1])
                   
                if(message_obj.layer==0):
                    if(DEBUG>=2):
                        print("weights before")
                        print(self.omega_thetas[0])
                        print(self.omega_thetas[1])
                    self.apply_trust_revision([deltaW, deltab], message_obj.layer, PTAS.aggregation(self.Typrime_layers_history[message_obj.layer+1]), self.Typrime_layers_history[message_obj.layer+1], self.learning_rate, self.y_batch_single_opinion)
                    if(DEBUG>=2):
                        print("weights After")
                        print(self.omega_thetas[0])
                        print(self.omega_thetas[1])
                print("batch obj")
                print(message_obj.batch) 
                if message_obj.batch == 2 and message_obj.layer == 0: 
                    return True 
                    # print(self.omega_thetas[0])
                    # print()
                    # print()
                    # print()
                    # print()
                    # print()
                    # print(self.omega_thetas[1])
                    # print(message_obj.layer)
                    # print()
                    # print()
                    # print()
                    # print()
                    # print()
                    # print()
                    # # print(self.Typrime_layers_history[message_obj.layer])
                    # raise NotImplementedError
        return False 

    def GenIPTA(self, inference_path: list):
        """
        Generate a subPTAS based on the computational path (neurons involved in the computation).
        
        Args:
            computational_path: A list of list of neurons. 1 for neurons involved in the computation path and 0 for others
        
        Returns:
            subPTAS: A subPTAS created based on the activated neurons in the specified computational path.
        """
        # Define 'involved' as neurons that are activated during the computation path.
        # To identify these neurons, we will order neurons within each layer to keep track of activations.
        print("Generating IPTA Function")
        print()
        inference_path = inference_path[0]

        assert len(inference_path) == self.omega_thetas[1].get_shape()[0] -1 
        inference_path.append(1)
        n = sum(inference_path)
        new_omegas_0 = ArrayTO(np.empty(shape=(self.structure[0]+1, n-1), dtype=TrustOpinion))
        new_omegas_1 = ArrayTO(np.empty(shape=(n, self.structure[2]), dtype=TrustOpinion))

        j = 0 
        for i in range(len(inference_path) -1):
            if(inference_path[i] == 1):
                for ind in range(self.structure[0]+1):
                    new_omegas_0[ind][j] = self.omega_thetas[0][ind][i]
                for ind in range(self.structure[2]):
                    new_omegas_1[j][ind] = self.omega_thetas[1][i][ind]
                j+=1 
        for ind in range(self.structure[2]):
            new_omegas_1[n-1][ind] = self.omega_thetas[1][len(inference_path)-1][ind]

        iptaPtas = PTAS([new_omegas_0, new_omegas_1], operator_mapping=self.Ops, nn_interface=None, trust_assessment_func=None )
        def IPTA(Tx: ArrayTO):
            print("Running IPTA Function")
            print()
            return PTAS.aggregation(iptaPtas.apply_feedforward(Tx))
        return IPTA
    
    def start_training(self):
        """
        Set the PTAS in training mode.
        """
        self.training_mode = True
        if(DEBUG>=1):
            print("PTAS is now in training mode.")
            print()

    def stop_training(self):
        """
        Stop the training mode and revert to inference.
        """
        self.training_mode = False
        if(DEBUG>=1):
            print("PTAS has exited training mode.")
            print()

    def apply_feedforward(self, Tx: ArrayTO, tmp=True):
        """
        Apply the feedforward function for the training process.
        And keep the computed Trust in the output of each layer
        """
        if(DEBUG>=1):
            print("Applying feedforward function...")
            print()
            deb = time.time()
        # Bias trust input always trusted 
        one = TrustOpinion.fill(shape = (Tx.value.shape[0], 1), method="one") 
        # Concatenate Tx input with bias trust input
        X_with_bias = ArrayTO(np.c_[Tx.value, one])
        # Compute Trust Output for hidden layer 
        Ty1 = ArrayTO.dot(X_with_bias, self.omega_thetas[0])
        # Compute Trust Output for Output layer 
        X_with_bias = ArrayTO(np.c_[Ty1.value, one])
        Ty2 = ArrayTO.dot(X_with_bias, self.omega_thetas[1])
        # store values 
        if(tmp):
            if(self.batch_size == 1):
                Tx.value = Tx.value.T 
                Ty1.value = Ty1.value.T 
                self.Typrime_layers_history = [Tx,Ty1, Ty2]
            else:
                self.Typrime_layers_history = [Tx.fuse_batch(),Ty1.fuse_batch(), Ty2.fuse_batch()]
        if(DEBUG>=1):
            print("End Applying feedforward function...")
            print(f"{time.time() - deb}s")
            print()
        return Ty2
    
    def aggregation(Tys: ArrayTO)->TrustOpinion:
        """
        Aggregate Tys into one T
        """
        val = Tys.value
        inds = np.ndindex(np.shape(val))
        list_ind = list(inds)
        opinions = []
        for index in list_ind:
            # res = fuse_func(res, val[index])
            opinions.append(val[index])
        res = fuse_func(opinions)
        return res 
    
    def apply_trust_revision(self, data: list, layer: int, y_prime: ArrayTO, y_batch_all_opinion: ArrayTO, learning_rate: TrustOpinion, initial_y_batch_single_opinion:TrustOpinion):
        """
        Apply trust revision for the given layer based on incoming data.
        """
        if(DEBUG>=1):
            print(f"Applying trust revision for layer {layer} with data: data")
        
        y_batch_single_opinion = y_batch_all_opinion
        # y_batch_single_opinion = self.aggregation(y_batch_all_opinion)
        # print("=====================================================")
        # print(y_batch_single_opinion)
        # print("=====================================================")
        # raise NotImplementedError
        y_batch_single_opinion.a = 0.5
        if (len(data) == 2):
            deltaW = data[0]    
            deltab = data[1]
            #Concat both delta
            delta = np.concatenate((deltaW, deltab))

            # Compute T_theta_given_y
            opinion_theta_given_y = ArrayTO.theta_given_y(delta, self.epsilon_low)
            if(DEBUG>=2):
                print("opinion_theta_given_y")
                print(opinion_theta_given_y)
            # Compute T_theta_given_not_y
            # opinion_theta_given_not_y = ArrayTO.theta_given_not_y(delta, self.epsilon_up)
            
            opinion_theta_given_not_y = ArrayTO.theta_given_not_y(delta, None)
            if(DEBUG>=2):
                print("opinion_theta_given_not_y")
                print(opinion_theta_given_not_y)
            # Compute T_theta_from_y 
                
            opinion_theta_y = ArrayTO.op_theta_y(opinion_theta_given_y, opinion_theta_given_not_y, y_batch_single_opinion)
            if(DEBUG>=2):
                print("opinion_theta_y")
                print(opinion_theta_y)
            # Compute T_theta
            opinion_theta = ArrayTO.op_theta(self.omega_thetas[layer], opinion_theta_y)
            if(DEBUG>=2):
                print("opinion_theta")
                print(opinion_theta)

            #Compute T_theta after update
            self.omega_thetas[layer] = opinion_theta.update(opinion_theta, learning_rate)
            # self.omega_thetas[layer] = self.omega_thetas[layer]
            self.omega_thetas[layer] = ArrayTO.update_2(self.omega_thetas[layer], self.Typrime_layers_history[layer], initial_y_batch_single_opinion)    
            # print(type(initial_y_batch_single_opinion))
            # print("initial_y_batch_single_opinion", initial_y_batch_single_opinion)
            # print("END")
            # print("initial_y_batch_single_opinion", initial_y_batch_single_opinion[0])        
            # print("END")
            # self.omega_thetas[layer] = ArrayTO.update_3(self.omega_thetas[layer], initial_y_batch_single_opinion[0][0])          
            # print(self.omega_thetas[layer][0][0])
        else:
            raise NotImplementedError
        

    # def eval_plot(EVAL, output_size, title, fname):
    #     eval_trust = EVAL["trust"]
    #     eval_untrust = EVAL["untrust"]
    #     eval_distrust = EVAL["distrust"]
    #     fig, axs = plt.subplots(3, 3, figsize=(20, 11))
    #     for digit_label in range(output_size):
    #         axs[0,0].plot(np.arange(1, len(eval_trust)+1, 1), [eval_trust[i][0][digit_label].t for i in range(len(eval_trust))], label=f"label = {digit_label}")
    #         axs[0,1].plot(np.arange(1, len(eval_trust)+1, 1), [eval_trust[i][0][digit_label].u for i in range(len(eval_trust))], label=f"label = {digit_label}")
    #         axs[0,2].plot(np.arange(1, len(eval_trust)+1, 1), [eval_trust[i][0][digit_label].d for i in range(len(eval_trust))], label=f"label = {digit_label}")
    #     axs[0,0].set_title('Evolution of trust mass for fully trusted input')
    #     axs[0,1].set_title('Evolution of uncertainty mass for fully trusted input')
    #     axs[0,2].set_title('Evolution of distrust mass for fully trusted input')

        
    #     for digit_label in range(output_size):
    #         axs[1,0].plot(np.arange(1, len(eval_trust)+1, 1), [eval_untrust[i][0][digit_label].t for i in range(len(eval_trust))], label=f"label = {digit_label}")
    #         axs[1,1].plot(np.arange(1, len(eval_trust)+1, 1), [eval_untrust[i][0][digit_label].u for i in range(len(eval_trust))], label=f"label = {digit_label}")
    #         axs[1,2].plot(np.arange(1, len(eval_trust)+1, 1), [eval_untrust[i][0][digit_label].d for i in range(len(eval_trust))], label=f"label = {digit_label}")
    #     axs[1,0].set_title('Evolution of trust mass for vacuous input')
    #     axs[1,1].set_title('Evolution of uncertainty mass for vacuous input')
    #     axs[1,2].set_title('Evolution of distrust mass for vacuous input')
        
    #     for digit_label in range(output_size):
    #         axs[2,0].plot(np.arange(1, len(eval_trust)+1, 1), [eval_distrust[i][0][digit_label].t for i in range(len(eval_trust))], label=f"label = {digit_label}")
    #         axs[2,1].plot(np.arange(1, len(eval_trust)+1, 1), [eval_distrust[i][0][digit_label].u for i in range(len(eval_trust))], label=f"label = {digit_label}")
    #         axs[2,2].plot(np.arange(1, len(eval_trust)+1, 1), [eval_distrust[i][0][digit_label].d for i in range(len(eval_trust))], label=f"label = {digit_label}")
    #     axs[2,0].set_title('Evolution of trust mass for fully distrusted input')
    #     axs[2,1].set_title('Evolution of uncertainty mass for fully distrusted input')
    #     axs[2,2].set_title('Evolution of distrust mass for fully distrusted input')

    #     # for ii in range(3):
    #     #     for jj in range(3):
    #     #         axs[ii,jj].legend()
    #     handles, labels = axs[0,0].get_legend_handles_labels()
    #     fig.legend(handles, labels)
    #     # plt.tight_layout(rect=[0, 0, 1, 0.95])  # make space for the suptitle and legend
    #     # fig.suptitle()
    #     # fig.suptitle(title)
    #     plt.savefig(fname)
    #     # plt.show()

    def eval_plot(EVAL, output_size, title, fname, n_epoch, patch=None):
        eval_trust = EVAL["trust"]
        eval_untrust = EVAL["untrust"]
        eval_distrust = EVAL["distrust"]

        num_rounds = len(eval_trust)
        rounds_per_epoch = num_rounds // n_epoch
        epoch_boundaries = [(i + 1) * rounds_per_epoch for i in range(n_epoch - 1)]
        if(patch):
            n_row = 5
            eval_patch_tr = EVAL["patch_tr"]
            eval_patch_vac = EVAL["patch_vac"]
            fig, axs = plt.subplots(n_row, 3, figsize=(20, 20))
        else:
            n_row = 3
            fig, axs = plt.subplots(n_row, 3, figsize=(20, 11))

        def plot_epoch_lines(ax):
            for boundary in epoch_boundaries:
                ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=1)

        for digit_label in range(output_size):
            axs[0,0].plot(range(1, num_rounds + 1), [eval_trust[i][0][digit_label].t for i in range(num_rounds)], label=f"label = {digit_label}")
            axs[0,1].plot(range(1, num_rounds + 1), [eval_trust[i][0][digit_label].u for i in range(num_rounds)], label=f"label = {digit_label}")
            axs[0,2].plot(range(1, num_rounds + 1), [eval_trust[i][0][digit_label].d for i in range(num_rounds)], label=f"label = {digit_label}")
        axs[0,0].set_title('Evolution of trust mass for fully trusted input')
        axs[0,1].set_title('Evolution of uncertainty mass for fully trusted input')
        axs[0,2].set_title('Evolution of distrust mass for fully trusted input')

        for digit_label in range(output_size):
            axs[1,0].plot(range(1, num_rounds + 1), [eval_untrust[i][0][digit_label].t for i in range(num_rounds)], label=f"label = {digit_label}")
            axs[1,1].plot(range(1, num_rounds + 1), [eval_untrust[i][0][digit_label].u for i in range(num_rounds)], label=f"label = {digit_label}")
            axs[1,2].plot(range(1, num_rounds + 1), [eval_untrust[i][0][digit_label].d for i in range(num_rounds)], label=f"label = {digit_label}")
        axs[1,0].set_title('Evolution of trust mass for vacuous input')
        axs[1,1].set_title('Evolution of uncertainty mass for vacuous input')
        axs[1,2].set_title('Evolution of distrust mass for vacuous input')

        for digit_label in range(output_size):
            axs[2,0].plot(range(1, num_rounds + 1), [eval_distrust[i][0][digit_label].t for i in range(num_rounds)], label=f"label = {digit_label}")
            axs[2,1].plot(range(1, num_rounds + 1), [eval_distrust[i][0][digit_label].u for i in range(num_rounds)], label=f"label = {digit_label}")
            axs[2,2].plot(range(1, num_rounds + 1), [eval_distrust[i][0][digit_label].d for i in range(num_rounds)], label=f"label = {digit_label}")
        axs[2,0].set_title('Evolution of trust mass for fully distrusted input')
        axs[2,1].set_title('Evolution of uncertainty mass for fully distrusted input')
        axs[2,2].set_title('Evolution of distrust mass for fully distrusted input')

        if(patch):
            for digit_label in range(output_size):
                axs[3,0].plot(range(1, num_rounds + 1), [eval_patch_tr[i][0][digit_label].t for i in range(num_rounds)], label=f"label = {digit_label}")
                axs[3,1].plot(range(1, num_rounds + 1), [eval_patch_tr[i][0][digit_label].u for i in range(num_rounds)], label=f"label = {digit_label}")
                axs[3,2].plot(range(1, num_rounds + 1), [eval_patch_tr[i][0][digit_label].d for i in range(num_rounds)], label=f"label = {digit_label}")
            axs[3,0].set_title('Evolution of trust mass for patch trusted input')
            axs[3,1].set_title('Evolution of uncertainty mass for patch trusted input')
            axs[3,2].set_title('Evolution of distrust mass for patch trusted input')

            for digit_label in range(output_size):
                axs[4,0].plot(range(1, num_rounds + 1), [eval_patch_vac[i][0][digit_label].t for i in range(num_rounds)], label=f"label = {digit_label}")
                axs[4,1].plot(range(1, num_rounds + 1), [eval_patch_vac[i][0][digit_label].u for i in range(num_rounds)], label=f"label = {digit_label}")
                axs[4,2].plot(range(1, num_rounds + 1), [eval_patch_vac[i][0][digit_label].d for i in range(num_rounds)], label=f"label = {digit_label}")
            axs[4,0].set_title('Evolution of trust mass for patch vacuous input')
            axs[4,1].set_title('Evolution of uncertainty mass for patch vacuous input')
            axs[4,2].set_title('Evolution of distrust mass for patch vacuous input')

        # Add vertical lines to mark epochs
        if(n_epoch>1):
            for i in range(n_row):
                for j in range(3):
                    plot_epoch_lines(axs[i,j])

        # Add legend and title
    
        epoch_line = Line2D([0], [0], color='gray', linestyle='--', linewidth=1, label='Epoch boundary')
        # Add all label handles + custom epoch line to legend
        handles, labels = axs[0,0].get_legend_handles_labels()
        handles.append(epoch_line)
        labels.append("Epoch boundary")
        handles, labels = axs[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=output_size)
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(fname)
        # plt.show()

    def eval_plot_simpl(EVAL, output_size, title, fname):
        eval_trust = EVAL["trust"]
        eval_untrust = EVAL["untrust"]
        eval_distrust = EVAL["distrust"]
        timesteps = np.arange(1, len(eval_trust) + 1)
        fig, axs = plt.subplots(1, 3, figsize=(20, 11))
        avg_t = []
        avg_u = []
        avg_d = []
        for i in range(len(eval_trust)):
            t_vals = [eval_trust[i][0][digit_label].t for digit_label in range(output_size)]
            u_vals = [eval_trust[i][0][digit_label].u for digit_label in range(output_size)]
            d_vals = [eval_trust[i][0][digit_label].d for digit_label in range(output_size)]
            
            avg_t.append(np.mean(t_vals))
            avg_u.append(np.mean(u_vals))
            avg_d.append(np.mean(d_vals))
        
        axs[0].plot(timesteps, avg_t, label=f"Avg Trust: {round(avg_t[-1],3)}")
        axs[0].plot(timesteps, avg_u, label=f"Avg Uncertainty: {round(avg_u[-1],3)}")
        axs[0].plot(timesteps, avg_d, label=f"Avg DisTrust: {round(avg_d[-1],3)}")
        axs[0].set_title('fully trusted input')
        axs[0].legend()

        avg_t = []
        avg_u = []
        avg_d = []
        for i in range(len(eval_untrust)):
            t_vals = [eval_untrust[i][0][digit_label].t for digit_label in range(output_size)]
            u_vals = [eval_untrust[i][0][digit_label].u for digit_label in range(output_size)]
            d_vals = [eval_untrust[i][0][digit_label].d for digit_label in range(output_size)]
            
            avg_t.append(np.mean(t_vals))
            avg_u.append(np.mean(u_vals))
            avg_d.append(np.mean(d_vals))
        
        axs[1].plot(timesteps, avg_t, label=f"Avg Trust: {round(avg_t[-1],3)}")
        axs[1].plot(timesteps, avg_u, label=f"Avg Uncertainty: {round(avg_u[-1],3)}")
        axs[1].plot(timesteps, avg_d, label=f"Avg DisTrust: {round(avg_d[-1],3)}")
        axs[1].set_title('vacuous input')
        axs[1].legend()

        avg_t = []
        avg_u = []
        avg_d = []
        for i in range(len(eval_untrust)):
            t_vals = [eval_distrust[i][0][digit_label].t for digit_label in range(output_size)]
            u_vals = [eval_distrust[i][0][digit_label].u for digit_label in range(output_size)]
            d_vals = [eval_distrust[i][0][digit_label].d for digit_label in range(output_size)]
            
            avg_t.append(np.mean(t_vals))
            avg_u.append(np.mean(u_vals))
            avg_d.append(np.mean(d_vals))
        
        axs[2].plot(timesteps, avg_t, label=f"Avg Trust: {round(avg_t[-1],3)}")
        axs[2].plot(timesteps, avg_u, label=f"Avg Uncertainty: {round(avg_u[-1],3)}")
        axs[2].plot(timesteps, avg_d, label=f"Avg DisTrust: {round(avg_d[-1],3)}")
        axs[2].set_title('fully distrusted input')
        axs[2].legend()

        # fig.suptitle(title)
        plt.savefig(fname, dpi=300, bbox_inches='tight')


input_dim = 30
hidden_dim = 16
output_dim = 2

















def T_NOT_poisoned(x: np.array, dim):
    op = "trust" 
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))

def T_poisoned(x: np.array, dim):
    op = "trust" 
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))

def T1(x: np.array, dim):
    op = "vacuous" 
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))
def Ttrust(x: np.array, dim):
    op = "trust" 
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))

def Tdistrust(x: np.array, dim):
    op = "distrust" 
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))

def Tvacuous(x: np.array, dim):
    op = "vacuous" 
    n = len(x)
    return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))

def T2(x: np.array, dim):

    op = "vacuous" 
    n = len(x)
    res = np.empty((n, dim))
    for i in range(n):
        for j in range(dim):
            res[i][j] = TrustOpinion.ftrust()

def Tgen(xx, yy):
    assert xx[0]=='x'
    assert yy[0]=='y'
    def inner_function(x: np.array, dim):
        if(dim==input_dim):
            op = xx[1:]
            n = len(x)
            return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))
        if(dim==output_dim):
            op = yy[1:]
            n = len(x)
            return ArrayTO(TrustOpinion.fill(shape = (n, dim), method=op))
    return inner_function






XX = ["xtrust", 'xvacuous', "xdistrust"]
YY = ["ytrust", 'yvacuous', "ydistrust"]
# XX = ["xtrust"]
# YY = ["ytrust"]



def run_uni_test(xx, yy, epsilon_low, epsilon_up):
    assert xx in XX
    assert yy in YY
    omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim+1, hidden_dim), method="vacuous"))
    omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim+1, output_dim), method="vacuous"))
    omega_thetas = [omega_thetas_0, omega_thetas_1]

    Tf = Tgen(xx, yy)
    # ptas = PTAS(omega_thetas, None, PTASInterface(5000), Tf, structure = [input_dim, hidden_dim, output_dim], epsilon_low=10**(-3), epsilon_up=10**(-2))
    ptas = PTAS(omega_thetas, None, PTASInterface(5000), Tf, 
                structure = [input_dim, hidden_dim, output_dim], 
                epsilon_low=epsilon_low, epsilon_up=epsilon_up)
    
    datapath = folder_path+'NN\\eval\\'+str(epsilon_low)+"-"+str(epsilon_up)+"\\"+xx+yy
    os.mkdir(datapath)
    ptas.run_chunk()
    
    # print("--------------------------- 0 0 0 ----------------------")
    # print(ptas.omega_thetas[0].get_shape())
    # print(ptas.omega_thetas[0])
    # print("--------------------------- 1 1 1 ----------------------")
    # print(ptas.omega_thetas[1].get_shape())
    # print(ptas.omega_thetas[1])
    # print("-------------------------------------------------")
    
    writeto(ptas.omega_thetas, datapath+"\\omegas.pkl")
    at = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="trust")))
    # print(at)
    writeto(at, datapath+"\\at.pkl")
    av = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="vacuous")))
    # print(av)
    writeto(av, datapath+"\\av.pkl")
    ad = ptas.apply_feedforward(ArrayTO(TrustOpinion.fill((1, input_dim), method="distrust")))
    # print(ad)
    writeto(ad, datapath+"\\ad.pkl")
    

def test():
    epsilon_low = 0.0001
    epsilon_up = 0.001
    datapath = folder_path+'NN\\eval\\'+str(epsilon_low)+"-"+str(epsilon_up)
    os.mkdir(datapath)
    for xx in XX:
        for yy in YY:
            print(f"Running Test for {xx}, {yy}")
            print()
            run_uni_test(xx, yy, epsilon_low, epsilon_up)
def main_cancer():
    input_dim = 30
    hidden_dim = 16
    output_dim = 2
    omega_thetas_0 = ArrayTO(TrustOpinion.fill(shape=(input_dim+1, hidden_dim), method="vacuous"))
    omega_thetas_1 = ArrayTO(TrustOpinion.fill(shape=(hidden_dim+1, output_dim), method="vacuous"))
    omega_thetas = [omega_thetas_0, omega_thetas_1]
    
    ptas = PTAS(omega_thetas, None, PTASInterface(5000), Tdistrust, structure = [input_dim, hidden_dim, output_dim], epsilon_low=0.03)
    datapath = folder_path+'NN\\eval_cancer'
    os.mkdir(datapath)

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
    print(a)
    print("Aggregated Value: ", ptas.aggregation(a))
    print()
    writeto(a, datapath+"\\ad.pkl")


def tryy():
    xx = "xtrust"
    yy = "ytrust"
    datapath = folder_path+'NN\\eval\\'+xx+yy
    with open(datapath+"\\omegas.pkl", "rb") as file:  # Use 'rb' for reading binary
        loaded_data = pickle.load(file)
    print("Loaded data:", loaded_data[0])


if __name__ == "__main__":
    # main_mnist_2()
    print("nothing to run")
    # a = ArrayTO(TrustOpinion.fill(shape = (5, 5), method="trust"))
    # for i in range(3):
    #     for j in range(3):
    #         a.value[i][j] = TrustOpinion.dtrust()
    # print(a)
# ttt()
# test_Tgen_pois()
# tryy()
# main_mnist()
# test()
# (0.383,0.312,0.305,0.226)
# (0.351,0.264,0.385,0.246)
# (0.248,0.61,0.141,0.304)



#(0.404,0.36,0.236,0.262)
#(0.254,0.468,0.279,0.254)
#(0.052,0.818,0.13,0.3)
