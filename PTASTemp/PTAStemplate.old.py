import socket 
import pickle 
from ptasInterface import PTASInterface
from messageObject import MessageObject 
from mode import Mode
        
class PTAS:
    def __init__(self, weights, operator_mapping, _nn_interface: PTASInterface, trust_assessment_func):
        """
        Initialize the PTAS with essential components.
        """
        self.omega_thetas = weights                        # Initialization of the weights for trust functions
                                                           # Also illustrate the structure mirroring
        
        self.Ops = operator_mapping                      # Define operator mappings

        self.nn_interface = _nn_interface                  # Interface with the neural network to retrieve relevant information
                                                           # For now it's a dict with port number

        self.TrustAssessment = trust_assessment_func       # Trust assessment function to evaluate dataset trustworthiness

    
    def process_data(self, message_obj: MessageObject):
        # Action to perform on received data
        print("Processing received message:")
        print(message_obj)
        if (message_obj.mode == Mode.TRAINING):
            self.build()
        if(message_obj.mode == Mode.INFERENCE):
            self.CPTA(None)


    def run(self):
        port=self.nn_interface.port_number
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind(('127.0.0.1', port))
            server_socket.listen()
            print(f"Listening on port {port}...")

            while True:
                client_socket, address = server_socket.accept()
                with client_socket:
                    print(f"Connected by {address}")
                    data = b""
                    while True:
                        packet = client_socket.recv(4096)
                        if not packet:
                            break
                        data += packet
                    # Deserialize the data
                    message_obj = pickle.loads(data)
                    # Process the deserialized object
                    self.process_data(message_obj)
    
    
    def CPTA(self, computational_path):
        """
        Generate a subPTAS based on the computational path (neurons involved in the computation).
        
        Args:
            computational_path: A list of neurons involved in the computation path.
        
        Returns:
            subPTAS: A subPTAS created based on the activated neurons in the specified computational path.
        """
        # Define 'involved' as neurons that are activated during the computation path.
        # To identify these neurons, we will order neurons within each layer to keep track of activations.

        
        return PTAS(self.omega_thetas, self.Ops, self.TrustAssessment)

    def feed_forward(self, trust_in_input):
        """
        Perform a feed-forward pass through the PTAS to calculate trust in the output.
        
        Args:
            trust_in_input: Trust values associated with the input data.
        
        Returns:
            trust_in_output: Computed trust in the output after applying all trust functions in trust nodes.
        """
        # Apply all the trust functions in each trust node through the structure
        # to propagate trust values from input to output.
   
        return None 

    def trust_revision(self, layer, delta):
        """
        Revise trust functions by updating all omega_thetas (weights) for a given layer.
        
        Args:
            layer: The specific layer in the network for which the trust functions need revision.
        """
        # Update trust functions for all trust nodes in the given layer by adjusting omega_thetas.
        T_thetai_given_ybatch = None 
        T_thetai_given_notybatch = None  
        T_thetai = None 
        self.update_omega_thetas(layer) 

    def build(self, data_point):
        """
        Build the PTAS by continuously listening to the neural network interface.
        This function monitors the training process of the neural network, 
        evaluates trust for each data point, and revises trust functions accordingly.
        """
        x, y = data_point
        
        # 1. Evaluate the trust in the data point using TrustAssessment.
        Tx = self.TrustAssessment(x)
        
        # 2. Perform feed-forward operation on the trust value of the input.
        Ty = self.feed_forward(Tx)
        TyBatch = Id(Ty) ### Fuse a Ty 
        for current_layer in range(1, 1, -1):
            self.trust_revision(current_layer)


        # 3. Retrieve delta data (error information) of the last layer through nn_interface.
        
        # 4. Apply trust revision for the last layer.
        
        
        # Continue revising trust functions, moving layer by layer up to the first hidden layer.
        #     # Move to the previous layer and apply trust revision
        #     self.trust_revision(current_layer, delta_current_layer)
            
        # Repeat with the next data point being processed by the neural network.

def Id(x):
    return x 
def main():
    ptas = PTAS(None, None, PTASInterface(5000), Id)
    ptas.run()

main()