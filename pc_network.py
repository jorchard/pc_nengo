### pc_network.py ###
# contains all the classes necessary for implementing a predictive coding network

import numpy as np
import nengo
from activations import *
from learning_rates import *

class PCLayer(nengo.Network):
    '''
    l = PCLayer(n_nodes=10, tau=0.01)

    Creates a layer for a predictive-coding network.
    This layer has an array of value nodes, and a corresponding array of error nodes.

    Inputs:
    n_nodes       number of nodes in the layer
    tau           synaptic time constant for internal error->value connections
    pred_layer    is this a prediction layer
    '''
    def __init__(self, n_nodes=10, tau=0.01, pred_layer=False):
        self.label = 'PCLayer'
        self.n_nodes = n_nodes
        self.pred_layer = pred_layer
        self.tau = tau
        self.inference_node = None

        # When the network is run in using the "Direct" neuron model, it treats
        # each "Ensemble" as a single variable, and the decodings are done by
        # the user-supplied decoding functions, not using decoding weights.
        # In this case, the "n_neurons" argument is ignored.

        # An "EnsembleArray" can be thought of as a group of nodes.
        # An EnsembleArray holding the values (v)
        self.v = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=n_nodes, radius=1.5)#, neuron_type=nengo.LIFRate())
        # and an EnsembleArray holding the errors (e)
        self.e = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=n_nodes, radius=1.5)#, neuron_type=nengo.LIFRate())

        #node in between e and v for inference

        nengo.Connection(self.v.output, self.v.input, transform=1, synapse=tau)               # sustain old state

        input_weight = 1
        
        if self.pred_layer:
            self.pass_through = nengo.Node(self.step, size_in=n_nodes, size_out=n_nodes)   #pass-through node between (e) and (v)
            nengo.Connection(self.e.output, self.pass_through, transform=input_weight, synapse=None)            # (v)<-(e)                                               
            nengo.Connection(self.pass_through, self.v.input, transform=-input_weight, synapse=tau) 
        else:           
            nengo.Connection(self.e.output, self.v.input, transform=-input_weight, synapse=tau)            
        nengo.Connection(self.v.output, self.e.input, transform=input_weight, synapse=tau)                            # (v)->(e)

    def step(self, t, x):
        if t > 0 and self.inference_node.output(t) == 0: #simply pass through the input
            return x
        else: #either initial step or we are doing inference, block input
            if hasattr(x, '__len__'): #input with length
                return [0 for i in range(len(x))]
            else: #should just be an int or float
                return 0



class Updater(nengo.Process):
    """
    Extends the nengo.Process class to be used in the PCConnection object. 
    A PCConnection instance will create an Updater during initialization.

    connection_inst     The instance of the PCConnection class that created this Updater.
    """
    def __init__(self, connection_inst, **kwargs):
        super().__init__(default_size_in=0, **kwargs)
        self.connection_inst = connection_inst
        self.shape_in = self.connection_inst.n_e + self.connection_inst.n_v
        self.shape_out = self.connection_inst.n_e + self.connection_inst.n_v
    

    def __repr__(self):
        return "Updater"
    

    def make_step(self, shape_in, shape_out, dt, rng, state):
        """
        Creates a function that will used at every time step in order to evaluate the output of the Updater.
        """
        def step_updater(t, x):
            e_in = x[:self.connection_inst.n_e]
            v_in = x[self.connection_inst.n_e:]

            err_out = self.connection_inst.activation.deriv(v_in) * (e_in @ self.connection_inst.W)
            pred_out = self.connection_inst.activation.func(v_in) @ self.connection_inst.M

            if self.connection_inst.inference_node.output(t) == 0: #not doing inference, so we learn
                dM = np.outer(self.connection_inst.activation.func(v_in), e_in)
                #print("tau learn", self.connection_inst.tau_learn(t))
                #print("dM", dM)
                #print("M", self.M)
                self.connection_inst.M += dt * dM / self.connection_inst.tau_learn(t)
                if not self.connection_inst.symmetric:
                    self.connection_inst.W += dt * dM.T / self.connection_inst.tau_learn(t)

            return np.concatenate((pred_out, err_out)) 
        return step_updater



class PCConnection(nengo.Network):
    '''
     c = PCConnection(below, above, learn=False, symmetric=True)
     
     This class builds all the connections between the two PCLayers.
     
     Inputs:
      below        PCLayer object below
      above        PCLayer object above
      learn        Boolean, whether to learn (default False)
      symmetric    Boolean, whether the connection is symmetric
                   ie. W = M.T (default True)
      activation   Activation function to use. Must be an object with func and deriv methods.
    '''
    def __init__(self, below, above, inference_node=None, tau_learn=ConstRate(20.), symmetric=True, M=None, activation=None):
        self.label = None
        self.below = below
        self.above = above
        self.tau_learn = tau_learn    # learning time constant
        self.symmetric = symmetric
        self.n_e = self.below.e.n_ensembles  # dimension of below layer
        self.n_v = self.above.v.n_ensembles  # dimension of above layer

        self.inference_node = inference_node #we will learn if and only if not doing inference
        self.updater = Updater(self) #internal node we use to apply the Euler step for updating weights
        
        self.activation = activation

        # Set up connect matrices
        if M is None:
            self.M = np.random.normal(size=(self.n_v, self.n_e))/10.
        else:
            self.M = M
        if self.symmetric:
            self.W = self.M.T
        else:
            self.W = np.random.normal(size=(self.n_e, self.n_v))


        # Set up the node that applies the connection weights
        dims = self.n_e + self.n_v
        #self.exchange = nengo.Node(Updater(self), size_in=dims, size_out=dims)
        self.exchange = nengo.Node(self.updater, size_in=dims, size_out=dims)
        

        n = self.n_e
        nengo.Connection(self.below.e.output, self.exchange[:n], synapse=None)               # inp -> exchange
        nengo.Connection(self.exchange[:n], self.below.e.input, transform=-1)  # inp <- con
        nengo.Connection(self.above.v.output, self.exchange[n:], transform=1, synapse=None)  # con <- hid
        nengo.Connection(self.exchange[n:], self.above.v.input)                # con -> hid




class PCNetwork:
    '''
    This class builds a PC neural network with.
    
    Inputs:
    n_nodes               List, the of number of nodes at each hidden layer
    learn                 Boolean, whether to learn (default False)
    symmetric             Boolean, whether the connection is symmetric
                          ie. W = M.T (default True)
    tau_learn             num or List, learning rate for the Euler step when updating the weights
                            - if num, each connection will have the same tau_learn
                            - if List, each connection will have a tau_learn corresponding to the value in that
                              position in the list
    M                     ndarray, List or None, value of M at each layer, defaults to None
                            - if ndarray or None, each connection will have M passed in as an argument during construction
                            - if List, each connection will have an M corresponding to the value in that
                              position in the list
    activation            None, string, or list, defaults to None
                            - if string or None, each connection will have the same activation function
                            - if list, each connection will have an activation fucntion corresponding to the value
                              in that position in the list
    learn_until           Num, the amount of simulation time to spend learning connection weights, defaults to 0
    '''
    def __init__(self, n_nodes=None, symmetric=True, tau_learn=ConstRate(20.), M=None, activation=None, learn_until=0):
        if n_nodes is None:
            n_nodes = []
        elif type(n_nodes) is not list:
            raise TypeError(f"n_nodes must be a list or an int, instead got {type(n_nodes)}")
        else:
            self.n_nodes = n_nodes
        self.num_hidden_layers = len(self.n_nodes)
        self.layers = [] #a list of all layers in order of connection
        self.connections = [] #a list of all connections between layers
        self.symmetric = symmetric
        self.learn_until = learn_until
        self.inference_node = nengo.Node(self.inference_output)

        #set up member attributes and input validation
        if type(tau_learn) is list:
            if len(tau_learn) != self.num_hidden_layers-1:
                raise ValueError(f"If tau_learn is a list, it must have {self.num_hidden_layers-1} elements, instead got {len(tau_learn)}")
            
            for tau in tau_learn:
                if not callable(tau): #is not callable
                    raise NotImplementedError("The learning rates must be callable with one input argument.")
        elif not callable(tau_learn): #is not callable
            raise NotImplementedError("The learning rates must be callable with one input argument.")
        else:
            self.tau_learn = [tau_learn for i in range(self.num_hidden_layers-1)]

  
        if type(M) is np.ndarray or M is None:
            self.M = [M for i in range(self.num_hidden_layers-1)]
        elif type(M) is not list:
            raise TypeError(f"M must be an ndarray, list, or None, instead got {type(M)}")
        else:
            if len(M) != self.num_hidden_layers-1:
                raise ValueError(f"If M is a list, it must have {self.num_hidden_layers-1} elements, instead got {len(M)}")
            self.M = M
        

        if type(activation) is list:
            if len(activation) != self.num_hidden_layers-1:
                raise ValueError(f"If activation is a list, it must have {self.num_hidden_layers-1} elements, instead got {len(activation)}")
            
            for act in activation:
                if not (hasattr(act, "func") and hasattr(act, "deriv")): #does not have the required methods
                    raise NotImplementedError("The activation function must implement \'func\' and \'deriv\' methods/function attributes.")
        elif not (hasattr(activation, "func") and hasattr(activation, "deriv")): #does not have the required methods
            raise NotImplementedError("The activation function(s) must implement \'func\' and \'deriv\' methods/function attributes.")
        else:
            self.activation = [activation for i in range(self.num_hidden_layers-1)]

        
        #create hidden layers
        self.layers.append(PCLayer(n_nodes=self.n_nodes[0]))
        for i in range(1, self.num_hidden_layers-1): #create and connect the rest of the layers
            self.layers.append(PCLayer(n_nodes=self.n_nodes[i]))
            self.connections.append(PCConnection(self.layers[i-1], 
                                                 self.layers[i], 
                                                 inference_node=self.inference_node, 
                                                 symmetric=self.symmetric,
                                                 M=self.M[i-1],
                                                 tau_learn=self.tau_learn[i-1],
                                                 activation=self.activation[i-1]))
        #prediction layer
        self.layers.append(PCLayer(n_nodes=self.n_nodes[-1], pred_layer=True))
        self.connections.append(PCConnection(self.layers[-2], 
                                                self.layers[-1], 
                                                inference_node=self.inference_node, 
                                                symmetric=self.symmetric,
                                                M=self.M[-1],
                                                tau_learn=self.tau_learn[-1],
                                                activation=self.activation[-1]))

            

    def inference_output(self, time):
        if time < self.learn_until: #not inference, training
            return 0
        else: #inference
            return 1
     
            
    def connect_input(self, stim):
        """ (self, nengo.Node) -> ()
        Connects the input (stimulus) layer to the first hidden layer of the network.
        """
        D = stim.size_out #dimensionality of inputs to the the network
        
        if self.n_nodes[0] != D:
            raise ValueError("Dimensionality of the input does not match the dimensionality of the first layer.")
        
        stim_err = nengo.Node(lambda t,x: x[:D]+x[D:], size_in=2*D, size_out=D)  # sensory error
        nengo.Connection(stim, stim_err[:D], synapse=None)                                 # stim -> stim_err
        nengo.Connection(stim_err, self.layers[0].v.input)                   # stim_err -> first layer
        nengo.Connection(self.layers[0].v.output, stim_err[D:], transform=-1) # stim_err <- first layer

    
    def connect_output(self, pred):
        """ (self, nengo.Node, num) -> ()
        Connects the output (prediction) layer to the last hidden layer of the network. 
        """
        if self.n_nodes[-1] != pred.size_out:
            raise ValueError("Dimensionality of the target does not match the dimensionality of the last layer.")
         
        self.layers[-1].inference_node = self.inference_node
        nengo.Connection(pred, self.layers[-1].e.input, transform=-1)         # pc2 <- pred

    
    def get_probes(self):
        """ (self) -> (list, list)
        Returns a list of nengo.Probes to track each layers values and errors.

        ##### NOT GREAT WHEN DIMENSIONALITY OF INPUT IS MORE THAN 1
        """
        val_probes = []
        err_probes = []
        for idx, layer in enumerate(self.layers):
            val_probes.append(nengo.Probe(layer.v.output, label=f"Hidden Layer {idx} Value"))
            err_probes.append(nengo.Probe(layer.e.output, label=f"Hidden Layer {idx} Error"))
        return val_probes, err_probes