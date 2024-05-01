### training.py ###
# contains all the functions necessary to train and test a predictive coding network

import numpy as np
import matplotlib.pyplot as plt
from pc_network import *
from activations import *

def create_learning_inputs(X, Y, epochs, stabilize_time, learn_time, shuffle=True):
    """ (np.array, np.array, int, num, num) -> (dict, dict, num)

    X                   Numpy array, datapoints to use as sensory input. Each row is a datapoint, the columns are features.
    Y                   Numpy array, targets for perception corresponding to each datapoint. Each row is a given observation's target.
    epochs              Int, the number of times to run through the dataset.
    stablize_time       Num, the amount of simulation time to spend at each datapoint without learning.
    learn_time          Num, the amount of simulation time to spend at each datapoint with learning.
    shuffle             Boolean, whether or not to shuffle the dataset at each epoch. Defaults to True.

    Returns
    x_dict              Dict, contains the data inputs.
    y_dict              Dict, contains the target vectors.
    t                   Num, the total simulation time for learning.
    """
    #our dictionaries to return
    x_dict = {}
    y_dict = {}

    #size of the dataset
    if len(X.shape) == 1:
        num_points = X.shape[0]
        num_features = 1
    else:
        num_points, num_features = X.shape
    #the size of the targets
    if len(Y.shape) == 1: #one-dimensional ouputs
        target_dimensions = 1 
    else:
        target_dimensions = Y.shape[1] 

    t = 0 #keep track of the simulation time
    point_time = stabilize_time + learn_time #the total amount of time to spend at one point

    indices = np.arange(num_points) #the indices of the datapoints

    for e in range(epochs):
        if shuffle: #shuffle the dataset each epoch
            np.random.shuffle(indices)
        for i in indices:
            if num_features == 1:
                x_dict[t] = X[i]
            else:
                x_dict[t] = X[i,:]
            if target_dimensions == 1:
                y_dict[t] = Y[i]
            else:
                y_dict[t] = Y[i,:]
            
            t += point_time #increment the time step
    
    return x_dict, y_dict, t


def train_network(sim, pc_net, stabilize_time, learn_time, learn_until):
    """ (nengo.Simulator, PCNetwork, num, num, num) -> ()

    sim                 nengo.Simulator, the simulator created using the network we wish to train.
    pc_net              PCNetwork, the network we wish to train.
    stablize_time       Num, the amount of simulation time to spend at each data point without learning.
    learn_time          Num, the amount of simulation time to spend at each data point with learning.
    learn_until         Num, the total simulation time.

    """
    sim.run(learn_until)
    """
    time = 0 #keep track of current time
    while time < t:
        pc_net.update_learning_rule(False)
        sim.run(stabilize_time)
        pc_net.update_learning_rule(True)
        sim.run(learn_time) 
        time += stabilize_time + learn_time
    pc_net.update_learning_rule(False) #set the network to not learn anymore
    """


def add_inference_inputs(X, Y, x_dict, y_dict, inf_time, learn_until):
    """ (np.array, np.array, num, num) -> (dict, dict, num)

    Adds entries to the sensory and prediction dictionnaries to be used in inference.

    X                   Numpy array, datapoints to use as sensory input. Each row is a datapoint, the columns are features.
    Y                   Numpy array, targets for perception corresponding to each datapoint. Each row is a given observation's target.
    x_dict              Dictionary, sensory inputs.
    y_dict              Dictionary, target outputs.
    inf_time            Num, the amount of simulation time to spend before reaching equilibrium.
    learn_until         Num, the time at which the network begins inference.

    Outputs:
        Y.shape[0]      Int, the number of predictions the network needs to make
    """
    #input validation
    if inf_time == 0:
        raise ValueError("inf_time cannot be zero or only one instance will be added to the dictionary, overriding previous inputs.")
    #size of the dataset
    if len(X.shape) == 1:
        num_points = X.shape[0]
        num_features = 1
    else:
        num_points, num_features = X.shape
    #the size of the targets
    if len(Y.shape) == 1: #one-dimensional outputs
        target_dimensions = 1 
    else:
        target_dimensions = Y.shape[1] 

    indices = np.arange(num_points) #the indices of the datapoints

    for i in indices:
        if num_features == 1:
            x_dict[learn_until] = X[i]
        else:
            x_dict[learn_until] = X[i,:]
        if target_dimensions == 1:
            y_dict[learn_until] = Y[i]
        else:
            y_dict[learn_until] = Y[i,:]
        
        learn_until += inf_time #increment the time step
    
    return Y.shape[0]


def test_network(sim, pc_net, num_predictions, inf_time, output_probe):
    """
    
    Inputs:
        sim              nengo.Simulator, the simulator created using the network we wish to train.
        pc_net           PCNetwork, the network we wish to train.
        num_predictions  Int, the number of predictions to make.
        inf_time         Num, the amount of simulation time to spend before reaching equilibrium.
        output_probe     Probe, an nengo probe connected to the last layer of the network used for inference.

    Outputs:
        predictions      Array, a list of predictions.
    """
    predictions = []

    for i in range(num_predictions):
        sim.run(inf_time) #reach equilibrium

        predictions.append(sim.data[output_probe][-1])
    
    return np.vstack(predictions)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    import matplotlib.pyplot as plt

    x_data, y_data = load_iris(return_X_y=True)
    y_data = y_data.reshape((y_data.shape[0],1))
    enc = OneHotEncoder()
    y_data = enc.fit_transform(y_data).toarray()
    #define data
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True)
    with nengo.Network() as net:
        tau = 0.01
        #============== Run in Direct mode =======================
        net.config[nengo.Ensemble].neuron_type = nengo.Direct()
        #=========================================================
        net.config[nengo.Connection].synapse = tau
        net.config[nengo.Probe].synapse = 0
        
        epochs = 5
        stab_time = 0
        learn_time = 0.2
        shuffle = True
        inf_time = 0.2
        #define data
        x_input, y_input, learn_until = create_learning_inputs(x_train, y_train, epochs, stab_time, learn_time, shuffle)
        #add testing phase
        num_preds = add_inference_inputs(x_test, y_test, x_input, y_input, inf_time=inf_time, learn_until=learn_until)

        stim = nengo.Node(nengo.processes.Piecewise(x_input))   # sensory (bottom layer)
        pred = nengo.Node(nengo.processes.Piecewise(y_input))  # percept (top layer)
        
        # PC layers
        PC_net = PCNetwork(n_nodes=[4, 10, 3], tau_learn=0.1, symmetric=True, activation=Tanh, learn_until=learn_until)
        PC_net.connect_input(stim=stim)
        PC_net.connect_output(pred)

        # Set up a bunch of probes (so we can plot stuff later)
        p_stim = nengo.Probe(stim)
        p_pc1v = nengo.Probe(PC_net.layers[0].v.output)
        p_pc1e = nengo.Probe(PC_net.layers[0].e.output)
        p_pc_end_v = nengo.Probe(PC_net.layers[-1].v.output)
        p_pc_end_e = nengo.Probe(PC_net.layers[-1].e.output)
        p_pred = nengo.Probe(pred)

        val_probes, err_probes = PC_net.get_probes()

    sim = nengo.Simulator(net)
    train_network(sim, PC_net, stab_time, learn_time, learn_until)
    preds = test_network(sim, PC_net, num_preds, inf_time, val_probes[-1])
    
    pred_class = np.zeros_like(preds)
    indices = np.argmax(preds, axis=1)
    i = 0
    for idx in indices:
        pred_class[i,idx] = 1
        i += 1

    acc = np.sum(pred_class*y_test)/y_test.shape[0]

    ### Print summary of the network learning ###
    print("Architecture:")
    print(f"- Layers: {[layer.n_nodes for layer in PC_net.layers]}")
    print(f"- Activations: {PC_net.activation}")
    print(f"- Symmetric: {PC_net.symmetric}")
    print(f"- Learning rate: {PC_net.tau_learn}")
    print(f"Trained the network for {epochs} epochs (shuffled={shuffle}), holding each sample for {learn_time} s. Inference time was {inf_time} s.")

    print("Testing:")
    print(f"The number of each class in the test set: {np.sum(y_test, axis=0)}")
    print(f"The number of each class predicted when testing: {np.sum(pred_class, axis=0)}")

    print(f"Test accuracy: {acc}")

    ### Plotting ###

    n_layers = len(PC_net.layers)

    fig, ax = plt.subplots(nrows=1, ncols=n_layers, figsize=(13,5))

    idx = [0, 4000]
    idx = [7000, 10000]
    #idx = [18000, 20000]
    idx = [0, len(sim.trange())]
    #idx = [115000, 120000]
    tt = sim.trange()[idx[0]:idx[1]]

    for k in range(n_layers):
        ax[k].plot(tt, sim.data[val_probes[k]][idx[0]:idx[1]]); ax[k].set_title(f'Layer {k}');

    ax[0].plot(tt, sim.data[p_stim][idx[0]:idx[1]], ':', label="Stim");  
    ax[-1].plot(tt, sim.data[p_pred][idx[0]:idx[1]], '--', label="Pred");


    print(f"We learn until t={learn_until}")

    n_layers = len(PC_net.layers)

    fig2, ax2 = plt.subplots(nrows=1, ncols=n_layers, figsize=(13,5))

    idx = [0, 2000]
    idx = [7000, 7800]
    idx = [int(learn_until/0.001), -1] #plot only learning segments
    #idx = [0, len(sim.trange())]
    tt = sim.trange()[idx[0]:idx[1]]

    for k in range(n_layers):
        ax2[k].plot(tt, sim.data[val_probes[k]][idx[0]:idx[1]]); ax2[k].set_title(f'Layer {k}');


    ax2[0].plot(tt, sim.data[p_stim][idx[0]:idx[1]], ':');    
    ax2[-1].plot(tt, sim.data[p_pred][idx[0]:idx[1]], '--');    