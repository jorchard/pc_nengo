### equilibrium.py ###
# contains various functions we can use to test the network for convergence rate/quality

import numpy as np
from pc_network import *
from activations import *
from training import *


## this is the model function that we will base our convergence tests off of
# run the network for some amount of time, check for convergence
# repeat until convergence
# Do we have multiple functions to test for convergence using different rules? 
# Or do we have one function that can take the rule as input?
# I think the first since it can give us more leeway to change the function and implement the rule more easily

def test_network_L2(sim, pc_net, num_predictions, timeout, inf_time, stab_time, output_probe, eps=1e-5):
    """ (nengo.Simulator, PCNetwork, int, num, num, num, nengo.Probe, num) -> (np.array, np.array)
    Runs the PCNetwork pc_net when it is doing inference. For all num_predictions inputs, tests for convergence and divergence.
    For a given input, runs the network for inf_time, then checks for convergence/divergence.
    Convergence is measured by the L2 norm with tolerance eps.
    Timeout provides a limit on the number of iterations we can spend at a single prediction. Defaults to infinity i.e. no limit.
    If the network diverges that prediction will be infinity.
    
    Inputs:
        sim                 nengo.Simulator, the simulator created using the network we wish to train.
        pc_net              PCNetwork, the network we wish to train.
        num_predictions     Int, the number of predictions to make.
        timeout             Num, the maximum amount of time to spend at one input.
        inf_time            Num, the amount of simulation time to spend before checking for equilibrium.
        stab_time           Num, the amount of time for the network to stabilize at zero
        output_probe        Probe, an nengo probe connected to the last layer of the network used for inference.
        eps                 Num, tolerance for convergence

    Outputs:
        predictions         Array, a list of predictions.
        equilibrium_times   Array, the time it takes for the network to reach the equilibrium for each prediction.
    """
    predictions = []
    equilibrium_times = []

    for i in range(num_predictions):
        sim.run(stab_time) #reset the networ to 0
        time = 0 #the time spent at this prediction

        starting_index = sim.data[output_probe].shape[0] #how many steps has the simulation run through so far?
        current_index = starting_index #how many steps since we started testing for equilibrium?

        at_equilibrium = False #flag whether the PC network ran until equilibrium
        diverged = False
        while not at_equilibrium and time < timeout:
            sim.run(inf_time) 
            time += inf_time

            #### check equilibrium conditions ####
            data = sim.data[output_probe][current_index:] #all points from this chunk of simulation steps
            #data used to test divergence
            #divergence_data = sim.data[output_probe][starting_index:] #all points for this prediction

            #check for divergence
            #diverged = True

            #check for equilibrium
            if np.sqrt(np.sum((data[-1] - data[0])**2)) > eps: #not at equilibrium
                current_index = sim.data[output_probe].shape[0] #we will run again
            else: #reached equilibrium
                at_equilibrium = True 

        if diverged or time >= timeout: #if the network diverged we will fill the prediction with infs
            predictions.append(np.full_like(sim.data[output_probe][-1], np.inf))
            equilibrium_times.append(np.inf)
        else:
            predictions.append(sim.data[output_probe][-1])
            equilibrium_times.append(time)

        if time < timeout: #run the network for the rest of the input to get to the next one
            sim.run(timeout - time)
    
    return np.vstack(predictions), np.array(equilibrium_times)


def test_network_L1(sim, pc_net, num_predictions, timeout, inf_time, stab_time, output_probe, eps=1e-5):
    """ (nengo.Simulator, PCNetwork, int, num, num, num, nengo.Probe, num) -> (np.array, np.array)
    Runs the PCNetwork pc_net when it is doing inference. For all num_predictions inputs, tests for convergence and divergence.
    For a given input, runs the network for inf_time, then checks for convergence/divergence.
    Convergence is measured by the L1 norm with tolerance eps.
    Timeout provides a limit on the number of iterations we can spend at a single prediction. Defaults to infinity i.e. no limit.
    If the network diverges that prediction will be infinity.
    
    Inputs:
        sim                 nengo.Simulator, the simulator created using the network we wish to train.
        pc_net              PCNetwork, the network we wish to train.
        num_predictions     Int, the number of predictions to make.
        timeout             Num, the maximum amount of time to spend at one input.
        inf_time            Num, the amount of simulation time to spend before checking for equilibrium.
        stab_time           Num, the amount of time for the network to stabilize at zero
        output_probe        Probe, an nengo probe connected to the last layer of the network used for inference.
        eps                 Num, tolerance for convergence

    Outputs:
        predictions         Array, a list of predictions.
        equilibrium_times   Array, the time it takes for the network to reach the equilibrium for each prediction.
    """
    predictions = []
    equilibrium_times = []

    for i in range(num_predictions):
        sim.run(stab_time) #reset the networ to 0
        time = 0 #the time spent at this prediction

        starting_index = sim.data[output_probe].shape[0] #how many steps has the simulation run through so far?
        current_index = starting_index #how many steps since we started testing for equilibrium?

        at_equilibrium = False #flag whether the PC network ran until equilibrium
        diverged = False
        while not at_equilibrium and time < timeout:
            sim.run(inf_time) 
            time += inf_time

            #### check equilibrium conditions ####
            data = sim.data[output_probe][current_index:] #all points from this chunk of simulation steps
            #data used to test divergence
            #divergence_data = sim.data[output_probe][starting_index:] #all points for this prediction

            #check for divergence
            #diverged = True

            #check for equilibrium
            if np.sqrt(np.sum(np.abs(data[-1] - data[0]))) > eps: #not at equilibrium
                current_index = sim.data[output_probe].shape[0] #we will run again
            else: #reached equilibrium
                at_equilibrium = True 

        if diverged or time >= timeout: #if the network diverged we will fill the prediction with infs
            predictions.append(np.full_like(sim.data[output_probe][-1], np.inf))
            equilibrium_times.append(np.inf)
        else:
            predictions.append(sim.data[output_probe][-1])
            equilibrium_times.append(time)

        if time < timeout: #run the network for the rest of the input to get to the next one
            sim.run(timeout - time)
    
    return np.vstack(predictions), np.array(equilibrium_times)


def test_network_max(sim, pc_net, num_predictions, timeout, inf_time, stab_time, output_probe, eps=1e-5):
    """ (nengo.Simulator, PCNetwork, int, num, num, num, nengo.Probe, num) -> (np.array, np.array)
    Runs the PCNetwork pc_net when it is doing inference. For all num_predictions inputs, tests for convergence and divergence.
    For a given input, runs the network for inf_time, then checks for convergence/divergence.
    Convergence is measured by the max norm with tolerance eps.
    Timeout provides a limit on the number of iterations we can spend at a single prediction. Defaults to infinity i.e. no limit.
    If the network diverges that prediction will be infinity.
    
    Inputs:
        sim                 nengo.Simulator, the simulator created using the network we wish to train.
        pc_net              PCNetwork, the network we wish to train.
        num_predictions     Int, the number of predictions to make.
        timeout             Num, the maximum amount of time to spend at one input.
        inf_time            Num, the amount of simulation time to spend before checking for equilibrium.
        stab_time           Num, the amount of time for the network to stabilize at zero
        output_probe        Probe, an nengo probe connected to the last layer of the network used for inference.
        eps                 Num, tolerance for convergence

    Outputs:
        predictions         Array, a list of predictions.
        equilibrium_times   Array, the time it takes for the network to reach the equilibrium for each prediction.
    """
    predictions = []
    equilibrium_times = []

    for i in range(num_predictions):
        sim.run(stab_time) #reset the networ to 0
        time = 0 #the time spent at this prediction

        starting_index = sim.data[output_probe].shape[0] #how many steps has the simulation run through so far?
        current_index = starting_index #how many steps since we started testing for equilibrium?

        at_equilibrium = False #flag whether the PC network ran until equilibrium
        diverged = False
        while not at_equilibrium and time < timeout:
            sim.run(inf_time) 
            time += inf_time

            #### check equilibrium conditions ####
            data = sim.data[output_probe][current_index:] #all points from this chunk of simulation steps
            #data used to test divergence
            #divergence_data = sim.data[output_probe][starting_index:] #all points for this prediction

            #check for divergence
            #diverged = True

            #check for equilibrium
            if np.sqrt(np.sum(np.max(np.abs(data[-1] - data[0])))) > eps: #not at equilibrium
                current_index = sim.data[output_probe].shape[0] #we will run again
            else: #reached equilibrium
                at_equilibrium = True 

        if diverged or time >= timeout: #if the network diverged we will fill the prediction with infs
            predictions.append(np.full_like(sim.data[output_probe][-1], np.inf))
            equilibrium_times.append(np.inf)
        else:
            predictions.append(sim.data[output_probe][-1])
            equilibrium_times.append(time)

        if time < timeout: #run the network for the rest of the input to get to the next one
            sim.run(timeout - time)
    
    return np.vstack(predictions), np.array(equilibrium_times)