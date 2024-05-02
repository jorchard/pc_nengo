### learning_rates.py ###
# contains classes used to define non-constant learning rates for the PC network

import numpy as np

class ConstRate:
    # constant learning rate
    def __init__(self, tau_learn):
        self.tau_learn = tau_learn
    
    def __call__(self, t):
        return self.tau_learn
    
    def __str__(self):
        return f"Constant ({self.tau_learn})"
    

class LinearRate:
    # linearly changing learning rate
    # starts at initial_tau and decreases/increases to final_tau after total_time seconds
    def __init__(self, initial_tau, final_tau, total_time):
        self.initial_tau = initial_tau
        self.final_tau = final_tau
        self.total_time = total_time

        self.slope = (self.final_tau - self.initial_tau)/self.total_time
    
    def __call__(self, t):
        return t*self.slope + self.initial_tau

    def __str__(self):
        return "Linear (start={0}, stop={1}, slope={2})".format(self.initial_tau, self.final_tau, self.slope)

class PowerRate:
    # power rate (**beta) decay/growth
    # starts at initial_tau and decreases/increases to final_tau after total_time seconds
    def __init__(self, initial_tau, final_tau, beta, total_time):
        #input validation
        if initial_tau < final_tau and beta < 0:
            raise ValueError("Beta must be positive when learning rate is increasing.")
        if initial_tau > final_tau and beta > 0:
            raise ValueError("Beta must be negative when learning rate is decreasing.")
        
        self.initial_tau = initial_tau
        self.final_tau = final_tau
        self.total_time = total_time
        self.beta = beta

        #params to define the function, slope and shift
        self.b = self.total_time/(1 - (self.final_tau/self.initial_tau)**(1/self.beta))
        self.a = self.initial_tau/(-self.b)**self.beta
    
    def __call__(self, t):
        return self.a*(t - self.b)**self.beta