### activations.py ###
# contains various classes that implement activation functions and their derivatives

import numpy as np

class ActivationFunction:
    """ 
    Abstract class representing an activation function.
    """
    def __init__(self):
        pass

    @classmethod
    def func(cls, x):
        pass

    @classmethod
    def deriv(cls, x):
        pass

    def __str__(self):
        return self.__class__


class Linear(ActivationFunction):
    """
    A linear activation function i.e. the identity mapping.
    """
    def __init__(self):
        super().__init__()
    
    @classmethod
    def func(cls, x):
        return x
    
    @classmethod
    def deriv(cls, x):
        return np.ones_like(x)
    
class ReLU(ActivationFunction):
    """
    Rectified linear unit activation function.
    """
    def __init__(self):
        super().__init__()
    
    @classmethod
    def func(cls, x):
        return np.maximum(0, x)
    
    @classmethod
    def deriv(cls, x):
        if hasattr(x, '__len__'): #array
            neg_indices = np.where(x <= 0)
            d = np.ones_like(x) #derivatives
            d[neg_indices] = 0
            return d
        else:
            if x <= 0:
                return 0
            else:
                return 1
            

class Logistic(ActivationFunction):
    """
    Logistic activation function.
    """
    def __init__(self):
        super().__init__()
    
    @classmethod
    def func(cls, x):
        return 1/(1 + np.exp(-x))
    
    @classmethod
    def deriv(cls, x):
        return cls.func(x)*(1-cls.func(x))


class Tanh(ActivationFunction):
    """
    Hyperbolic tan activation function.
    """
    def __init__(self):
        super().__init__()
    
    @classmethod
    def func(cls, x):
        return np.tanh(x)
    
    @classmethod
    def deriv(cls, x):
        return 1 - np.tanh(x)**2


class Threshold(ActivationFunction):
    """
    Threshold activation function.
    """
    def __init__(self):
        super().__init__()
    
    @classmethod
    def func(cls, x):
        if hasattr(x, '__len__'): #array
            vals = np.ones_like(x)
            neg_indices = np.where(x <= 0)
            vals[neg_indices] = 0
            return vals
        elif x <= 0:
            return 0
        else:
            return 1
    
    @classmethod
    def deriv(cls, x):
        return np.zeros_like(x)


class Softplus(ActivationFunction):
    """
    Softplus activation function.
    """
    def __init__(self):
        super().__init__()
    
    @classmethod
    def func(cls, x):
        return np.log(1 + np.exp(x))
    
    @classmethod
    def deriv(cls, x):
        return Logistic.func(x)


class LeakyReLU(ActivationFunction):
    """
    Leaky rectified linear unit activation function.
    """
    def __init__(self):
        super().__init__()
    
    @classmethod
    def func(cls, x):
        return np.maximum(x, 0.01*x)
    
    @classmethod
    def deriv(cls, x):
        if hasattr(x, '__len__'):
            d = np.ones_like(x) #derivatives
            neg_indices = np.where(x <= 0)
            d[neg_indices] = 0.01
            return d
        elif x <= 0:
            return 0.01
        else:
            return 1


class Gaussian(ActivationFunction):
    """
    Gaussian activation function.
    """
    def __init__(self):
        super().__init__()
    
    @classmethod
    def func(cls, x):
        return np.exp(-x**2)
    
    @classmethod
    def deriv(cls, x):
        return -2*x*np.exp(-x**2)


if __name__ == '__main__':
    print(ReLU)