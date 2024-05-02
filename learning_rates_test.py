import numpy as np
from learning_rates import *
import unittest


class TestConstRate(unittest.TestCase):
    def setUp(self):
        self.time = np.random.rand()
        self.tau = np.random.rand()
        self.Rate = ConstRate(self.tau)

    def test_call(self):
        self.assertEqual(self.tau, self.Rate(self.time))


class TestLinearRate(unittest.TestCase):
    def setUp(self):
        self.time = 1
        self.initial_tau = 0
        self.final_tau = 1

        self.Rate = LinearRate(self.initial_tau, self.final_tau, self.time)

        #number to evaluate at
        self.num = np.random.rand()
    
    def test_call(self):
        self.assertEqual(self.num, self.Rate(self.num))


class TestPowerRate(unittest.TestCase):
    def test_increasing(self):
        self.initial_tau = np.random.rand()
        self.final_tau = self.initial_tau + np.random.rand()
        self.beta = np.random.rand()
        self.time = np.random.rand()

        self.Rate = PowerRate(self.initial_tau, self.final_tau, self.beta, self.time)

        self.assertAlmostEqual(self.initial_tau, self.Rate(0))
        self.assertAlmostEqual(self.final_tau, self.Rate(self.time))
    
    def test_decreasing(self):
        self.initial_tau = np.random.rand() + 1
        self.final_tau = self.initial_tau - np.random.rand()
        self.beta = -np.random.rand()
        self.time = np.random.rand()

        self.Rate = PowerRate(self.initial_tau, self.final_tau, self.beta, self.time)

        self.assertAlmostEqual(self.initial_tau, self.Rate(0))
        self.assertAlmostEqual(self.final_tau, self.Rate(self.time))

    def test_input_validation(self):
        self.initial_tau = np.random.rand()
        self.final_tau = self.initial_tau - np.random.rand()
        self.beta = np.random.rand()
        self.time = np.random.rand()

        with self.assertRaises(ValueError):
            self.Rate = PowerRate(self.initial_tau, self.final_tau, self.beta, self.time)

        self.initial_tau = np.random.rand()
        self.final_tau = self.initial_tau + np.random.rand()
        self.beta = -np.random.rand()
        self.time = np.random.rand()

        with self.assertRaises(ValueError):
            self.Rate = PowerRate(self.initial_tau, self.final_tau, self.beta, self.time)


if __name__ == '__main__':
    unittest.main()
