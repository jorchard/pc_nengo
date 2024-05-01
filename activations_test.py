import numpy as np
from activations import *
import unittest


class TestLinearMethods(unittest.TestCase):
    def setUp(self):
        self.num = np.random.randn()
        self.size = np.random.randint(2, 100)
        self.arr = np.random.randn(self.size)

    def test_func(self):
        self.assertEqual(self.num, Linear.func(self.num))
        self.assertTrue(np.all(self.arr == Linear.func(self.arr)))

    def test_deriv(self):
        self.assertEqual(1, Linear.deriv(self.num))
        self.assertTrue(np.all(np.ones_like(self.size) == Linear.deriv(self.arr)))


class TestReLUMethods(unittest.TestCase):
    def setUp(self):
        self.num = np.random.rand() #nonnegative
        self.neg_num = -self.num
        self.arr = np.arange(-2, 2)

    def test_func(self):
        self.assertEqual(self.num, ReLU.func(self.num))
        self.assertEqual(0, ReLU.func(self.neg_num))
        self.assertTrue(np.all(np.array([0, 0, 0, self.arr[-1]]) == ReLU.func(self.arr)))

    def test_deriv(self):
        self.assertEqual(1, ReLU.deriv(self.num))
        self.assertEqual(0, ReLU.deriv(0))
        self.assertEqual(0, ReLU.deriv(self.neg_num))
        self.assertTrue(np.all(np.array([0, 0, 0, 1]) == ReLU.deriv(self.arr)))


class TestLogisticMethods(unittest.TestCase):
    def test_func(self):
        self.assertEqual(0.5, Logistic.func(0))
        self.assertAlmostEqual(1, Logistic.func(25))
        self.assertAlmostEqual(0, Logistic.func(-25))

    def test_deriv(self):
        self.assertEqual(0.25, Logistic.deriv(0))
        self.assertAlmostEqual(0, Logistic.deriv(25))
        self.assertAlmostEqual(0, Logistic.deriv(-25))


class TestTanhMethods(unittest.TestCase):
    def test_func(self):
        self.assertEqual(0, Tanh.func(0))
        self.assertAlmostEqual(1, Tanh.func(25))
        self.assertAlmostEqual(-1, Tanh.func(-25))

    def test_deriv(self):
        self.assertEqual(1, Tanh.deriv(0))
        self.assertAlmostEqual(0, Tanh.deriv(25))
        self.assertAlmostEqual(0, Tanh.deriv(-25))



class TestThresholdMethods(unittest.TestCase):
    def setUp(self):
        self.num = np.random.rand() + 1 #positive
        self.neg_num = -self.num
        self.arr = np.arange(-2, 2)

    def test_func(self):
        self.assertEqual(1, Threshold.func(self.num))
        self.assertEqual(0, Threshold.func(0))
        self.assertEqual(0, Threshold.func(self.neg_num))
        self.assertTrue(np.all(np.array([0, 0, 0, 1]) == Threshold.func(self.arr)))

    def test_deriv(self):
        self.assertEqual(0, Threshold.deriv(self.num))
        self.assertEqual(0, Threshold.deriv(0))
        self.assertEqual(0, Threshold.deriv(self.neg_num))
        self.assertTrue(np.all(np.zeros_like(self.arr) == Threshold.deriv(self.arr)))



class TestSoftplusMethods(unittest.TestCase):
    def test_func(self):
        self.assertAlmostEqual(np.log(2), Softplus.func(0))
        self.assertAlmostEqual(0, Softplus.func(-100))
        self.assertAlmostEqual(100, Softplus.func(100))

    def test_deriv(self):
        self.assertEqual(0.5, Softplus.deriv(0))
        self.assertAlmostEqual(1, Softplus.deriv(25))
        self.assertAlmostEqual(0, Softplus.deriv(-25))



class TestLeakyReLUMethods(unittest.TestCase):
    def setUp(self):
        self.num = np.random.rand() + 1 #positive
        self.neg_num = -self.num
        self.arr = np.array([-2., -1., 0., 1., 2.])

    def test_func(self):
        self.assertEqual(self.num, LeakyReLU.func(self.num))
        self.assertEqual(self.neg_num*0.01, LeakyReLU.func(self.neg_num))
        self.assertEqual(0, LeakyReLU.func(0))
        self.assertTrue(np.all(np.array([-0.02, -0.01, 0, 1, 2]) == LeakyReLU.func(self.arr)))

    def test_deriv(self):
        self.assertEqual(1, LeakyReLU.deriv(self.num))
        self.assertEqual(0.01, LeakyReLU.deriv(self.neg_num))
        self.assertEqual(0.01, LeakyReLU.deriv(0))
        self.assertTrue(np.all(np.array([0.01, 0.01, 0.01, 1, 1]) == LeakyReLU.deriv(self.arr)))


class TestGaussianMethods(unittest.TestCase):
    def setUp(self):
        self.num = np.random.rand() + 1 #positive
        self.neg_num = -self.num
        self.arr = np.array([-1, 0, 1])

    def test_func(self):
        self.assertEqual(Gaussian.func(self.num), Gaussian.func(self.neg_num))
        self.assertTrue(np.all(np.array([1/np.e, 1, 1/np.e]) == Gaussian.func(self.arr)))

    def test_deriv(self):
        self.assertEqual(-Gaussian.deriv(self.num), Gaussian.deriv(self.neg_num))
        self.assertEqual(0, Gaussian.deriv(0))
        self.assertTrue(np.all(np.array([2/np.e, 0, -2/np.e]) == Gaussian.deriv(self.arr)))


if __name__ == '__main__':
    unittest.main()