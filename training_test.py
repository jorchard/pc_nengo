import numpy as np
from training import *
import unittest

class TestCreateLearningInputs(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(3*8).reshape((8, 3))
        self.Y = np.arange(2*8).reshape((8, 2))
        self.indices = np.arange(8)

    def test_no_shuffle(self):
        x_dict, y_dict, t = create_learning_inputs(self.X, self.Y, 1, 0.5, 0.5, False)

        x_dict_test = {
            0: np.array([0, 1, 2]),
            1: np.array([3, 4, 5]),
            2: np.array([6, 7, 8]),
            3: np.array([9, 10, 11]),
            4: np.array([12, 13, 14]),
            5: np.array([15, 16, 17]),
            6: np.array([18, 19, 20]),
            7: np.array([21, 22, 23])
        }
        y_dict_test = {
            0: np.array([0, 1]),
            1: np.array([2, 3]),
            2: np.array([4, 5]),
            3: np.array([6, 7]),
            4: np.array([8, 9]),
            5: np.array([10, 11]),
            6: np.array([12, 13]),
            7: np.array([14, 15])
        }

        #8 data points, 1s time, 1 epoch
        self.assertEqual(8, t)
        
        for key in x_dict.keys():
            self.assertTrue(np.all(x_dict[key] == x_dict_test[key]))
        for key in y_dict.keys():
            self.assertTrue(np.all(y_dict[key] == y_dict_test[key]))

    def test_shuffle(self):
        np.random.seed(1)
        x_dict, y_dict, t = create_learning_inputs(self.X, self.Y, 1, 0.25, 0.75, True)

        x_dict_test = {
            0: np.array([21, 22, 23]),
            1: np.array([6, 7, 8]),
            2: np.array([3, 4, 5]),
            3: np.array([18, 19, 20]),
            4: np.array([0, 1, 2]),
            5: np.array([12, 13, 14]),
            6: np.array([9, 10, 11]),
            7: np.array([15, 16, 17])
        }
        y_dict_test = {
            0: np.array([14, 15]),
            1: np.array([4, 5]),
            2: np.array([2, 3]),
            3: np.array([12, 13]),
            4: np.array([0, 1]),
            5: np.array([8, 9]),
            6: np.array([6, 7]),
            7: np.array([10, 11])
        }

        #8 data points, 1s time, 1 epoch
        self.assertEqual(8, t)
        
        for key in x_dict.keys():
            self.assertTrue(np.all(x_dict[key] == x_dict_test[key]))
        for key in y_dict.keys():
            self.assertTrue(np.all(y_dict[key] == y_dict_test[key]))


    def test_no_stab_time(self):
        np.random.seed(1)
        x_dict, y_dict, t = create_learning_inputs(self.X, self.Y, 1, 0, 1, True)

        x_dict_test = {
            0: np.array([21, 22, 23]),
            1: np.array([6, 7, 8]),
            2: np.array([3, 4, 5]),
            3: np.array([18, 19, 20]),
            4: np.array([0, 1, 2]),
            5: np.array([12, 13, 14]),
            6: np.array([9, 10, 11]),
            7: np.array([15, 16, 17])
        }
        y_dict_test = {
            0: np.array([14, 15]),
            1: np.array([4, 5]),
            2: np.array([2, 3]),
            3: np.array([12, 13]),
            4: np.array([0, 1]),
            5: np.array([8, 9]),
            6: np.array([6, 7]),
            7: np.array([10, 11])
        }

        #8 data points, 1s time, 1 epoch
        self.assertEqual(8, t)
        
        for key in x_dict.keys():
            self.assertTrue(np.all(x_dict[key] == x_dict_test[key]))
        for key in y_dict.keys():
            self.assertTrue(np.all(y_dict[key] == y_dict_test[key]))


    def test_no_learn_time(self):
        np.random.seed(1)
        x_dict, y_dict, t = create_learning_inputs(self.X, self.Y, 1, 1, 0, True)

        x_dict_test = {
            0: np.array([21, 22, 23]),
            1: np.array([6, 7, 8]),
            2: np.array([3, 4, 5]),
            3: np.array([18, 19, 20]),
            4: np.array([0, 1, 2]),
            5: np.array([12, 13, 14]),
            6: np.array([9, 10, 11]),
            7: np.array([15, 16, 17])
        }
        y_dict_test = {
            0: np.array([14, 15]),
            1: np.array([4, 5]),
            2: np.array([2, 3]),
            3: np.array([12, 13]),
            4: np.array([0, 1]),
            5: np.array([8, 9]),
            6: np.array([6, 7]),
            7: np.array([10, 11])
        }

        #8 data points, 1s time, 1 epoch
        self.assertEqual(8, t)
        
        for key in x_dict.keys():
            self.assertTrue(np.all(x_dict[key] == x_dict_test[key]))
        for key in y_dict.keys():
            self.assertTrue(np.all(y_dict[key] == y_dict_test[key]))


    def test_zero_epochs(self):
        np.random.seed(1)
        x_dict, y_dict, t = create_learning_inputs(self.X, self.Y, 0, 0.25, 0.75, True)

        self.assertEqual(0, t)
        self.assertEqual(x_dict, {})
        self.assertEqual(y_dict, {})


class TestAddInferenceInputs(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(3*8).reshape((8, 3))
        self.Y = np.arange(2*8).reshape((8, 2))
        
    def test_empty_dicts(self):
        x_dic = {}
        y_dic = {}
        timeout = 1
        stab_time = 1
        learn_until = 10
        num_predictors = add_inference_inputs(self.X, self.Y, x_dic, y_dic, timeout, stab_time, learn_until)
        self.assertEqual(num_predictors, self.Y.shape[0])


    def test_mismatch_size(self):
        x_dic = {}
        y_dic = {}
        timeout = 1
        stab_time = 1
        learn_until = 10
        with self.assertRaises(IndexError):
            num_predictors = add_inference_inputs(self.X, np.array([1, 2]), x_dic, y_dic, timeout, stab_time, learn_until)


    def test_empty_inputs(self):
        x_dic = {}
        y_dic = {}
        timeout = 1
        stab_time = 1
        learn_until = 10
        num_predictors = add_inference_inputs(np.array([]), np.array([]), x_dic, y_dic, timeout, stab_time, learn_until)
        self.assertEqual(0, num_predictors)
        self.assertEqual(0, len(x_dic.keys()))
        self.assertEqual(0, len(y_dic.keys()))

    def test_no_timeout(self):
        x_dic = {}
        y_dic = {}
        timeout = 0
        stab_time = 1
        learn_until = 10
        with self.assertRaises(ValueError):
            num_predictors = add_inference_inputs(self.X, self.Y, x_dic, y_dic, timeout, stab_time, learn_until)
    
    def test_no_stab_time(self):
        x_dic = {}
        y_dic = {}
        timeout = 1
        stab_time = 0
        learn_until = 10
        num_predictors = add_inference_inputs(self.X, self.Y, x_dic, y_dic, timeout, stab_time, learn_until)

        for index, key in enumerate([learn_until + k*timeout for k in range(num_predictors)]):
            self.assertTrue(np.all(x_dic[key] == self.X[index,:]))
            self.assertTrue(np.all(y_dic[key] == self.Y[index,:]))


if __name__ == '__main__':
    #np.random.seed(1)
    #t = np.arange(8)
    #np.random.shuffle(t)
    #print(t)
    #print(np.arange(3*8).reshape((8, 3)))
    #print(np.arange(2*8).reshape((8, 2)))
    unittest.main()