import torch
import numpy as np
from output_embedding import *


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#DEVICE = torch.device("cpu")


def set_tensor(tensor):
    return tensor.to(DEVICE).float()


# Transform the inputs
def to_vector(batch):
    batch_size = batch.size(0)
    return batch.reshape(batch_size, -1).squeeze()


# Transform the outputs
def one_hot(labels, n_classes=10):
    arr = torch.eye(n_classes)
    return arr[labels]


def Want_accuracy(pred_labels, true_labels):
    batch_size = pred_labels.size(0)
    correct = 0
    for b in range(batch_size):
        if torch.argmax(pred_labels[b, :]) == torch.argmax(true_labels[b, :]):
            correct += 1
    return correct / batch_size

def vector_accuracy(pred_labels, true_labels, output_vec_dic):
    batch_size = pred_labels.size(0)
    correct = 0
    for b in range(batch_size):
        pred = pred_labels[b, :].to('cpu').numpy()
        actual = true_labels[b, :].to('cpu').numpy()

        face_dist = np.sqrt(np.sum((pred - output_vec_dic["Face Vector"])**2))
        not_face_dist = np.sqrt(np.sum((pred - output_vec_dic["Not Face Vector"])**2))

        if (face_dist <= not_face_dist):
            if np.sum(np.abs(output_vec_dic["Face Vector"] - actual)) < 1:
                correct += 1
        elif (face_dist > not_face_dist):
            if np.sum(np.abs(output_vec_dic["Not Face Vector"] - actual)) < 1:
                correct += 1
    return correct / batch_size


# ====================================================================================
#
# Activation Functions
#
# ====================================================================================
class Activation(object):
    def forward(self, inputs):
        raise NotImplementedError

    def deriv(self, inputs):
        raise NotImplementedError

    def __call__(self, inputs):
        return self.forward(inputs)


class Linear(Activation):
    def forward(self, inputs):
        return inputs

    def deriv(self, inputs):
        return set_tensor(torch.ones((1,)))


class ReLU(Activation):
    def forward(self, inputs):
        return torch.relu(inputs)

    def deriv(self, inputs):
        out = self(inputs)
        out[out > 0] = 1.0
        return out


class Tanh(Activation):
    def forward(self, inputs):
        return torch.tanh(inputs)

    def deriv(self, inputs):
        return 1.0 - torch.tanh(inputs) ** 2.0

class Logistic(Activation):
    def forward(self, inputs):
        return torch.sigmoid(inputs)
    
    def deriv(self, inputs):
        return torch.sigmoid(inputs)*(1 - torch.sigmoid(inputs))