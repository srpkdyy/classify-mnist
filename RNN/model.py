import cupy as cp

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, iterators
from chainer import training
from chainer.training import extensions




class RNN(chainer.Chain):

    def __init__(self, batch_size, n_units=1000, n_out=10):
        super().__init__()
        with self.init_scope():
            self.fc1 = L.Linear(None, n_units)
            self.fc2 = L.Linear(None, n_units)
            self.fc3 = L.Linear(None, n_out)
            self.w1 = L.Linear(None, n_units)
            self.w2 = L.Linear(None, n_units)

        self.z1 = cp.zeros((batch_size, n_units), dtype=cp.float32)
        self.z2 = cp.zeros((batch_size, n_units), dtype=cp.float32)

    
    def __call__(self, x):
        print(type(self.fc1(x)))
        h = F.relu(self.fc1(x) + self.w1(self.z1))
        self.z1 = h
        h = F.relu(self.fc2(h) + self.w2(self.z2))
        self.z2 = h
        return self.fc3(h)
