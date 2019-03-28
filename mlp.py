import matplotlib; matplotlib.use('agg'); del matplotlib

import numpy as np
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import iterators, optimizers
from chainer.training import extensions




class MLP(chainer.Chain):

    def __init__(self, n_nodes=1000, n_out=10):
        super().__init__()
        with self.init_scope():
            self.fc1 = L.Linear(None, n_nodes)
            self.fc2 = L.Linear(n_nodes, n_nodes)
            self.fc3 = L.Linear(n_nodes, n_out)


    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=100)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--nodes', '-n', type=int, default=1000)
    parser.add_argument('--out', '-o', default='result')
    args = parser.parse_args()

    model = L.Classifier(MLP(args.nodes))

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist()

    train_itr = iterators.SerialIterator(train, args.batch_size)
    test_itr = iterators.SerialIterator(test, args.batch_size, repeat=False, shuffle=False)

    updater = chainer.training.updaters .StandardUpdater(train_itr, optimizer)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_itr, model))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()




if __name__ == '__main__':
    main()

