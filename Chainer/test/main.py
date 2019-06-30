import argparse

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import iterators, optimizers
from chainer import training
from chainer.training import extensions

class Net(chainer.Chain):
    def __init__(self):
        super(Net, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 20, 5, 1)
            self.conv2 = L.Convolution2D(20, 50, 5, 1)
            self.fc1 = L.Linear(None, 500)
            self.fc2 = L.Linear(500, 10)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.fc1(h))
        return self.fc2(h)




def main():
    parser = argparse.ArgumentParser(description='AlexNet for MNIST')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', type=str, default='result')
    args = parser.parse_args()

    train_ds, test_ds = chainer.datasets.get_mnist(ndim=3)
    train_iter = iterators.SerialIterator(train_ds, args.batch_size)
    test_iter = iterators.SerialIterator(test_ds, 1000, repeat=False, shuffle=False)

    model = L.Classifier(Net())
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
    
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.5)
    optimizer.setup(model)

    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()




if __name__ == '__main__':
    main()
