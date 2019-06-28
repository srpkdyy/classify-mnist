import argparse

import chainer
import chainer.links as L
from chainer import optimizers, iterators
from chainer import training
from chainer.training import extensions

from model import RNN




def main():
    parser = argparse.ArgumentParser(description='CNN for MNIST')
    parser.add_argument('--batch_size', '-b', type=int, default=100, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--nodes', '-n', type=int, default=1000, help='')
    parser.add_argument('--out', '-o', default='result')
    args = parser.parse_args()

    train_ds, test_ds = chainer.datasets.get_mnist()
    
    train_itr = iterators.SerialIterator(train_ds, args.batch_size)
    test_itr = iterators.SerialIterator(test_ds, args.batch_size, repeat=False, shuffle=False)
    
    rnn = L.Classifier(RNN(args.batch_size))
    if args.gpu >= 0:
        rnn.to_gpu(args.gpu)

    optimizer = optimizers.Adam()
    optimizer.setup(rnn)

    updater = training.updaters.StandardUpdater(train_itr, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_itr, rnn, device=args.gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    
    trainer.run()
        

    

if __name__ == '__main__':
    main()
