import chainer
import chainer.links as L
import chainer.functions as F




class AlexNet(chainer.Chain):
    def __init__(self, n_units=4096, n_out=10):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 96, ksize=4, stride=2, pad=2)
            self.conv2 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
            self.conv3 = L.Convolution2D(None, 384, ksize=3, stride=1, pad=1)
            self.conv4 = L.Convolution2D(None, 384, ksize=3, stride=1, pad=1)
            self.conv5 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
            self.fc1 = L.Linear(None, n_units)
            self.fc2 = L.Linear(None, n_units)
            self.fc3 = L.Linear(None, n_out)


    def __call__(self, x):
        h = F.max_pooling_2d(self.conv1(x), ksize=3, stride=2)
        h = F.max_pooling_2d(self.conv2(h), ksize=3, stride=2)
        h = self.conv3(h)
        h = self.conv4(h)
        h = F.max_pooling_2d(self.conv5(h), ksize=3, stride=2)
        h = F.dropout(F.relu(self.fc1(h)))
        h = F.dropout(F.relu(self.fc2(h)))
        return self.fc3(h)
