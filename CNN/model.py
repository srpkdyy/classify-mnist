import chainer
import chainer.functions as F
import chainer.links as L




class CNN(chainer.Chain):

    def __init__(self, n_nodes=1000, n_out=10):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 99, ksize=3, pad=1)
            self.conv2 = L.Convolution2D(None, 297, ksize=3, pad=1)
            self.conv3 = L.Convolution2D(None, 891, ksize=3, pad=1)
            self.bn1 = L.BatchNormalization(99)
            self.bn2 = L.BatchNormalization(297)
            self.bn3 = L.BatchNormalization(891)
            self.fc1 = L.Linear(None, n_nodes)
            self.fc2 = L.Linear(None, n_nodes)
            self.fc3 = L.Linear(None, n_out)



    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.bn1(self.conv1(x))), ksize=2, stride=2)
        h = F.max_pooling_2d(F.relu(self.bn2(self.conv2(h))), ksize=2, stride=2)
        #h = F.max_pooling_2d(F.relu(self.bn3(self.conv3(h))), ksize=2, stride=2)
        h = F.dropout(h)
        h = F.dropout(F.relu(self.fc1(h)))
        #h = F.relu(self.fc1(h))
        h = F.dropout(F.relu(self.fc2(h)))
        return F.dropout(self.fc3(h))
