from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch
from torch import nn,optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

# I want this program to run on CUDA.
# However, runs without it. How come?

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default='False')
    args = parser.parse_args()

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    print(args.no_cuda)
    print(torch.cuda.is_available())
    print(use_cuda)
    print(args)

    # idk why this line dosen't run correctly. tell me.
    #device = torch.device('cuda' if use_cuda else 'cpu')
    device = torch.device('cuda')
    print(device)

    ds_train = datasets.MNIST('../dataset', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))
    
    ds_test = datasets.MNIST('../dataset', train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))
                                                           
    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = DataLoader(ds_train, batch_size=64, shuffle=True,  **kwargs)
    test_loader = DataLoader(ds_test, batch_size=64, shuffle=False, **kwargs)

    #disp_ds(ds_train, 10)

    model = nn.Sequential()
    model.add_module('view1', View((-1, 28*28)))
    model.add_module('fc1', nn.Linear(28*28*1, 100))
    model.add_module('relu1', nn.ReLU())
    model.add_module('fc2', nn.Linear(100, 100))
    model.add_module('relu2', nn.ReLU())
    model.add_module('fc3', nn.Linear(100, 10))
    model.to(device)

    #cudnn.benchmark = True
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    for epoch in tqdm(range(1, 11)):
        train(model, device, train_loader, loss_fn, optimizer, epoch)
        test(model, device, test_loader, loss_fn)

 
def get_mnsit():
    return fetch_openml('MNIST original', data_home='./datasets/')


def disp_ds(ds, num=None):
    if num is None: num = 100
    fig = plt.figure()

    ims = []
    for i in range(num):
        img = ds[i][0].numpy().reshape(28, 28)
        ims.append([plt.imshow(img)])

    animation = anim.ArtistAnimation(fig, ims, interval=100, repeat=False)
    plt.show()
    

def train(model, device, train_loader, loss_fn, optimizer, epoch):
    model.train()
    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    #print('epoch:{} Loss:{}'.format(epoch, loss))


def test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += loss_fn(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum()
    #test_loss /= len(test_loader.dataset)
    print('accracy : {:.0f}%'.format(100. * correct / len(test_loader.dataset)))

'''
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
'''


if __name__ == '__main__':
    main()
