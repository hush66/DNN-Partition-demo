import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import scipy.io as io
import numpy as np

from torch.utils import data

NUM_CLASSES = 10
IMAGE_DIM = 32

class NetExit3Part1L(nn.Module):
    '''
    AlexNet Exit at main branch and part at point 1
    '''
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.norm = nn.LocalResponseNorm(size=3, alpha=0.00005, beta=0.75)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(64, 192, 5, padding=1)
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu = nn.ReLU()
        self.mainConvLayers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 2 * 2), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        self.init_bias()

    def init_bias(self):
        for layer in self.mainConvLayers:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            nn.init.constant_(layer.bias, 0)

        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.conv2.bias, 1)
        nn.init.constant_(self.conv4.bias, 1)
        nn.init.constant_(self.conv5.bias, 1)

    def forward(self, x):
        """
        Pass the input through the net.
        Args:
            x (Tensor): input tensor
        Returns:
            output (Tensor): output tensor
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.pool(x)
        return x

class NetExit3Part1R(nn.Module):
    '''
    AlexNet Exit at main branch and part at point 1
    '''
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.norm = nn.LocalResponseNorm(size=3, alpha=0.00005, beta=0.75)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(64, 192, 5, padding=1)
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu = nn.ReLU()
        self.mainConvLayers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 2 * 2), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        self.init_bias()

    def init_bias(self):
        for layer in self.mainConvLayers:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            nn.init.constant_(layer.bias, 0)

        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.conv2.bias, 1)
        nn.init.constant_(self.conv4.bias, 1)
        nn.init.constant_(self.conv5.bias, 1)

    def forward(self, x):
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(x.size(0), 256 * 2 * 2)  # reduce the dimensions for linear layer input
        x = self.classifier(x)
        return x



if __name__ == '__main__':
    netL = NetExit3Part1L()
    netR = NetExit3Part1R()
    Lmodel_dict = netL.state_dict()
    Rmodel_dict = netR.state_dict()
    params = torch.load('./alexnet_data_out/models/epoch_910_model.pt', map_location=torch.device('cpu'))
    # filter out needed keys
    Lloaded_dict = {k: v for k, v in params.items() if k in Lmodel_dict}
    Rloaded_dict = {k: v for k, v in params.items() if k in Rmodel_dict}
    # overwrite entries in the existing state dict
    Lmodel_dict.update(Lloaded_dict)
    Rmodel_dict.update(Rloaded_dict)
    netL.load_state_dict(Lmodel_dict)
    netR.load_state_dict(Rmodel_dict)

    netL.eval()
    netR.eval()

    # testdata
    testtransform = transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    testdata = datasets.CIFAR10(root='./CIFAR', train=False, transform=testtransform, download=True)

    valdataloader = data.DataLoader(
        testdata,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=1)

    dataiter = iter(valdataloader)
    images, labels = dataiter.next()
    outL = netL(images)
    intermediate = outL.detach().numpy()
    np.save('intermediate.npy', intermediate)
    readed = np.load('intermediate.npy')
    newOutL = torch.from_numpy(readed)
    outR = netR(newOutL)
    print(outR.size())





