import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils import data
from config import *


def test_data():
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
    return dataiter
