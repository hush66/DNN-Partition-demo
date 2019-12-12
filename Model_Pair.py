import torch.nn as nn
from config import *

####################
# Exit 1 Part 1
####################
class NetExit1Part1L(nn.Module):
    '''
    AlexNet Exit at branch 1 and part at point 1, left part
    '''
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2, stride=1)
        self.branch1 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.mainConvLayers = [self.conv1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.branch1(x)
        return x

class NetExit1Part1R(nn.Module):
    '''
    AlexNet Exit at branch 1 and part at point 1, right part
    '''
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.LocalResponseNorm(size=3, alpha=0.00005, beta=0.75),
            nn.Conv2d(64, 32, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.branch1fc = nn.Linear(1568, 10)

    def forward(self, x):
        x = self.branch1(x)
        x = x.view(-1, 1568)
        x = self.branch1fc(x)
        return x


####################
# Exit 1 Part 2
####################
class NetExit1Part2L(nn.Module):
    '''
    AlexNet Exit at branch 1 and part at point 2, left part
    '''
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2, stride=1)
        self.branch1 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=3, alpha=0.00005, beta=0.75),
            nn.Conv2d(64, 32, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.branch1(x)
        return x

class NetExit1Part2R(nn.Module):
    '''
    AlexNet Exit at branch 1 and part at point 2, right part
    '''
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.branch1fc = nn.Linear(1568, 10)

    def forward(self, x):
        x = x.view(-1, 1568)
        x = self.branch1fc(x)
        return x


####################
# Exit 2 Part 1
####################
class NetExit2Part1L(nn.Module):
    '''
    AlexNet Exit at branch 2 and part at point 1, left part
    '''
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2, stride=1)
        self.relu = nn.ReLU()
        self.norm = nn.LocalResponseNorm(size=3, alpha=0.00005, beta=0.75)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.pool(x)
        return x

class NetExit2Part1R(nn.Module):
    '''
    AlexNet Exit at branch 2 and part at point 1, right part
    '''
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv2 = nn.Conv2d(64, 192, 5, padding=1)
        self.branch2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=3, alpha=0.00005, beta=0.75),
            nn.Conv2d(192, 32, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.branch2fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv2(x)
        x = self.branch2(x)
        x = x.view(-1, 128)
        x = self.branch2fc(x)
        return x

####################
# Exit 2 Part 2
####################
class NetExit2Part2L(nn.Module):
    '''
    AlexNet Exit at branch 2 and part at point 2, left part
    '''
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2, stride=1)
        self.relu = nn.ReLU()
        self.norm = nn.LocalResponseNorm(size=3, alpha=0.00005, beta=0.75)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, 5, padding=1)

        self.branch2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.branch2(x)
        return x

class NetExit2Part2R(nn.Module):
    '''
    AlexNet Exit at branch 2 and part at point 2, right part
    '''
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.branch2 = nn.Sequential(
            nn.LocalResponseNorm(size=3, alpha=0.00005, beta=0.75),
            nn.Conv2d(192, 32, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.branch2fc = nn.Linear(128, 10)
    def forward(self, x):
        x = self.branch2(x)
        x = x.view(-1, 128)
        x = self.branch2fc(x)
        return x

####################
# Exit 2 Part 3
####################
class NetExit2Part3L(nn.Module):
    '''
    AlexNet Exit at branch 2 and part at point 3, left part
    '''
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2, stride=1)
        self.relu = nn.ReLU()
        self.norm = nn.LocalResponseNorm(size=3, alpha=0.00005, beta=0.75)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, 5, padding=1)

        self.branch2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=3, alpha=0.00005, beta=0.75),
            nn.Conv2d(192, 32, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.branch2(x)
        return x

class NetExit2Part3R(nn.Module):
    '''
    AlexNet Exit at branch 2 and part at point 2, right part
    '''
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.branch2fc = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 128)
        x = self.branch2fc(x)
        return x

####################
# Exit 3 Part 1
####################
class NetExit3Part1L(nn.Module):
    '''
    AlexNet Exit at main branch and part at point 1
    '''
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.norm = nn.LocalResponseNorm(size=3, alpha=0.00005, beta=0.75)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2, stride=1)
        self.relu = nn.ReLU()
        self.mainConvLayers = [self.conv1]

    def forward(self, x):
        x = self.conv1(x)
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
        self.conv2 = nn.Conv2d(64, 192, 5, padding=1)
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu = nn.ReLU()
        self.mainConvLayers = [self.conv2, self.conv3, self.conv4, self.conv5]

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 2 * 2), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):
        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

####################
# Exit 3 Part 2
####################
class NetExit3Part2L(nn.Module):
    '''
    AlexNet Exit at main branch and part at point 2
    '''
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.norm = nn.LocalResponseNorm(size=3, alpha=0.00005, beta=0.75)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(64, 192, 5, padding=1)
        self.relu = nn.ReLU()
        self.mainConvLayers = [self.conv1, self.conv2]

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.pool(x)
        return x

class NetExit3Part2R(nn.Module):
    '''
    AlexNet Exit at main branch and part at point 2
    '''
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.norm = nn.LocalResponseNorm(size=3, alpha=0.00005, beta=0.75)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu = nn.ReLU()
        self.mainConvLayers = [self.conv3, self.conv4, self.conv5]

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 2 * 2), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

####################
# Exit 3 Part 3
####################
class NetExit3Part3L(nn.Module):
    '''
    AlexNet Exit at main branch and part at point 3
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        return x

class NetExit3Part3R(nn.Module):
    '''
    AlexNet Exit at main branch and part at point 3
    '''
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 2 * 2), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

####################
# Model Pair
####################
NetExit1Part1 = [NetExit1Part1L, NetExit1Part1R]
NetExit1Part2 = [NetExit1Part2L, NetExit1Part2R]
NetExit2Part1 = [NetExit2Part1L, NetExit2Part1R]
NetExit2Part2 = [NetExit2Part2L, NetExit2Part2R]
NetExit2Part3 = [NetExit2Part3L, NetExit2Part3R]
NetExit3Part1 = [NetExit3Part1L, NetExit3Part1R]
NetExit3Part2 = [NetExit3Part2L, NetExit3Part2R]
NetExit3Part3 = [NetExit3Part3L, NetExit3Part3R]