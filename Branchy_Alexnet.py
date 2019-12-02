import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model parameters
NUM_EPOCHS = 1000
BATCH_SIZE = 128
IMAGE_DIM = 32  # pixels
NUM_CLASSES = 10  # 10 classes for Cifar-10 dataset
DEVICE_IDS = [0]  # GPUs to use
OUTPUT_DIR = 'alexnet_data_out'
CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints

# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    
    def __init__(self, num_classes=NUM_CLASSES):
        """
        Define and allocate layers for this neural net.
        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()

        #main net
        self.norm = nn.LocalResponseNorm(size=3, alpha=0.00005, beta=0.75)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(64, 192, 5, padding=1)
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu = nn.ReLU()

        self.mainConvLayers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]

        #Branch 1
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
        self.branch1fc = nn.Linear(1568, 10)

        #Branch 2
        self.branch2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=3, alpha=0.00005, beta=0.75),
            nn.Conv2d(192, 32, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.branch2fc = nn.Linear(128, 10)

        #linear layers of main branch
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 2*2), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.branch1:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

        for layer in self.branch2:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

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

        #BRANCH 1: 2 3x3 conv and one FC layer
        x1 = self.branch1(x)
        x1 = x1.view(-1, 1568)
        x1 = self.branch1fc(x1)

        x = self.relu(x)
        x = self.norm(x)
        x = self.pool(x)
        x = self.conv2(x)

        #BRANCH 2: 1 3x3 conv and one FC layer
        x2 = self.branch2(x)
        x2 = x2.view(-1, 128)
        x2 = self.branch2fc(x2)

        x = self.relu(x)
        x = self.norm(x)
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(x.size(0), 256 * 2 * 2)  # reduce the dimensions for linear layer input
        x = self.classifier(x)
        return x1, x2, x


if __name__ == '__main__':
    # create model
    alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)
    # train on multiple GPUs
    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=DEVICE_IDS)
    print(alexnet)
    print('AlexNet created')

    transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(15),
            transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.CenterCrop(IMAGE_DIM),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    traindata = datasets.CIFAR10(root='./CIFAR', train=True, transform=transform, download=True)
    
    testtransform = transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    testdata = datasets.CIFAR10(root='./CIFAR', train=False, transform=testtransform, download=True)
    print('Dataset created')
    
    dataloader = data.DataLoader(
        traindata,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)
    print('Dataloader created')

    optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
    print('Optimizer created')

    # multiply LR by 1 / 10 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('LR Scheduler created')

    # start training!!
    print('Starting training...')
    alexnet.train()
    total_steps = 1
    end = False
    for epoch in range(NUM_EPOCHS):
        for imgs, classes in dataloader:
            imgs, classes = imgs.to(device), classes.to(device)
            optimizer.zero_grad()
            # calculate the loss
            output = alexnet(imgs)[-1]
            loss = F.cross_entropy(output, classes)

            # update the parameters
            loss.backward()
            optimizer.step()

            # log the information and add to tensorboard
            if total_steps % 50 == 0:
                _, preds = torch.max(output, 1)
                accuracy = torch.sum(preds == classes)

                print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                    .format(epoch + 1, total_steps, loss.item(), accuracy.item()))

            if total_steps % 300 == 0:

                #~~~~~~~VALIDATION~~~~~~~~~
                valdataloader = data.DataLoader(
                    testdata,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=8,
                    drop_last=True,
                    batch_size=128)
                correct_count = 0
                total_count = 0
                alexnet.eval()
                for images, labels in valdataloader:
                    images, labels = images.to(device), labels.to(device)
                    with torch.no_grad(): #no gradient descent!
                        logps = alexnet(images)[-1]
                    
                    for i in range(BATCH_SIZE):
                        ps = torch.exp(logps)
                        prob = list(ps.cpu().numpy()[i])
                        pred_label = prob.index(max(prob))
                        true_label = labels.cpu().numpy()[i]
                        if(true_label == pred_label):
                            correct_count += 1
                        total_count += 1
                print("Number Of Images Tested =", total_count)
                print("\nModel Accuracy =", (correct_count/total_count))
                if correct_count/total_count > 0.95:
                    end = True
                alexnet.train()
            if end:
                break

            total_steps += 1
        if end:
            break
        lr_scheduler.step()

        if epoch%100 == 9:
            if len(DEVICE_IDS) == 1:            
                torch.save(alexnet.state_dict(), CHECKPOINT_DIR+'/epoch_'+str(epoch+1)+'_model.pt')
            else:
                torch.save(alexnet.module.state_dict(), CHECKPOINT_DIR+'/epoch_'+str(epoch+1)+'_model.pt') 