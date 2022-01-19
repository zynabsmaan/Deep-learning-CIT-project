import torchvision as tv
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from torchsummary import summary
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os 

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

IMG_SIZE = 64
all_architectures = ['resnet20', 'resnet32', 'resnet44', 'resnet56']
all_plain_architectures = ['resnet_plain_20', 'resnet_plain_32', 'resnet_plain_44', 'resnet_plain_56']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _weights_init(m):
    """
        Initialization of CNN weights
    """
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    """
      Identity mapping between ResNet blocks with diffrenet size feature map
    """
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if option == 'A':
                """
                For CIFAR10 experiment, ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_channels//4, out_channels//4), "constant", 0))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        self.linear = nn.Linear(256* block.expansion, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def get_model_summary(model):
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, model.parameters()))), "\n")
    summary(model.to(device), (1,64,64))


class MyResNetArgs:
   """
    Passing the hyperparameters to the model
   """
   def __init__(self, arch='resnet20' ,epochs=100, start_epoch=0, batch_size=128, lr=0.1, momentum=0.9, weight_decay=1e-4):
        
        self.weight_decay = weight_decay
        self.momentum = momentum 
        self.lr = lr #Learning rate
        self.batch_size = batch_size 
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.arch = arch #ResNet model


if __name__ == "__main__":
    for model_name in all_architectures:
        if model_name.startswith('resnet'):
            print(model_name)
            get_model_summary(globals()[model_name]())


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):

        input_var = input.float().cuda()
        target_var = target.cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        acc = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(acc.item(), input.size(0))

    print(f'Epoch: [{epoch}]\t Loss {losses.val:.4f} ({losses.avg:.4f})\t acc {top1.val:.3f} ({top1.avg:.3f})')
    return top1.avg, losses.avg
       

def testing(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var = input.float().cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss and acc
            acc = accuracy(output.data, target_var)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(acc.item(), input.size(0))


    print(f'Test\t  accuracy: {top1.avg:.3f} (Err: {losses.avg:.3f} )\n')

    return top1.avg, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=1):
    batch_size = target.size(0)

    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True)
    res.append(correct_k.mul_(100.0 / batch_size))
    return res


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, special_transform = None, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.special_transform = special_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):       
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        image = fn.resize(image, size=[IMG_SIZE, IMG_SIZE])
        label = self.img_labels.iloc[idx, 1]
        if image.shape[0] > 1:
            image = self.special_transform(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


train_loader = DataLoader(CustomImageDataset('train.csv', '.', special_transform=transforms.Compose([tv.transforms.Grayscale(num_output_channels=1)]), transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(.05)])), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

test_loader = DataLoader(CustomImageDataset('test.csv', '.', special_transform=transforms.Compose([tv.transforms.Grayscale(num_output_channels=1)])), batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

import datetime

datetime.datetime.now()




def plot_progress(mode, model_name, data):
    sns.set_theme(style="darkgrid")
    sns.lineplot(list(range(1, len(data)+1, 1)), data)
    plt.title(f"{mode} progress for {model_name} model")
    plt.xlabel("epochs", fontsize=14)
    plt.ylabel(mode, fontsize=14)
    plt.show()
    plt.close()


def main(model_name):
    best_acc = 0
    global args
    args=MyResNetArgs(model_name)
    model = globals()[model_name]()

    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), .1,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    print('Training {} model'.format(args.arch))
    test_accs, test_losses = [], []
    train_accs, train_losses = [], []
    for epoch in range(args.start_epoch, args.epochs):

        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        # evaluate on validation set
        acc, loss = testing(test_loader, model, criterion)
        test_accs.append(acc)
        test_losses.append(loss)

    return {"best_acc": max(test_accs), "test_accs":test_accs, "test_losses": test_losses, "train_acss": train_accs, "train_losses": train_losses}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if __name__ == '__main__':
   models_info = {}
   for architecture in all_architectures:
       start = datetime.datetime.now() 
       model_info = main(architecture)
       end = datetime.datetime.now()
       print("The start time is {} and end time {}".format(start, end))
       models_info[architecture] = model_info
       print('The higher accuracy of {} model after {} epochs is {acc:.3f}'.format(args.arch,args.epochs, acc=model_info["best_acc"]))
       plot_progress("Testing Accuracy", architecture, model_info["test_accs"])
       plot_progress("Training Accuracy", architecture, model_info["train_acss"])
       plot_progress("Testing Loss", architecture, model_info["test_losses"])
       plot_progress("Training Loss", architecture, model_info["train_losses"])


