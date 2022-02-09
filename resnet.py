import datetime
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from custom_dataset import Chest3Dataset, CovidDataset
from settings import IMG_SIZE, ALL_ARCHITECTURES
from model_metadata import create_model_dir, save_common_info, plot_progress, get_model_summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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


class MyResNetArgs:
   """
    Passing the hyperparameters to the model
   """
   def __init__(self, arch='resnet20' ,epochs=5, start_epoch=0, batch_size=128, lr=0.1, momentum=0.9, weight_decay=1e-4):
        
        self.weight_decay = weight_decay
        self.momentum = momentum 
        self.lr = lr #Learning rate
        self.batch_size = batch_size 
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.arch = arch #ResNet model


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



# train_loader = DataLoader(Chest3Dataset('data/chest_xray_3/train.csv', 'data/chest_xray_3', special_transform=transforms.Compose([tv.transforms.Grayscale(num_output_channels=1)]), transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(.05)])), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
# test_loader = DataLoader(Chest3Dataset('data/chest_xray_3/test.csv', 'data/chest_xray_3', special_transform=transforms.Compose([tv.transforms.Grayscale(num_output_channels=1)])), batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

# TODO: Don't forget to change this variable {DATASET} to the dataset name you are training on 
train_loader = DataLoader(CovidDataset('data/covid_data/train.csv', 'data/covid_data/train', special_transform=transforms.Compose([tv.transforms.Grayscale(num_output_channels=1)])), batch_size=256, shuffle=True, pin_memory=True)
test_loader = DataLoader(CovidDataset('data/covid_data/test.csv', 'data/covid_data/test', special_transform=transforms.Compose([tv.transforms.Grayscale(num_output_channels=1)])), batch_size=256, shuffle=False, pin_memory=True)


def main(model_path, model_name):
    best_acc = 0
    global args
    args=MyResNetArgs(model_name)
    model = globals()[model_name]()

    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)

    print('Training {} model'.format(args.arch))
    test_accs, test_losses = [], []
    train_accs, train_losses = [], []
    for epoch in range(args.start_epoch, args.epochs):

        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        train_accs.append(train_acc)
        train_losses.append(train_loss)

        # evaluate on validation set
        acc, loss = testing(test_loader, model, criterion)
        test_accs.append(acc)
        test_losses.append(loss)
        if acc > best_acc:
            best_acc = acc
            # save model 
            torch.save(model.state_dict(), f'{model_path}/model.pth')

    return {"best_acc": max(test_accs), "test_accs":test_accs, "test_losses": test_losses, "train_acss": train_accs, "train_losses": train_losses}


if __name__ == '__main__':
   models_info = {}
   for architecture in ALL_ARCHITECTURES:
       model_path = create_model_dir('results')
       get_model_summary(model_path, globals()[architecture]())
       start = datetime.datetime.now() 
       model_info = main(model_path, architecture)
       end = datetime.datetime.now()
       models_info[architecture] = model_info
       save_common_info(model_path, args, model_info, start, end)
       plot_progress(model_path, "Testing Accuracy", architecture, model_info["test_accs"])
       plot_progress(model_path, "Training Accuracy", architecture, model_info["train_acss"])
       plot_progress(model_path, "Testing Loss", architecture, model_info["test_losses"])
       plot_progress(model_path, "Training Loss", architecture, model_info["train_losses"])
