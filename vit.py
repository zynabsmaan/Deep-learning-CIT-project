import datetime, glob
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils import AverageMeter
from custom_dataset import Chest3Dataset, CovidDataset
from settings import IMG_SIZE, NUM_CLASSES, NUM_LAYERS_LIST, NUM_HEADS_LIST
from model_metadata import create_model_dir, save_common_info, plot_progress, get_model_summary, save_custom_info

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class MyResNetArgs:
   """
    Passing the hyperparameters to the model
   """
   def __init__(self,epochs=100, start_epoch=0, batch_size=64, lr=0.1, momentum=0.9, weight_decay=1e-4):
        
        self.weight_decay = weight_decay
        self.momentum = momentum 
        self.lr = lr 
        self.batch_size = batch_size 
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.arch = 'ViT'
        self.embedding_dim = 128
       

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

def accuracy(output, target, topk=1):
    batch_size = target.size(0)

    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True)
    res.append(correct_k.mul_(100.0 / batch_size))
    return res

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



# train_loader = DataLoader(Chest3Dataset('data/chest_xray_3/train.csv', 'data/chest_xray_3', special_transform=transforms.Compose([tv.transforms.Grayscale(num_output_channels=1)]), transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(.05)])), batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
# test_loader = DataLoader(Chest3Dataset('data/chest_xray_3/test.csv', 'data/chest_xray_3', special_transform=transforms.Compose([tv.transforms.Grayscale(num_output_channels=1)])), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

train_loader = DataLoader(CovidDataset('data/covid_data/train.csv', 'data/covid_data/train', special_transform=transforms.Compose([tv.transforms.Grayscale(num_output_channels=1)])), batch_size=256, shuffle=True, pin_memory=True)
test_loader = DataLoader(CovidDataset('data/covid_data/test.csv', 'data/covid_data/test', special_transform=transforms.Compose([tv.transforms.Grayscale(num_output_channels=1)])), batch_size=256, shuffle=False, pin_memory=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def main(model_path, model_name):
    best_acc = 0
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
    global args
    args=MyResNetArgs()
    for depth in NUM_LAYERS_LIST: 
        for heads in NUM_HEADS_LIST:
            model = ViT(image_size = IMG_SIZE, patch_size = args.batch_size, num_classes = NUM_CLASSES, dim = args.embedding_dim, depth = depth, heads = heads, mlp_dim = 256, dropout = 0.1, emb_dropout = 0.1)
            model_path = create_model_dir('covid_vit_results')
            get_model_summary(model_path, model)
            start = datetime.datetime.now() 
            model_info = main(model_path, args.arch)
            end = datetime.datetime.now()
            save_common_info(model_path, args, model_info, start, end)
            plot_progress(model_path, "Testing Accuracy", args.arch, model_info["test_accs"])
            plot_progress(model_path, "Training Accuracy", args.arch, model_info["train_acss"])
            plot_progress(model_path, "Testing Loss", args.arch, model_info["test_losses"])
            plot_progress(model_path, "Training Loss", args.arch, model_info["train_losses"])
