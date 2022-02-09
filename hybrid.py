from ast import arg
import glob, datetime
import torchvision as tv
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_dataset import Chest3Dataset, CovidDataset
from settings import IMG_SIZE, N_INPUT_CHANNELS, NUM_CLASSES
from model_metadata import create_model_dir, save_common_info, plot_progress, get_model_summary, save_custom_info
from utils import AverageMeter
# Modules
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1  = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1    = nn.LayerNorm(d_model)
        self.linear2  = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        super(Tokenizer, self).__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=conv_bias),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return self.flattener(self.conv_layers(x)).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class TransformerClassifier(nn.Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=10,
                 dropout_rate=0.1,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.1,
                 sequence_length=None,
                 *args, **kwargs):
        super().__init__()
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert sequence_length is not None, \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim),
                                          requires_grad=True)
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        self.dropout = nn.Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = nn.LayerNorm(embedding_dim)

        self.fc = nn.Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        if x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Hybrid(nn.Module):
    def __init__(self,
                 img_size=64,
                 embedding_dim=128,
                 n_input_channels=1,
                 n_conv_layers=2,
                 kernel_size=3,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 *args, **kwargs):
        super(Hybrid, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout_rate=0.1,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs)

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)

class MyResNetArgs:
   """
    Passing the hyperparameters to the model
   """
   def __init__(self,epochs=5, start_epoch=0, batch_size=128, lr=0.1, momentum=0.9, weight_decay=1e-4, n_conv_layers=1, kernel_size=3,
   stride = 2, padding = 1, pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, num_layers=2, num_heads=2):
        
        self.weight_decay = weight_decay
        self.momentum = momentum 
        self.lr = lr 
        self.batch_size = batch_size 
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.arch = 'hybrid'
        self.embedding_dim = 128
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.pooling_kernel_size=pooling_kernel_size
        self.pooling_stride=pooling_stride
        self.pooling_padding=pooling_padding

        self.num_layers=num_layers
        self.num_heads=num_heads

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

def main(model_path, model):
    best_acc = 0

    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_loader = DataLoader(Chest3Dataset('data/chest_xray_3/train.csv', 'data/chest_xray_3', special_transform=transforms.Compose([tv.transforms.Grayscale(num_output_channels=1)]), transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(.05)])), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(Chest3Dataset('data/chest_xray_3/test.csv', 'data/chest_xray_3', special_transform=transforms.Compose([tv.transforms.Grayscale(num_output_channels=1)])), batch_size=128, shuffle=False, num_workers=2, pin_memory=True)


# train_loader = DataLoader(CovidDataset('data/covid_data/train.csv', 'data/covid_data/train', special_transform=transforms.Compose([tv.transforms.Grayscale(num_output_channels=1)])), batch_size=256, shuffle=True, pin_memory=True)
# test_loader = DataLoader(CovidDataset('data/covid_data/test.csv', 'data/covid_data/test', special_transform=transforms.Compose([tv.transforms.Grayscale(num_output_channels=1)])), batch_size=256, shuffle=False, pin_memory=True)


if __name__ == '__main__':
    models_info = {}

    global args
    args=MyResNetArgs(kernel_size=3, n_conv_layers=2)
    model = Hybrid(
        img_size=IMG_SIZE,
        embedding_dim=args.embedding_dim,
        n_input_channels=N_INPUT_CHANNELS,
        n_conv_layers=args.n_conv_layers,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        pooling_kernel_size=args.pooling_kernel_size,
        pooling_stride=args.pooling_stride,
        pooling_padding=args.pooling_padding,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_classes=NUM_CLASSES)

    model_path = create_model_dir('hybrid_results')
    get_model_summary(model_path, model)
    start = datetime.datetime.now() 
    model_info = main(model_path, model)
    end = datetime.datetime.now()
    save_common_info(model_path, args, model_info, start, end)
    save_custom_info(model_path, args)
    plot_progress(model_path, "Training Accuracy", "hybrid", model_info["train_acss"])
    plot_progress(model_path, "Testing Accuracy", "hybrid", model_info["test_accs"])
    plot_progress(model_path, "Testing Loss", "hybrid", model_info["test_losses"])
    plot_progress(model_path, "Training Loss", "hybrid", model_info["train_losses"])
    

