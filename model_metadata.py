import os, re, sys

import torch
from torchsummary import summary
import seaborn as sns
import matplotlib.pyplot as plt

from settings import IMG_SIZE, DATASET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_progress(model_path, mode, model_name, data):
    sns.set_theme(style="darkgrid")
    sns.lineplot(x = list(range(1, len(data)+1, 1)), y = data)
    plt.title(f"{mode} progress for {model_name} model")
    plt.xlabel("epochs", fontsize=14)
    plt.ylabel(mode, fontsize=14)
    plt.savefig(f'{model_path}/{mode.replace(" ", "")}.png')
    plt.close()

def create_model_dir(folder_name):
    dir_list = os.listdir(folder_name)
    if not dir_list:
        model_path = f"{folder_name}/model_1"
        os.mkdir(model_path)
        return model_path
    else:
        next_number = max([int(re.findall(r'[0-9]+', dir)[0]) for dir in dir_list]) + 1
        model_path = f"{folder_name}/model_{next_number}"
        os.mkdir(model_path)
        return model_path

def save_common_info(model_path, args, model_info, start, end):
    with open(f'{model_path}/save_info.txt', 'a+') as f:
        s = f"""The dataset is: {DATASET}\nThe start time is {start} and end time {end}\nArchitecture: {args.arch}\nlearning rate: {args.lr}\nepoch: {args.epochs}\nweight_decay: {args.weight_decay}\nMomentum: {args.momentum}\n
        batch_size: {args.batch_size}\nTesting Accuracy: {model_info["test_accs"]}\nTraining Accuracy": {model_info["train_acss"]}\nTesting Loss":
         {model_info["test_losses"]}\nTraining Loss": {model_info["train_losses"]}\nThe higher accuracy: {model_info["best_acc"]}"""
        f.write(s)

def save_custom_info(model_path, args):
    with open(f'{model_path}/save_info.txt', 'a+') as f:
        s = f"""\nembedding_dim: {args.embedding_dim}\n
                n_conv_layers: {args.n_conv_layers}\n
                kernel_size: {args.kernel_size}\n
                stride: {args.stride}\n
                padding: {args.padding}\n
                pooling_kernel_size: {args.pooling_kernel_size}\n
                pooling_stride: {args.pooling_stride}\n
                pooling_padding: {args.pooling_padding}\n
                num_layers: {args.num_layers}\n
                num_heads: {args.num_heads}\n
                mlp_radio: {args.mlp_radio}
        """
        f.write(s)

def get_model_summary(model_path, model):
    original_stdout = sys.stdout
    with open(f'{model_path}/model_summary.out', 'a+') as f:
        f.write(f"Total layers {len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, model.parameters())))}\n")
        sys.stdout = f
        summary(model.to(device), (1,IMG_SIZE, IMG_SIZE))
        sys.stdout = original_stdout