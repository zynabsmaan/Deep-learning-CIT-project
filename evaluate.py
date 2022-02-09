import os, re
import torch 
import torchvision as tv
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import classification_report
from resnet import resnet20, resnet32, resnet44, resnet56
from hybrid import Hybrid
from vit import  ViT
from custom_dataset import Chest3Dataset, CovidDataset
from utils import AverageMeter
from settings import EMBEDDING_DIM, IMG_SIZE, N_INPUT_CHANNELS, NUM_CLASSES,POOLING_KERNEL_SIZE, POOLING_STRIDE, POOLING_PADDING, NUM_LAYERS_LIST, NUM_HEADS_LIST


test_loader = DataLoader(Chest3Dataset('data/chest_xray_3/test.csv', 'data/chest_xray_3', special_transform=transforms.Compose([tv.transforms.Grayscale(num_output_channels=1)])), batch_size=624, shuffle=False, pin_memory=True)
test_loader = DataLoader(CovidDataset('data/covid_data/test.csv', 'data/covid_data/test', special_transform=transforms.Compose([tv.transforms.Grayscale(num_output_channels=1)])), batch_size=400, shuffle=False, pin_memory=True)

precision = AverageMeter()
recall = AverageMeter()
f_score = AverageMeter()


def hybrid_evaluate(parent_folder):
    """
    parent_folder: string - folder name where the results of the moel saved.
    returns:
        file contains all measured mertics for the models.
    """
    dir_list = os.listdir(parent_folder)
    for dir_name in dir_list:

        info_file_path = os.path.join(parent_folder, dir_name, 'save_info.txt')
        model_path = os.path.join(parent_folder, dir_name, 'model.pth')
        print(info_file_path)
        with open(info_file_path, 'r') as f:
            info = f.read()
            if info:
                n_conv_layers = int(re.findall('n_conv_layers: ([0-9]+)', info)[0])
                kernel_size = int(re.findall('kernel_size: ([0-9]+)', info)[0])
                num_layers = int(re.findall('num_layers: ([0-9]+)', info)[0])
                num_heads = int(re.findall('num_heads: ([0-9]+)', info)[0])
                mlp_radio = int(re.findall('mlp_radio: ([0-9]+)', info)[0])

                model = Hybrid(
                    img_size=IMG_SIZE,
                    embedding_dim=EMBEDDING_DIM[0],
                    n_input_channels=N_INPUT_CHANNELS,
                    n_conv_layers=n_conv_layers,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                    pooling_kernel_size=POOLING_KERNEL_SIZE,
                    pooling_stride=POOLING_STRIDE,
                    pooling_padding=POOLING_PADDING,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    mlp_radio=mlp_radio,
                    num_classes=NUM_CLASSES)
                model.load_state_dict(torch.load(model_path))
                model.eval()

                with torch.no_grad():
                    for i, (input, target) in enumerate(test_loader):
                        input_var = input.float()
                        target_var = target.float()
                    
                        output = model(input_var).float()
                        _, pred = output.topk(1, 1, True, True)
                        
                        target_names = ['class 0', 'class 1']
                        report = classification_report(target_var, pred.t()[0], target_names=target_names)
                        
                        with open(f'{parent_folder}/all_metrics.out', 'a') as f :
                            f.write(f'{dir_name}\n\n')
                            f.write(report)
                            f.write('====================================================================')



def evaluate_resnet(parent_folder):
    architectures = {'resnet20': resnet20, 'resnet32': resnet32, 'resnet44': resnet44, 'resnet56': resnet56}
    dir_list = os.listdir(parent_folder)
    for dir_name in dir_list:

        info_file_path = os.path.join(parent_folder, dir_name, 'save_info.txt')
        model_path = os.path.join(parent_folder, dir_name, 'model.pth')
        print(info_file_path)
        with open(info_file_path, 'r') as f:
            info = f.read()
            if info:
                architecture = re.findall('Architecture: (.*)', info)[0]
                model = architectures[architecture]()
                
                model.load_state_dict(torch.load(model_path))
                model.eval()

                with torch.no_grad():
                    for i, (input, target) in enumerate(test_loader):
                        input_var = input.float()
                        target_var = target.float()
                    
                        output = model(input_var).float()
                        _, pred = output.topk(1, 1, True, True)
                        
                        target_names = ['class 0', 'class 1']
                        report = classification_report(target_var, pred.t()[0], target_names=target_names)

                        with open(f'{parent_folder}/all_metrics.out', 'a') as f :
                            f.write(f'{dir_name}\n\n')
                            f.write(report)
                            f.write('====================================================================')





def vit_evaluate(parent_folder):
    ind = 2
    for depth in NUM_LAYERS_LIST: 
        for head in NUM_HEADS_LIST:
            model_path = os.path.join(parent_folder, f'model_{ind}', 'model.pth')
            print(model_path)
            model = ViT(image_size = IMG_SIZE, patch_size = 64, num_classes = NUM_CLASSES, dim = 128, depth = depth, heads = head, mlp_dim = 256, dropout = 0.1, emb_dropout = 0.1)               
            model.load_state_dict(torch.load(model_path))
            model.eval()

            with torch.no_grad():
                for i, (input, target) in enumerate(test_loader):
                    input_var = input.float()
                    target_var = target.float()
                
                    output = model(input_var).float()
                    _, pred = output.topk(1, 1, True, True)
                    
                    target_names = ['class 0', 'class 1']
                    try:
                        report = classification_report(target_var, pred.t()[0], target_names=target_names)

                        with open(f'{parent_folder}/all_metrics.out', 'a') as f :
                            f.write(f'model_{ind}\n\n')
                            f.write(report)
                            f.write('====================================================================')
                    except:
                        print(ind)
            ind += 1



# evaluate_resnet('covid_resnet_results')
hybrid_evaluate('hybrid_results')
# vit_evaluate('covid_vit_results')