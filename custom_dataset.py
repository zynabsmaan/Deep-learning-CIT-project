import os, re
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import torchvision.transforms.functional as fn
import pandas as pd
from settings import IMG_SIZE

class Chest3Dataset(Dataset):
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

class CovidDataset(Dataset):
    def __init__(self, annotations_file, img_dir, special_transform = None, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, sep=" ")
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.special_transform = special_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):       
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        try:
            image = read_image(img_path)
        except:
            print(img_path)

        image = fn.resize(image, size=[IMG_SIZE, IMG_SIZE])
        label = self.img_labels.iloc[idx, 1]
        if image.shape[0] == 4:
            image = image[0,:,:]
            image = torch.reshape(image, (1, IMG_SIZE, IMG_SIZE))
        if image.shape[0] > 1:
            image = self.special_transform(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        
class ReadData(object):
    def __init__(self, parent_folder_path):

        self.build_csv_files(parent_folder_path)


    def build_csv_files(self, parent_folder_path):
        sub_dirs = [name for name in os.listdir(parent_folder_path)]
        for sub_dir in sub_dirs:
            
            all_image_paths = []
            all_labels = []

            data_paths = [data_path for data_path in os.walk(os.path.join(parent_folder_path, sub_dir))]

            for data_path in data_paths:
                # import pdb; pdb.set_trace()
                if data_path[-1]:
                    folder_path = data_path[0]
                    image_paths = data_path[-1]

                    full_image_paths = [os.path.join(folder_path, image_path) for image_path in image_paths]
                    labels = self.get_annotation(image_paths)

                    all_image_paths.extend(full_image_paths)
                    all_labels.extend(labels)
            self.save_to_csv(sub_dir, all_image_paths, all_labels)


    def get_annotation(self, image_paths):
        labels = []
        for image_path in image_paths:
            label = re.findall(r'bacteria|normal|virus|im', image_path.lower())[0]
            if label == 'im':
                labels.append('normal')
            else:
                labels.append(label)
        return labels

    def save_to_csv(self, file_name, image_paths, labels):
        data = {'image_path': image_paths, 'labels': labels}
        df = pd.DataFrame(data)
        df['labels'] = df['labels'].map({'bacteria':1,'virus':1, 'normal':0})
        df.to_csv(f'{file_name}.csv', index=False)
