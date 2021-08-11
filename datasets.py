import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import cv2
from PIL import Image
import torchvision.transforms as transforms

def load_full_data():
    df_train = pd.read_csv("data/train.csv")
    df_valid = pd.read_csv("data/test.csv")
    
    return df_train, df_valid


def load_data():
    df_train = pd.read_csv("data/train.csv")[:1000]
    df_valid = pd.read_csv("data/test.csv")[:50]
    
    return df_train, df_valid


def load_tiny():
    df_train = pd.read_csv("data/100examples.csv")
    df_valid = pd.read_csv("data/8examples.csv")
    
    return df_train, df_valid


class YOLODataset(Dataset):
    """
    - No data augmentation because there are so many bugs in Albumentation
    - S: split an image by (S x S)
    - B: the number of box in an image
    - C: number of classes
    """
    def __init__(self, df, S=7, B=2, C=20, img_size=448, img_dir="data/images", label_dir="data/labels"):
        self.df = df
        self.S = S
        self.B = B
        self.C = C
        self.img_dir = img_dir
        self.label_dir = label_dir
        
        self.resize = transforms.Resize((img_size, img_size))
        self.to_tensor = transforms.ToTensor()
        
        
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):
        image = self._load_image(idx)
        boxes = self._load_boxes(idx)
        label_matrix = self._load_label_matrix(boxes)
#         print(torch.max(image), torch.max(torch.tensor(boxes)), torch.max(label_matrix))

        return image, label_matrix
        
        
    def _load_image(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]["img"])
        image = Image.open(img_path)
        image = self.resize(image)
        image = self.to_tensor(image)
        
        return image
    
    
    def _load_boxes(self, idx):
        """
        read box files, get a list of boxes
        each box is formated as [class_label, x, y, width, height]
        """
        label_path = os.path.join(self.label_dir, self.df.iloc[idx]['label'])
        
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = label.split()
                box = [int(class_label), float(x), float(y), float(width), float(height)]
                boxes.append(box)
            
        return boxes
    
    
    def _load_label_matrix(self, boxes):
        """
        convert a list of boxes [[...], [...], [...]]
        to tensor of shape (S, S, C+5B), where SxS is the split of image
        5 means (probability_of_some_class, x, y, width, height)
        """
        label_matrix = torch.zeros((self.S, self.S, self.C+5*self.B))
        for box in boxes:
            # convert each box to fit the label matrix 
            
            class_label, x, y, width, height = box
            
            # which cell in (S x S) split does the center (x, y) belongs to
            i = int(self.S * y)
            j = int(self.S * x)
            
            # what is coordinate does the center in the cell (i, j)
            cell_x = self.S * x - j
            cell_y = self.S * y - i
            
            # what is the relative with and height of the box if
            # assuming the width and height of each cell in (S x S) split is 1?
            width_cell = width * self.S
            height_cell = height * self.S
            
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1 # for computing loss, the probability is 1
                label_matrix[i, j, class_label] = 1
                box_coord = torch.tensor([cell_x, cell_y, width_cell, height_cell])
                
                label_matrix[i, j, 21:25] = box_coord
#             print(label_matrix[i, j])
        
        return label_matrix  