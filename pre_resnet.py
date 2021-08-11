import torch
import torch.nn as nn

import torchvision
from torchvision import models, transforms

class ResnetObj(nn.Module):
    
    def __init__(self, split_size=7, num_boxes=2, num_classes=20, pretrained=True, tune=True):
        super().__init__()
        self.resnet = models.resnet34(pretrained=pretrained)
        self.num_classes = num_classes
        self.num_boxes = num_boxes

        # turn off gradients for all parameters if tune is False
        for param in self.resnet.parameters():
            param.requires_grad = tune
            
        # re-initialize the last layer
        self.resnet.fc = nn.Linear(512, split_size*split_size*(num_classes+num_boxes*5))

    def forward(self, x):
        out = self.resnet(x)

        return out
    
    def get_params_to_update(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad is True:
                params_to_update.append(param)
                
        return params_to_update