from torchvision import models
import torch.nn as nn
import torch
from pathlib import Path
import os
from collections import OrderedDict
from torchvision.transforms import Compose, ToTensor, Normalize



current_file = Path(__file__).resolve()
root_folder = current_file.parents[2]


class ShuffenetEncoder():
    def __init__(self, path, genes_number):
        self.model_weights = path
        self.genes_number = genes_number


    def get_encoder(self):
        model_ft = models.shufflenet_v2_x0_5()
        
        model_ft.fc = nn.Linear(1024, self.genes_number)
        checkpoint = torch.load(os.path.join(root_folder, self.model_weights),  weights_only=False)

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items(): 
            new_key = k.replace("encoder.", "")  
            new_state_dict[new_key] = v

        model_ft.load_state_dict(new_state_dict)
        model_ft = nn.Sequential(*list(model_ft.children())[:-1],
                                 nn.AdaptiveAvgPool2d(1),  # (1024,7,7) -> (1024,1,1)
                                nn.Flatten())
        
        model_ft.to('cuda')

        transforms = Compose([
                        ToTensor(),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

        return model_ft, transforms
    