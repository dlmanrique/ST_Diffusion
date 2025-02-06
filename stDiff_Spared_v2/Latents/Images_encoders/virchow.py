import timm
import torch
from torchvision import transforms
from timm.layers import SwiGLUPacked
import torch.nn as nn


class ModifiedVirchow1(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model  # Modelo original Virchow

    def forward(self, x):
        output = self.model(x)  # Salida original: [1, 257, 1280]

        class_token = output[:, 0]   # [1, 1280]
        patch_tokens = output[:, 1:] # [1, 256, 1280]

        # Concatenar class token y el promedio de los patch tokens
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # [1, 2560]

        return embedding  # Salida corregida [1, 2560]


class Virchow():
    def __init__(self):
        pass

    def get_encoder(self):

        # Cargar el modelo base
        base_model = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )

        # Modificar el modelo para que el output tenga dimensión [1, 2560]
        model = ModifiedVirchow1(base_model)

        # Enviar a evaluación y a GPU
        model = model.eval().to('cuda')


        transforms_test = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]
                )
        return model, transforms_test
    


class ModifiedVirchow2(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model  # Modelo original Virchow2
    

    def forward(self, x):
        output = self.model(x)  # Salida original: [1, 261, 1280]

        class_token = output[:, 0]   # [1, 1280]
        patch_tokens = output[:, 5:] # [1, 256, 1280] (ignoramos los 4 primeros tokens de registro)

        # Concatenar class token y el promedio de los patch tokens
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # [1, 2560]

        return embedding  # Salida corregida [1, 2560]


class Virchow2():
    def __init__(self):
        pass

    def get_encoder(self):

        # Cargar el modelo base
        base_model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )

        # Modificar el modelo para asegurar salida [1, 2560]
        model = ModifiedVirchow2(base_model)

        # Enviar a evaluación y a GPU
        model = model.eval().to('cuda')

        transforms_test = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]
                )
        return model, transforms_test

