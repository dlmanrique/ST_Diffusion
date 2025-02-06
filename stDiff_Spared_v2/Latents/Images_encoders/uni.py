from uni import get_encoder
import timm
import torch
from torchvision import transforms


class UniEncoder():
    def __init__(self):
        pass
    def get_encoder(self):
        model, transforms = get_encoder(enc_name='uni', device='cuda')

        return model, transforms
    

class UniEncoder2():
    def __init__(self):
        pass
    def get_encoder(self):
        #uni2-h
        # Initialize the UNI model
        timm_kwargs = {
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        model = model.to('cuda')
        
        transforms_test = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]
                )

        return model, transforms_test