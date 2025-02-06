from conch.open_clip_custom import create_model_from_pretrained
from torchvision.transforms import Compose, ToTensor, Normalize



class CONCHEncoder():
    def __init__(self):
        pass


    def get_encoder(self):
        
        model, _ = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch",
                                                  hf_auth_token="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")
        model = model.to('cuda')
        transforms = Compose([
                        ToTensor(),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

        return model, transforms