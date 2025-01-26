from uni import get_encoder


class UniEncoder():
    def __init__(self):
        pass
    def get_encoder(self):
        model, transforms = get_encoder(enc_name='uni', device='cuda')
        return model, transforms