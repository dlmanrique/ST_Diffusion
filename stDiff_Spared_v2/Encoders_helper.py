import torch
import os

class ImageEncoder():
    def __init__(self, name, dataset):
        self.encoder_name = name
        self.dataset = dataset
    
    def get_patch_encoder_model(self):
        """
        This method returns the desired patch encoder model based on the name.
        The return is the weights of the encoder
        """
        if self.encoder_name == 'uni':
            from Latents.Images_encoders.uni import UniEncoder
            encoder_class =  UniEncoder()

        elif self.encoder_name == 'shufflenet':
            #Debo buscar a partir del name los pesos del modelo
            from Latents.Images_encoders.shuffenet import ShuffenetEncoder 
            
            model_weights_path = os.path.join('Pretrained_Encoders_Autoencoders', 'Images_encoders', self.encoder_name, self.dataset, 'image_encoder.ckpt')
            encoder_class =  ShuffenetEncoder(model_weights_path)
            
        else:
            raise ValueError('The gene encoder not exist')

        patch_encoder_model, transforms = encoder_class.get_encoder()


        for param in patch_encoder_model.parameters():
            param.requires_grad = False

        patch_encoder_model.eval()

        return patch_encoder_model, transforms
        

class GeneAutoencoder():
    def __init__(self, name, path):
        self.name = name
        self.autoencoder_path = path

    def get_gene_autoencoder(self, configs: dict):
        #TODO: if I have more than one option, add the if statements to support that case.
        if self.name == "Transformer_encoder_mlp_decoder":
            from Latents.Gene_Autoencoders.gene_autoencoder import Transformer_encoder_mlp_decoder
            autoencoder =  Transformer_encoder_mlp_decoder(input_dim = configs['input_dim'], 
                                            latent_dim = configs['latent_dim'],
                                            embedding_dim = configs['embedding_dim'],
                                            num_layers = configs['num_layers'],
                                            num_heads = configs['num_heads'])
        else:
            raise ValueError('The gene encoder not exist')


        checkpoint = torch.load(self.autoencoder_path)
        autoencoder.load_state_dict(checkpoint['state_dict'])
        autoencoder.to('cuda')
        
        for param in autoencoder.parameters():
            param.requires_grad = False

        autoencoder.eval()

        return autoencoder


