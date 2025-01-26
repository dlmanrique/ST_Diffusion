
class ImageEncoder():
    def __init__(self, name):
        self.encoder_name = name
    
    def get_patch_encoder_model(self):
        """
        This method returns the desired patch encoder model based on the name.
        The return is the weights of the encoder
        """
        if self.encoder_name == 'uni':
            from Latents.Images_encoders.uni import UniEncoder
            encoder_class =  UniEncoder()

        else:
            raise ValueError('The gene encoder not exist')

        patch_encoder_model = encoder_class.get_encoder()
        return patch_encoder_model
        

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
        
        breakpoint()
        autoencoder = None
        

        return autoencoder


