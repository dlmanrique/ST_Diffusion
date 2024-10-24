import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention, Mlp
from utils import get_main_parser

# Get parser and parse arguments
parser = get_main_parser()
args = parser.parse_args()
args_dict = vars(args)

# Seed
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
 
    
class Attention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Cambio de lineal a conv1d
        # Conv1d layers for q, k, v
        #self.qkv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias)
        self.q_conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias)
        self.k_conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias)
        self.v_conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias)


        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # Opcion convolucional
        # x shape: (batch_size, patches, feature_dim)
        B, G, C = x.shape

        # Transpose for Conv1d: (batch_size, feature_dim, seq_length)
        # Se permuta para que pueda entrar en la capa convolucional
        x = x.permute(0, 2, 1)

        # Compute q, k, v using Conv1d
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)
        # B, num_head, G, head_dim--> q, v ,k
        # Reshape for multi-head attention: (batch_size, num_heads, seq_length, head_dim)
        q = q.reshape(B, self.num_heads, self.head_dim, G).permute(0, 1, 3, 2)  # (B, num_heads, seq_length, head_dim)
        k = k.reshape(B, self.num_heads, self.head_dim, G).permute(0, 1, 3, 2)  # (B, num_heads, seq_length, head_dim)
        v = v.reshape(B, self.num_heads, self.head_dim, G).permute(0, 1, 3, 2)  # (B, num_heads, seq_length, head_dim)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, num_heads, seq_length, seq_length)
        attn = attn.softmax(dim=-1)  # Apply softmax to get probabilities
        attn = self.attn_drop(attn)  # Dropout for regularization

        # Apply attention scores to the value vectors
        x = torch.matmul(attn, v)  # (B, num_heads, patches, head_dim)

        # Combine heads: (batch_size, patches, feature_dim)
        x = x.permute(0, 2, 1, 3).reshape(B, G, C)

        # Project back to original dimension
        x = x.permute(0, 2, 1)  # Back to (batch_size, feature_dim, patches) for Conv1d
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 1)  # Return to (batch_size, patches, feature_dim)

        return x

def modulate(x, shift, scale):
    res = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return res

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    time emb to frequency_embedding_size dim, then to hidden_size
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DiTblock(nn.Module):
    # adaBN -> attn -> mlp
    def __init__(self,
                 feature_dim=2000,
                 mlp_ratio=4.0,
                 num_heads=10,
                 **kwargs) -> None:
        super().__init__()
        
        self.norm1 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        #self.attn = Attention2(feature_dim, num_heads=num_heads, qkv_bias=True, **kwargs) 
        self.attn = Attention(feature_dim, num_heads=num_heads, qkv_bias=True, **kwargs)
        #self.attn = AttentionPool2d(feature_dim, num_heads=num_heads, qkv_bias=True, **kwargs)
        
        self.norm2 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU()
        
        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp = Mlp(in_features=feature_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feature_dim, 6 * feature_dim, bias=True)
        )
    
    def forward(self, x, cond):
        """
        Forward pass for DiTblock
        :param x: Input tensor of shape (batch_size, hidden_size, 128, 7)
        :param cond: Condition tensor of shape (batch_size, hidden_size, 128, 7)
        :return: Output tensor of shape (batch_size, hidden_size, 128, 7)
        """
        # Project condition to 6 * hiddensize and then slice it into 6 parts along the column.
        # (batch_size, patches, hidden_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)
        
        # attention block
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # mlp block
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x) 
        x = x.permute(0, 2, 1)
        return x    
    
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=(128,7),
            patch_size=(128,7),
            in_chans=1,
            embed_dim=512,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        #self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        #if self.flatten:
        #    x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        #x = self.norm(x)
        return x
 

BaseBlock = {'dit':DiTblock}

class DiT_stDiff_2D(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 depth,
                 dit_type,
                 num_heads,
                 classes,
                 args,
                 in_channels = 1,
                 mlp_ratio=4.0,
                 **kwargs) -> None:
        """ denoising model

        Args:
            input_size (_type_): input dim
            hidden_size (_type_): scale input to hidden dim
            depth (_type_): dit block num
            dit_type (_type_): which type block to use
            num_heads (_type_): transformer head num
            classes (_type_): class num
            mlp_ratio (float, optional): _description_. Defaults to 4.0.
        """        
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.classes = classes
        self.mlp_ratio = mlp_ratio
        self.dit_type = dit_type
        #self.out_size = input_size.shape[0]*input_size.shape[1]
        self.out_size = 1
        self.cond_channels = 1 #en el caso de concatenar noise, x_cond, mask
        self.img_size = input_size
        
        # time emb
        self.time_emb = TimestepEmbedder(hidden_size=self.hidden_size)
        self.norm_layer = nn.LayerNorm
        # image and condition embedding 
        #self.x_emb = PatchEmbed(self.img_size, self.patch_size, self.out_size, hidden_size, self.norm_layer, bias=True)
        #self.y_emb = PatchEmbed(self.img_size, self.patch_size, self.cond_channels, hidden_size, self.norm_layer, bias=True)
        
        # 2D Convolutional cond layer
        self.in_layer = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        )
        #self.in_layer = nn.Sequential(
        #    nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #)
        
        # conditional emb
        self.condi_emb = nn.Embedding(classes, hidden_size)
        
        # DiT block
        self.blks = nn.ModuleList([
            BaseBlock[dit_type](self.hidden_size, mlp_ratio=self.mlp_ratio, num_heads=self.num_heads) for _ in
            range(self.depth)
        ])

        # out layer
        self.out_layer = FinalLayer(self.hidden_size, self.out_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                # xavier 
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    # bias  0
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        # Initialize label embedding table:
        nn.init.normal_(self.condi_emb.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_emb.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.dit_type == 'dit':
            for block in self.blks:
                # adaLN_modulation , adaLN  0 initial
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.out_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.linear.weight, 0)
        nn.init.constant_(self.out_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size[0] * patch_size[1] * C)
        imgs: (N, H, W, C)
        """
        c = self.out_size
        p_0 = self.x_emb.patch_size[0] 
        p_1 = self.x_emb.patch_size[1] 
        h = int(self.x_emb.img_size[0]/self.x_emb.patch_size[0])
        w = int(self.x_emb.img_size[1]/self.x_emb.patch_size[1])

        x = x.reshape(shape=(x.shape[0], h, w, p_0, p_1, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p_0, w * p_1))
        return imgs

    def forward(self, x, t, y,**kwargs): 
        """
        Forward pass for DiT_stDiff
        :param x: Input tensor of shape (batch_size, 1, 128, 7)
        :param t: Timestep tensor
        :param y: Condition tensor of shape (batch_size, 1, 128, 7)
        :return: Output tensor of shape (batch_size, 1, 128, 7)
        """
        x = x.float().unsqueeze(1) #(batch_size, 1, 128, 7)
        cond = y[0].unsqueeze(1) #(batch_size, 1, 128, 7)
        mask = y[1].unsqueeze(1) #(batch_size, 1, 128, 7)
        
        #Concatenar las entradas
        input = torch.cat((x, cond, mask), dim=1)
        input = input.reshape(input.shape[0], input.shape[1], input.shape[2]*input.shape[3])
        
        input = self.in_layer(input) #(bs, hidden_size, 896)
        # Embed time and repeat along the 7 dimensionn
        t = self.time_emb(t) 
        c = t
        
        #Convert input to (bs, 896, hidden_size)
        input = input.permute(0, 2, 1)
        
        # Pass through each DiTblock
        for blk in self.blks:
            x = blk(input, c)
        # Output layer
        # (bs, 128*7, hidden_size)
        x = self.out_layer(x, c) 
        # (bs, 1, 896)
        x = x.reshape(x.shape[0], x.shape[1], self.input_size[0], self.input_size[1]) 
        # (bs, 1, 128, 7)
        x = x.squeeze(1) # (batch_size, 128, 7)
        return x
        