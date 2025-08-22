import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_Linear
# from layers.MaskModel import MaskModel
from layers.MaskDWT import MaskModel
from layers.MSPI import MSPINet

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        if configs.freq == 'h':
            self.x_mark_dim = 4
        elif configs.freq == 't':
            self.x_mark_dim = 5
        elif configs.freq == 's':
            self.x_mark_dim = 6
        else :
            self.x_mark_dim = 4
            
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.mask_type = configs.mask_type
        self.use_norm = configs.use_norm

        mask_dim = configs.enc_in + self.x_mark_dim
        mask_len = configs.seq_len + sum([i ** 2 for i in range(1, configs.mspi_layers + 1)])
        
        # CI Masking
        self.mask_dwt = MaskModel(input_shape=(configs.batch_size, configs.enc_in, configs.seq_len))
        
        # Multi-Scale Pyramid Integration
        self.mspi = MSPINet(configs.mspi_layers, configs.pool_type)

        # Embedding
        self.embedding = DataEmbedding_Linear(configs.seq_len, mask_len, configs.d_model,
                                               configs.embed, configs.freq, configs.dropout)

        # Encoder with Multi-Scale Attention Fussion
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    mask_dim,
                    configs.batch_size,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Projection
        # self.mapping = nn.Linear(configs.d_model, configs.map_dim, bias=True)
        # self.projection = nn.Linear(configs.map_dim, configs.pred_len, bias=True)
        self.project = nn.Sequential(
            nn.Linear(configs.d_model, configs.map_dim, bias=True),  
            nn.GELU(),                                              
            nn.Linear(configs.map_dim, configs.pred_len, bias=True))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # x_enc = self.mask_sft(x_enc)
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, C = x_enc.shape  # B L C
        # print(x_enc.shape)
        y = x_enc.permute(0, 2, 1)  # B C L
        
        # CI Masking
        y = self.mask_dwt(y)

        # Multi-Scale Pyramid Integration
        y_mspi = self.mspi(y)  # B C L'
        if x_mark_enc is None:
            y_mark = None # B C' L'
        else:
            y_mark = self.mspi(x_mark_enc.permute(0, 2, 1)) # B C' L'
             
        # Embedding
        y_mspi_embed, y_embed = self.embedding(y_mspi, y, y_mark, x_mark_enc)

        # Encoder with Multi-Scale Attention Fussion
        z, _ = self.encoder(y_mspi_embed, y_embed, attn_mask=None)

        # Projection
        dec_out = self.project(z).permute(0, 2, 1)[:, :, :C]

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, :]   # [B, N, C]
