import math

import torch.nn
from models.vae import LineVaeEnc
from models.embedder import Embedder


class PosEncoder(torch.nn.Module):
    def __init__(self, d_model: int = 128, max_len: int = 300):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        positions = torch.arange(max_len).unsqueeze(1)
        div_factor = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(positions * div_factor)
        pe[:, 1::2] = torch.cos(positions * div_factor)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, :x.size(1), :]


class VAETransformer(torch.nn.Module):
    def __init__(self, z_dim: int = 64, d_model: int = 512, n_head: int = 8, dec_layer: int = 6,
                 enc_layer: int = 6, dim_forward=1024):
        super().__init__()
        self.z_dim = z_dim
        self.d_model = d_model
        self.n_head = n_head
        self.dec_layer = dec_layer
        self.enc_layer = enc_layer
        self.dim_forward = dim_forward

        self.fc_in = torch.nn.Linear(in_features=z_dim, out_features=d_model)
        self.pe = PosEncoder(d_model=d_model,max_len=300)
        self.transformer = torch.nn.Transformer(d_model=d_model, batch_first=True, nhead=n_head,
                                                num_decoder_layers=dec_layer, num_encoder_layers=enc_layer,
                                                dim_feedforward=dim_forward)
        self.fc_out = torch.nn.Linear(in_features=d_model, out_features=z_dim)

    def forward(self, z: torch.Tensor, tgt_z: torch.Tensor, masks: torch.Tensor):
        src = self.fc_in(z) * math.sqrt(self.d_model)
        src = self.pe(src)
        tgt = self.fc_in(tgt_z) * math.sqrt(self.d_model)
        tgt = self.pe(tgt)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.size(1),
                                                                        device=tgt.device)

        out = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=masks,
                               src_key_padding_mask=masks)

        return self.fc_out(out)
