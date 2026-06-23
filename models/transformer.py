import math
import torch.nn


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


class PredTransformer(torch.nn.Module):
    def __init__(self, d_model: int = 512, n_head: int = 8, dec_layer: int = 6,
                 enc_layer: int = 6, dim_forward=1024):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dec_layer = dec_layer
        self.enc_layer = enc_layer
        self.dim_forward = dim_forward

        self.pe = PosEncoder(d_model=d_model, max_len=300)
        self.transformer = torch.nn.Transformer(d_model=d_model, batch_first=True, nhead=n_head,
                                                num_decoder_layers=dec_layer, num_encoder_layers=enc_layer,
                                                dim_feedforward=dim_forward)

    def forward(self, z: torch.Tensor, tgt_z: torch.Tensor, masks: torch.Tensor):
        src = z * math.sqrt(self.d_model)
        src = self.pe(src)
        tgt = tgt_z * math.sqrt(self.d_model)
        tgt = self.pe(tgt)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.size(1),
                                                                        device=tgt.device)
        tgt_padding_masks = torch.cat([torch.zeros(masks.size(0), 1, device=masks.device), masks[:, :-1]], dim=1)
        out = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_masks,
                               src_key_padding_mask=masks)

        return out

class TaggedTransformer(torch.nn.Module):
    def __init__(self, d_model: int = 128, dim_forward: int = 1024, n_head: int = 2,
                 num_layers: int = 2, num_class: int = 5):
        super().__init__()
        self.d_model = d_model
        self.dim_forward = dim_forward
        self.n_head = n_head
        self.num_layers = num_layers
        self.num_class = num_class

        self.pe = PosEncoder(d_model=d_model, max_len=300)

        enc_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_head,
                                                     dim_feedforward=self.dim_forward, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = torch.nn.Sequential(torch.nn.Linear(in_features=d_model, out_features=d_model // 2),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(in_features=d_model // 2, out_features=self.num_class))

    def forward(self, z: torch.Tensor, masks: torch.Tensor):
        z = z * math.sqrt(self.d_model)
        z = self.pe(z)
        z = self.transformer(z, src_key_padding_mask=masks)
        z = self.fc(z)
        return z
