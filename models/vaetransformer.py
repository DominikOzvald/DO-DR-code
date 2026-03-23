import torch.nn
from models.vae import LineVaeEnc
from models.embedder import Embedder


class VAETransformer(torch.nn.Module):
    def __init__(self, matrix: torch.nn.Embedding, enc: LineVaeEnc, n_head: int = 8, dec_layer: int = 6,
                 enc_layer: int = 6, dim_forward=1024):
        super().__init__()
        self.embedder = Embedder(matrix, enc)
        self.transformer = torch.nn.Transformer(enc.latent_size, batch_first=True, nhead=n_head,
                                                num_decoder_layers=dec_layer, num_encoder_layers=enc_layer,
                                                dim_feedforward=dim_forward)

    def forward(self, x, lengths, masks):
        src = self.embedder(x, lengths)
        b, t, f = src.shape
        sos = torch.zeros((b, 1, f)).to(self.embedder.matrix.weight.device)
        tgt = torch.cat([sos, src[:, :-1, :]], dim=1)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(t, device=self.embedder.matrix.weight.device)

        out = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=masks,
                               src_key_padding_mask=masks)

        return out, src
