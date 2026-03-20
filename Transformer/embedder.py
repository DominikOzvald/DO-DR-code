import torch.nn
from RNNVAE.char_vae import LineVaeEnc


class Embedder(torch.nn.Module):
    def __init__(self, matrix: torch.nn.Embedding, enc: LineVaeEnc):
        super().__init__()
        self.matrix = matrix
        self.enc = enc

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        B, T, N = x.shape
        x_re = x.view(B * T, -1)
        lengths_re = lengths.view(B * T)
        char_embedding = self.matrix(x_re)
        encoded, _, _ = self.enc(char_embedding, lengths_re)
        encoded = encoded.view(B, T, -1)
        return encoded
