import torch.nn
from models.vae import LineVaeEnc
from models.convlstm import ConvLSTM


class Embedder(torch.nn.Module):
    def __init__(self, matrix: torch.nn.Embedding, enc: LineVaeEnc):
        super().__init__()
        self.matrix = matrix
        self.enc = enc

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        b, t, f = x.shape
        x_re = x.view(b * t, -1)
        lengths_re = lengths.view(b * t)
        char_embedding = self.matrix(x_re)
        encoded, _, _ = self.enc(char_embedding, lengths_re)
        encoded = encoded.view(b, t, -1)
        return encoded


class ConvEmbedder(torch.nn.Module):
    def __init__(self, embed_size: int, hidden_size_enc: int, hidden_size_dec: int, latent_size: int, vocab_size: int,
                 use_embed_matrix: bool = False, max_in_len: int = 256,
                 letter_chunk: int = 8):
        super().__init__()
        self.conv_lstm = ConvLSTM(embed_size=embed_size, hidden_size_enc=hidden_size_enc,
                                  hidden_size_dec=hidden_size_dec, latent_size=latent_size, vocab_size=vocab_size,
                                  use_embed_matrix=use_embed_matrix, max_in_len=max_in_len, letter_chunk=letter_chunk)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        b, t, f = x.shape
        x_re = x.view(b * t, -1)
        lengths_re = lengths.view(b * t)
        encoded = self.conv_lstm.get_z(x_re,lengths_re)
        encoded = encoded.view(b, t, -1)
        return encoded
