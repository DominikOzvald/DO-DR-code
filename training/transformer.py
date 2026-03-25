import torch.nn
from os import path
from models.vae import LineVae
from utils.datasets import CharVocab, TransformerDataset
from models.vaetransformer import VAETransformer
from torch.optim import Adam
from utils.train import transformer_train_loop
from torch.utils.data import DataLoader
from torch import save
from models.embedder import Embedder
import matplotlib.pyplot as plt

if __name__ == "__main__":

    data_folder = "../train_data"
    save_folder = "../trained_models"
    image_folder = "../train_images"
    # ----------------------------------------------------------------------------

    vae_embedding_dim = 32
    vae_hidden_size = 48
    vae_latent_size = 64
    # ----------------------------------------------------------------------------

    vae_name = f"LINE_VAE_I_{vae_embedding_dim}_H_{vae_hidden_size}_L_{vae_latent_size}"
    char_vocab = CharVocab()
    vae = LineVae(torch.nn.Embedding(len(char_vocab), embedding_dim=vae_embedding_dim, padding_idx=0),
                  embedding_size=vae_embedding_dim, latent_size=vae_latent_size, hidden_size=vae_hidden_size)
    # ----------------------------------------------------------------------------

    try:
        vae.load_state_dict(torch.load(path.join(save_folder, vae_name) + ".pt", weights_only=True))
    except:
        print("Can not load VAE", vae_name)
        exit(-1)
    # ----------------------------------------------------------------------------

    dec_layer = 2
    enc_layer = 2
    n_head = 2
    dim_forward = 1024
    transformer_name = f"VAETransformer_DE_{dec_layer}_H_{n_head}_F_{dim_forward}"
    lr = 1e-3
    # ----------------------------------------------------------------------------
    embedder = Embedder(vae.enc_matrix,vae.enc)
    model = VAETransformer(z_dim=vae.enc.latent_size,d_model=128, n_head=n_head, dec_layer=dec_layer, enc_layer=enc_layer,
                           dim_forward=dim_forward)

    optimizer = Adam(model.parameters(), lr=lr)
    # ----------------------------------------------------------------------------

    step = 5
    frame_size = 30
    max_len = 200
    batch_size = 64
    epochs = 1000
    out_every = 100
    data_set = TransformerDataset(data_folder, step=step, frame_size=frame_size)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)
    # ----------------------------------------------------------------------------

    loses = transformer_train_loop(model, embedder,optimizer, data_loader, epochs, show_every_n=out_every)

    # ----------------------------------------------------------------------------

    plt.plot(range(50,len(loses)), loses[50:])
    plt.grid()
    plt.title(f"{transformer_name}_loss")
    plt.savefig(path.join(image_folder, f"{transformer_name}_loss.png"))
    # ----------------------------------------------------------------------------

    save(model.state_dict(), path.join(save_folder, transformer_name + ".pt"))
