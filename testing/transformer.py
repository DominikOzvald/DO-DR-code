import torch.nn
from os import path
from utils.datasets import CharVocab, TransformerDataset
from models.vae import LineVae
from models.embedder import Embedder
from models.vaetransformer import VAETransformer
from torch.utils.data import DataLoader
from utils.train import transformer_loss
import matplotlib.pyplot as plt
if __name__ == "__main__":
    data_folders = ["../test_data/order", "../test_data/rand"]
    save_folder = "../trained_models"
    image_folder = "../test_images"

    vae_embedding = 32
    vae_hidden = 196
    vae_latent = 64

    vae_name = f"LINE_VAE_I_{vae_embedding}_H_{vae_hidden}_L_{vae_latent}"
    char_vocab = CharVocab()

    vae = LineVae(torch.nn.Embedding(len(char_vocab), vae_embedding), embedding_size=vae_embedding,
                  latent_size=vae_latent, hidden_size=vae_hidden)
    try:
        vae.load_state_dict(torch.load(path.join(save_folder, vae_name + '.pt'), weights_only=True))
    except:
        print("Can not load VAE:", vae_name)
        exit(-1)

    embedder = Embedder(vae.enc_matrix, vae.enc)

    dec_enc_layer = 2
    n_head = 2
    dim_forward = 1024
    transformer_name = f"VAETransformer_DE_{dec_enc_layer}_H_{n_head}_F_{dim_forward}"
    d_model = 96
    transformer = VAETransformer(z_dim=vae_latent, d_model=d_model, n_head=n_head, enc_layer=dec_enc_layer,
                                 dec_layer=dec_enc_layer, dim_forward=dim_forward)

    try:
        transformer.load_state_dict(torch.load(path.join(save_folder, transformer_name + ".pt"), weights_only=True))
    except:
        print("Can not load Transformer:", transformer_name)
        exit(-1)
    step_size = 5
    frame_size = 15
    max_len = 200
    for k, folder in enumerate(data_folders):
        data_set = TransformerDataset(folder, step=step_size, frame_size=frame_size, max_len=max_len)
        data_loader = DataLoader(data_set, batch_size=1, shuffle=False)
        losses = []
        for i, (data,lengths,masks) in enumerate(data_loader):
            with torch.no_grad():
                z = embedder(data,lengths)
                sos = torch.zeros(z.size(0),1,z.size(2))
                tgt = torch.cat([sos,z[:,:-1,:]],dim=1)
                out = transformer(z,tgt,masks)
                loss = transformer_loss(out,z,masks)
                losses.append(loss.item())
            if i > 25:
                break
            print(f"Line {i*step_size}-{i*step_size+frame_size}: {loss.item()}")
        print(f"Average loss: {sum(losses)/len(data_set)}")
        plt.plot(range(len(losses)),losses)
        plt.grid()
        plt.title(folder)
        plt.savefig(path.join(image_folder, f"{transformer_name}_{k}.png"))
        plt.clf()


