from utils.datasets import LogCharDataSet
from utils.data import pad_len_collate_fn
from utils.embeddings import create_embedding_matrix
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.vae import CharVae,LineVae
from torch.optim import Adam
from utils.train import vae_train_loop_lengths
from os import path
from torch import save

if __name__ == "__main__":
    data_folder = "../train_data"
    save_folder = "../trained_models"
    image_folder = "../train_images"
    batch_size = 128
    embedding_dim = 32
    hidden_size = 196
    latent_size = 64
    lr = 1e-3
    epochs = 100
    print_every = 10
    model_name = f"LINE_VAE_I_{embedding_dim}_H_{hidden_size}_L_{latent_size}"
    beta = 0.001
    # ----------------------------------------------------------------------------

    data_set = LogCharDataSet(data_folder)
    matrix = create_embedding_matrix(data_set.vocab,embedding_dim)
    data_loader = DataLoader(data_set,batch_size=batch_size,shuffle=True,collate_fn=pad_len_collate_fn)
    # ----------------------------------------------------------------------------

    _, axs = plt.subplots(1, 2)
    axs[0].imshow(matrix.weight.clone().detach().cpu().numpy()[:50, :50])
    axs[0].set_title("Embedding matrix before training")

    # ----------------------------------------------------------------------------

    model = LineVae(matrix,embedding_size=embedding_dim,hidden_size=hidden_size,latent_size=latent_size)
    print(model.enc_matrix.weight[40:45, 40:45])
    optimizer = Adam(model.parameters(),lr=lr)
    loss = vae_train_loop_lengths(model,data_loader,optimizer,epochs=epochs,show_every_n=print_every,beta=beta)
    # ----------------------------------------------------------------------------
    print(model.enc_matrix.weight[40:45, 40:45])
    axs[1].imshow(matrix.weight.clone().detach().cpu().numpy()[:50,:50])
    axs[1].set_title("Embedding matrix after training")

    plt.savefig(path.join(image_folder, model_name + '_embedding.png'))
    plt.clf()

    plt.plot(range(2, len(loss)), loss[2:], )
    plt.title(model_name + " loss")
    plt.savefig(path.join(image_folder, model_name + "_loss.png"))

    save(model.state_dict(), path.join(save_folder, model_name + '.pt'))