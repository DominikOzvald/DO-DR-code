import matplotlib.pyplot as plt
from datasets import LogDataSet
from data_utils import pad_collate_fn
from embeddings import create_embedding_matrix
from rnn_vae import RnnVae
from torch.utils.data import DataLoader
from torch.optim import Adam
from TrainUtils import vae_train_loop
from torch import save
from os import path
import pickle

if __name__ == "__main__":
    # ----------------------------------------------------------------------------
    data_folder = "../train_data"
    save_folder = "../trained_models"
    image_folder = "../train_images"
    min_freq = 2
    max_size = 2000
    step_size = 10
    frame_size = 30
    re = [r"^[\.s]+$", r'##\[\w+\]']
    log_format = "<DateTime> <Content>"
    embedding_dim = 384
    batch_size = 64
    hidden_size = 400
    latent_size = 100
    lr = 1e-4
    epochs = 1000
    print_every = 100
    model_name = f"GRU_VAE_I_{embedding_dim}_H_{hidden_size}_L_{latent_size}"
    beta = 0.1

    # ----------------------------------------------------------------------------

    data_set = LogDataSet(data_folder, step=step_size,frame_size=frame_size,minFreq=min_freq, rex=re)
    matrix = create_embedding_matrix(data_set.vocab, dim=embedding_dim,use_transformer=False)
    train_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True,
                              collate_fn=pad_collate_fn)

    _, axs = plt.subplots(1, 2)
    axs[0].imshow(matrix.weight.clone().detach().cpu().numpy()[:50,:50])
    axs[0].set_title("Embedding matrix before training")
    # ----------------------------------------------------------------------------

    model = RnnVae(matrix, input_size=embedding_dim, hidden_size=hidden_size, latent_size=latent_size)
    optimizer = Adam(model.parameters(), lr=lr)
    loss = vae_train_loop(model, dataloader=train_loader, optimizer=optimizer, epochs=epochs, show_every_n=print_every,beta = beta)
    # ----------------------------------------------------------------------------

    axs[1].imshow(matrix.weight.clone().detach().cpu().numpy()[:50,:50])
    axs[1].set_title("Embedding matrix after training")

    plt.savefig(path.join(image_folder, model_name + '_embedding.png'))
    plt.clf()
    plt.plot(range(50,len(loss)), loss[50:], )
    plt.title(model_name + " loss")
    plt.savefig(path.join(image_folder, model_name + "_loss.png"))

    save(model.state_dict(), path.join(save_folder, model_name + '.pt'))

    try:
        with open(path.join(save_folder,model_name+".vcb"),"wb") as vocab_file:
            pickle.dump(data_set.vocab.str2int,vocab_file)
    except:
        print("could not save vocab dictionary")
