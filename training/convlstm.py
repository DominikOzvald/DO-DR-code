from utils.datasets import CharVocab, LogCharDataSet
from utils.data import fixed_pad_fn_factory,pad_len_collate_fn
from torch.optim import Adam
from models.convlstm import ConvLSTM
from torch.utils.data import DataLoader
from utils.train import conv_lstm_train_loop
import matplotlib.pyplot as plt
from os import path
from torch import save

if __name__ == "__main__":
    data_folder = "../train_data"
    save_folder = "../trained_models"
    image_folder = "../train_images"

    char_vocab = CharVocab()
    embed_size = 32
    hidden_size_enc = 196
    hidden_size_dec = 384
    latent_size = 64
    vocab_size = len(char_vocab)
    use_embed_matrix = True
    max_in_len = 200
    letter_chunk = 4

    epochs = 1000
    lr = 1e-3
    show_every = 10

    batch_size = 128
    print_every = 10
    milestones = [100,150]
    model_name = f"ConvLSTM_E_{embed_size}_H_{hidden_size_enc}_L_{latent_size}"

    data_set = LogCharDataSet(log_dir=data_folder, cut_off=max_in_len)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, collate_fn=fixed_pad_fn_factory(max_in_len))

    model = ConvLSTM(embed_size=embed_size, hidden_size_enc=hidden_size_enc, hidden_size_dec=hidden_size_dec,
                     latent_size=latent_size, letter_chunk=letter_chunk,
                     max_in_len=max_in_len, use_embed_matrix=use_embed_matrix, vocab_size=vocab_size)
    optimizer = Adam(model.parameters(), lr=lr)
    losses,model = conv_lstm_train_loop(model, optimizer, data_loader, epochs=epochs, show_every_n=show_every,milestones=milestones)

    plt.plot(range(5, len(losses)), losses[5:])
    plt.title(model_name + " loss")
    plt.grid()
    plt.savefig(path.join(image_folder, model_name + ".png"))

    save(model.state_dict(), path.join(save_folder, model_name + '.pt'))
