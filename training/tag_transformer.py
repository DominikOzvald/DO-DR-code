import torch.nn
from os import path
from utils.datasets import CharVocab, DummyLogDataSet
from models.transformer import TaggedTransformer
from torch.optim import Adam
from utils.train import tagged_train_loop
from torch.utils.data import DataLoader
from torch import save
from models.embedder import ConvEmbedder
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_folder = "../train_data"
    save_folder = "../trained_models"
    image_folder = "../train_images"
    # ----------------------------------------------------------------------------
    char_vocab = CharVocab()
    embed_size = 32
    hidden_size_enc = 196
    hidden_size_dec = 384
    latent_size = 128
    vocab_size = len(char_vocab)
    use_embed_matrix = True
    max_in_len = 200
    letter_chunk = 4
    lstm_conv_name = f"ConvLSTM_E_{embed_size}_H_{hidden_size_enc}_L_{latent_size}"
    embedder = ConvEmbedder(embed_size=embed_size, hidden_size_enc=hidden_size_enc, hidden_size_dec=hidden_size_dec,
                            latent_size=latent_size, letter_chunk=letter_chunk,
                            max_in_len=max_in_len, use_embed_matrix=use_embed_matrix, vocab_size=vocab_size)
    # ----------------------------------------------------------------------------

    try:
        embedder.conv_lstm.load_state_dict(
            torch.load(path.join(save_folder, lstm_conv_name) + ".pt", weights_only=True))
    except:
        print("Can not load ConvLSTM", lstm_conv_name)
        exit(-1)
    # ----------------------------------------------------------------------------

    enc_layer = 2
    n_head = 2
    dim_forward = 1024
    d_model = 128
    transformer_name = f"TaggedTransformer_E_{enc_layer}_H_{n_head}_F_{dim_forward}_D_{d_model}"
    model = TaggedTransformer(d_model=d_model, dim_forward=dim_forward, n_head=n_head, num_layers=enc_layer,
                              num_class=5)
    # ----------------------------------------------------------------------------
    lr = 3e-3
    optimizer = Adam(model.parameters(), lr=lr)
    # ----------------------------------------------------------------------------

    step = 10
    frame_size = 30
    batch_size = 96
    epochs = 50
    out_every = 10
    dataset = DummyLogDataSet(data_folder,step=step,frame_size=frame_size,pad_tag=6)
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    # ----------------------------------------------------------------------------

    losses, model,_ = tagged_train_loop(model,embedder,optimizer,data_loader,epochs,show_every=out_every)

    # ----------------------------------------------------------------------------

    plt.plot(range(0, len(losses)), losses[0:])
    plt.grid()
    plt.title(f"{transformer_name}_loss")
    plt.savefig(path.join(image_folder, f"{transformer_name}_loss.png"))
    # ----------------------------------------------------------------------------

    save(model.state_dict(), path.join(save_folder, transformer_name + ".pt"))
