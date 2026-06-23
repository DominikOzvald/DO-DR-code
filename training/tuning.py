from models.embedder import ConvEmbedder
from utils.datasets import CharVocab, DummyLogDataSet
from torch.utils.data import DataLoader
from utils.train import tagged_train_loop
import torch
from os import path
from torch.optim import Adam
from models.transformer import TaggedTransformer
from torch import save
if __name__ == "__main__":
    data_folder = "C:/Faks/Diplomski rad/data/tagged_gha/train"
    save_folder = "../trained_models"

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
    transformer_name = f"TaggedTransformer_E_{enc_layer}_H_{n_head}_F_{dim_forward}_D_{d_model}_tuned"
    model = TaggedTransformer(d_model=d_model, dim_forward=dim_forward, n_head=n_head, num_layers=enc_layer,
                              num_class=5)
    # ----------------------------------------------------------------------------
    lr = 1.5e-3
    optimizer = Adam(model.parameters(), lr=lr)
    # ----------------------------------------------------------------------------

    step = 30
    frame_size = 30
    batch_size = 64
    epochs = 15
    out_every = 2
    weights = torch.Tensor([0.05,0.1,0.5,0.5,0.2])
    dataset = DummyLogDataSet(data_folder,step=step,frame_size=frame_size,pad_tag=6)
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    # ----------------------------------------------------------------------------

    losses,_, model = tagged_train_loop(model,embedder,optimizer,data_loader,epochs,show_every=out_every,weights=weights)

    # ----------------------------------------------------------------------------

    save(model.state_dict(), path.join(save_folder, transformer_name + "_tuned.pt"))