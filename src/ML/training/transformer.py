import torch.nn
from os import path
from src.ML.utils.datasets import CharVocab, TransformerDataset
from src.ML.models.transformer import PredTransformer
from torch.optim import Adam
from src.ML.utils.train import transformer_train_loop
from torch.utils.data import DataLoader
from torch import save
from src.ML.models.embedder import ConvEmbedder

if __name__ == "__main__":
    file_path = path.dirname(path.abspath(__file__))
    data_folder = path.join(file_path,"../../../data/dummy/train_no_anomaly")
    save_folder = path.join(file_path,"../../../models")
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
    # ----------------------------------------------------------------------------

    lstm_conv_name = f"ConvLSTM_E_{embed_size}_H_{hidden_size_enc}_L_{latent_size}"
    embedder = ConvEmbedder(embed_size=embed_size, hidden_size_enc=hidden_size_enc, hidden_size_dec=hidden_size_dec,
                            latent_size=latent_size, letter_chunk=letter_chunk,
                            max_in_len=max_in_len, use_embed_matrix=use_embed_matrix, vocab_size=vocab_size)
    # ----------------------------------------------------------------------------

    try:
        embedder.conv_lstm.load_state_dict(
            torch.load(path.join(save_folder, lstm_conv_name) + ".pt", weights_only=True,map_location=torch.device('cpu')))
    except Exception as e:
        print("Can not load ConvLSTM", lstm_conv_name)
        print(e)
        exit(-1)
    # ----------------------------------------------------------------------------

    dec_layer = 2
    enc_layer = 2
    n_head = 2
    dim_forward = 1024
    d_model = 128
    transformer_name = f"PredTransformer_DE_{dec_layer}_H_{n_head}_F_{dim_forward}"
    lr = 2e-3
    # ----------------------------------------------------------------------------

    model = PredTransformer(d_model=d_model, n_head=n_head, dec_layer=dec_layer, enc_layer=enc_layer,
                            dim_forward=dim_forward)

    optimizer = Adam(model.parameters(), lr=lr)
    # ----------------------------------------------------------------------------

    step = 10
    frame_size = 30
    max_len = 200
    batch_size = 64
    epochs = 200
    out_every = 2
    data_set = TransformerDataset(data_folder, step=step, frame_size=frame_size)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    # ----------------------------------------------------------------------------

    loses, model = transformer_train_loop(model, embedder, optimizer, data_loader, epochs, show_every_n=out_every)
    # ----------------------------------------------------------------------------

    save(model.state_dict(), path.join(save_folder, transformer_name + ".pt"))
