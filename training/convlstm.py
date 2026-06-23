from utils.datasets import CharVocab, DummyCharDataSet
from utils.data import fixed_pad_fn_factory
from torch.optim import Adam
from models.convlstm import ConvLSTM
from torch.utils.data import DataLoader
from utils.train import conv_lstm_train_loop
from os import path
from torch import save

if __name__ == "__main__":
    data_folder = "../train_data"
    save_folder = "../trained_models"

    char_vocab = CharVocab()
    embed_size = 32
    hidden_size_enc = 196
    hidden_size_dec = 384
    latent_size = 128
    vocab_size = len(char_vocab)
    use_embed_matrix = True
    max_in_len = 200
    letter_chunk = 4

    epochs = 50
    lr = 1e-3
    show_every = 10

    batch_size = 64
    print_every = 10
    milestones = [100, 150]
    model_name = f"ConvLSTM_E_{embed_size}_H_{hidden_size_enc}_L_{latent_size}"

    data_set = DummyCharDataSet(log_dir=data_folder, max_in_len=max_in_len)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, collate_fn=fixed_pad_fn_factory(max_in_len))

    model = ConvLSTM(embed_size=embed_size, hidden_size_enc=hidden_size_enc, hidden_size_dec=hidden_size_dec,
                     latent_size=latent_size, letter_chunk=letter_chunk,
                     max_in_len=max_in_len, use_embed_matrix=use_embed_matrix, vocab_size=vocab_size)
    optimizer = Adam(model.parameters(), lr=lr)
    losses, model = conv_lstm_train_loop(model, optimizer, data_loader, epochs=epochs, show_every_n=show_every,
                                         milestones=milestones)

    save(model.state_dict(), path.join(save_folder, model_name + '.pt'))
