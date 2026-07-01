import torch.nn
from os import path
from src.ML.utils.datasets import CharVocab, DummyLogDataSet
from src.ML.models.embedder import ConvEmbedder
from src.ML.models.transformer import RecTransformer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import os

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))

    data_folder = os.path.join(file_path,"../../../data/dummy/test")
    save_folder = os.path.join(file_path,"../../../models")

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
    try:
        embedder.conv_lstm.load_state_dict(
            torch.load(path.join(save_folder, lstm_conv_name) + ".pt", weights_only=True,map_location=torch.device('cpu')))
    except Exception as e:
        print("Can not load ConvLstmEncoder", lstm_conv_name)
        print(e)
        exit(-1)

    dec_enc_layer = 2
    n_head = 2
    dim_forward = 1024
    transformer_name = f"PredTransformer_DE_{dec_enc_layer}_H_{n_head}_F_{dim_forward}"
    d_model = 128
    transformer = RecTransformer(d_model=d_model, n_head=n_head, enc_layer=dec_enc_layer,
                                 dec_layer=dec_enc_layer, dim_forward=dim_forward)

    try:
        transformer.load_state_dict(torch.load(path.join(save_folder, transformer_name + ".pt"), weights_only=True,map_location=torch.device('cpu')))
    except:
        print("Can not load Transformer:", transformer_name)
        exit(-1)
    transformer.eval()
    step_size = 60
    frame_size = 60
    max_len = 200
    pad_tag = 6
    dataset = DummyLogDataSet(data_folder, step=step_size, frame_size=frame_size, max_in_len=max_in_len,
                              pad_tag=pad_tag)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    y_true = torch.zeros(0)
    y_score = torch.zeros(0)
    for i, (data, lengths, masks, tags) in enumerate(data_loader):
        with torch.no_grad():
            z = embedder(data, lengths)
            sos = torch.zeros(z.size(0), 1, z.size(2))
            tgt = torch.cat([sos, z[:, :-1, :]], dim=1)
            out = transformer(z, tgt, masks)
            tags = tags.view(-1)

            loss = F.mse_loss(out, z, reduction="none")
            loss = torch.mean(loss, dim=-1).view(-1)
            loss = loss[tags != pad_tag]
            tags = tags[tags != pad_tag]
            tags = tags > 0
            tags = tags.to(torch.int32)
            y_score = torch.cat([y_score, loss])
            y_true = torch.cat([y_true, tags])
    fpr, tpr, trsh = roc_curve(y_true, y_score)
    area = roc_auc_score(y_true, y_score)

    plt.plot(fpr, tpr, label=f"ROC krivulja (površina {area:.4f})")
    plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), linestyle='dashed', color="gray",
             label="Nasumični klasifikator (površina 0.5)")
    plt.ylabel("Stopa stvarnih pozitiva")
    plt.xlabel("Stopa lažnog alarma")
    plt.ylim((0, 1.05))
    plt.xlim((0, 1.0))
    plt.legend()
    plt.show()
