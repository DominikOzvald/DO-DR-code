import torch.nn
from os import path
from utils.datasets import CharVocab, TransformerDataset
from models.vae import LineVae
from models.embedder import ConvEmbedder
from models.vaetransformer import VAETransformer
from torch.utils.data import DataLoader
from utils.train import transformer_loss
import matplotlib.pyplot as plt
import math
import numpy as np

if __name__ == "__main__":
    data_folders = ["../test_data/S", "../test_data/F"]
    truth_folder = ["../test_data/T/0", "../test_data/T/1", "../test_data/T/2"]
    save_folder = "../trained_models"
    image_folder = "../test_images"

    char_vocab = CharVocab()
    embed_size = 32
    hidden_size_enc = 196
    hidden_size_dec = 384
    latent_size = 64
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
            torch.load(path.join(save_folder, lstm_conv_name) + ".pt", weights_only=True))
    except:
        print("Can not load ConvLstmEncoder", lstm_conv_name)
        exit(-1)

    dec_enc_layer = 2
    n_head = 2
    dim_forward = 1024
    transformer_name = f"VAETransformer_DE_{dec_enc_layer}_H_{n_head}_F_{dim_forward}"
    d_model = 64
    transformer = VAETransformer(d_model=d_model, n_head=n_head, enc_layer=dec_enc_layer,
                                 dec_layer=dec_enc_layer, dim_forward=dim_forward)

    try:
        transformer.load_state_dict(torch.load(path.join(save_folder, transformer_name + ".pt"), weights_only=True))
    except:
        print("Can not load Transformer:", transformer_name)
        exit(-1)
    step_size = 5
    frame_size = 30
    max_len = 200
    truth_losses = []
    for folder in truth_folder:
        losses = []
        data_set = TransformerDataset(folder, step=step_size, frame_size=frame_size, max_len=max_len)
        data_loader = DataLoader(data_set, batch_size=1, shuffle=False)
        for i, (data, lengths, masks) in enumerate(data_loader):
            with torch.no_grad():
                z = embedder(data, lengths)
                sos = torch.zeros(z.size(0), 1, z.size(2))
                tgt = torch.cat([sos, z[:, :-1, :]], dim=1)
                out = transformer(z, tgt, masks)
                loss = transformer_loss(out, z, masks)
                losses.append(loss.item())
                if i > 100:
                    break
        truth_losses.append(torch.Tensor(losses).unsqueeze(-1))
    truth_losses = torch.cat(truth_losses, dim=-1)
    thresholds = torch.mean(truth_losses, dim=-1) + 3 * torch.std(truth_losses, dim=-1)

    threshold_window_size = 50
    for k, folder in enumerate(data_folders):
        data_set = TransformerDataset(folder, step=step_size, frame_size=frame_size, max_len=max_len)
        data_loader = DataLoader(data_set, batch_size=1, shuffle=False)
        losses = []
        loss_windows = []
        anomalies = []

        for i, (data, lengths, masks) in enumerate(data_loader):
            with torch.no_grad():
                z = embedder(data, lengths)
                sos = torch.zeros(z.size(0), 1, z.size(2))
                tgt = torch.cat([sos, z[:, :-1, :]], dim=1)
                out = transformer(z, tgt, masks)
                loss = transformer_loss(out, z, masks)
                is_anomaly = False
                if loss.item() > thresholds[i].item():
                    is_anomaly = True

                losses.append(loss.item())
                anomalies.append(is_anomaly)
            if i > 100:
                break
            print(f"Line {i * step_size}-{i * step_size + frame_size}: {loss.item():.4f} | is anomaly: {is_anomaly}")
        print(f"Average loss: {sum(losses) / len(losses):.4f}")
        plt.plot(range(len(thresholds)), thresholds, color="orange",
                 label="threshold")
        plt.plot(range(len(losses)), losses, label="MSE")
        plt.scatter(np.arange(len(losses))[anomalies], np.array(losses)[anomalies], color="red", label="anomaly")
        plt.grid()
        plt.legend()
        plt.title(folder)
        plt.savefig(path.join(image_folder, f"{transformer_name}_{k}.png"))
        plt.clf()
