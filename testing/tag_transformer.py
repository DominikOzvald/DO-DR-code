import numpy as np
from torch.utils.data import DataLoader

from models.embedder import ConvEmbedder
from models.transformer import TaggedTransformer
from utils.datasets import DummyLogDataSet
from utils.embeddings import CharVocab
import torch.nn
from os import path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
import matplotlib.pyplot as plt


def format_matrix_display(disp):
    for text in disp.text_.ravel():
        val = text.get_text()
        index = len(val)
        for i in range(len(val) - 1, -1, -1):
            if val[i] != "0" and val[i] != ".":
                index = i + 1
                break
            if i == 0:
                index = 1
        text.set_text(val[:index])


if __name__ == "__main__":
    data_folder = "../test_data"
    save_folder = "../trained_models"
    image_folder = "../test_images"

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
            torch.load(path.join(save_folder, lstm_conv_name) + ".pt", weights_only=True))
    except:
        print("Can not load ConvLstmEncoder", lstm_conv_name)
        exit(-1)

    enc_layer = 2
    n_head = 2
    dim_forward = 1024
    d_model = 128
    transformer_name = f"TaggedTransformer_E_{enc_layer}_H_{n_head}_F_{dim_forward}_D_{d_model}"
    model = TaggedTransformer(d_model=d_model, dim_forward=dim_forward, n_head=n_head, num_layers=enc_layer,
                              num_class=5)

    try:
        model.load_state_dict(torch.load(path.join(save_folder, transformer_name) + ".pt", weights_only=True))
    except:
        print("Can not load Transformer", transformer_name)
        exit(-1)
    model.eval()
    step = 60
    frame_size = 60
    batch_size = 64

    dataset = DummyLogDataSet(data_folder, step=step, frame_size=frame_size, pad_tag=6)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    matrix = np.zeros((5, 5))
    y_true = torch.zeros(1)
    y_pred = torch.zeros(1)
    for data, lengths, masks, tags in data_loader:
        with torch.no_grad():
            z = embedder(data, lengths)
            out = model(z, masks)
            prediction = torch.argmax(torch.softmax(out, dim=-1), dim=-1)
            tags = tags.view(-1)
            prediction = prediction.view(-1)
            prediction = prediction[tags != 6]
            tags = tags[tags != 6]
            y_pred = torch.cat([y_pred, prediction])
            y_true = torch.cat([y_true, tags])

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4], normalize="true")  # .astype(np.int32)
    matrix = np.round(matrix, 5)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix,
                                  display_labels=["Standard", 'Nestabilni', "Pomak", "Sigurnost", "Tihi"])
    fmt = lambda x: "0" if x == 0 else f"{x:.5f}"
    disp.plot(cmap="Reds", values_format=".6f")
    # plt.show()

    format_matrix_display(disp)
    plt.xlabel("Predviđene oznake")
    plt.ylabel("Stvarne oznake")
    plt.savefig(path.join(image_folder, f"{transformer_name}_matrix.png"))
    rep = classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4],
                                target_names=["Standard", "Flaky", "Drift", "Security", "Silent"], zero_division=np.nan)
    rep_dict = classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4],
                                target_names=["Standard", '"Flaky"', "Pomak", "Sigurnost", "Tihi"], zero_division=np.nan,output_dict=True)
    y_pred = y_pred[y_true!=0]
    y_true = y_true[y_true!=0]
    anomaly_acc = torch.sum(y_pred==y_true)/len(y_true)

    with open(path.join(image_folder, f"{transformer_name}_report.txt"), "w") as f:
        f.write(rep)
        for key in rep_dict:
            f.write(f"{key} : {rep_dict[key]}\n")
        f.write(f"Anomaly accuracy : {anomaly_acc.item()}")