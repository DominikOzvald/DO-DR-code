from utils.embeddings import create_embedding_matrix, CharVocab
from utils.data import extract_raw
from models.vae import LineVae, CharVae
import torch
from os import path, listdir
import matplotlib.pyplot as plt
import torch.nn.functional as F

if __name__ == "__main__":
    data_folder = "../test_data"
    save_folder = "../trained_models"
    image_folder = "../test_images"
    embedding_dim = 32
    hidden_size = 196
    latent_size = 64
    model_name = f"LINE_VAE_I_{embedding_dim}_H_{hidden_size}_L_{latent_size}"
    char_vocab = CharVocab()
    matrix = create_embedding_matrix(char_vocab, embedding_dim)

    model = LineVae(matrix, embedding_size=embedding_dim, latent_size=latent_size, hidden_size=hidden_size)
    model.load_state_dict(torch.load(path.join(save_folder, model_name + ".pt"), weights_only=True))
    model.eval()

    files = [file for file in listdir(data_folder) if file[-4:] == ".txt"]

    for log_file in files:
        print("Testing file:", log_file)
        data = extract_raw(path.join(data_folder, log_file))
        with torch.no_grad():
            accuracies = []
            for log in data:
                instance = char_vocab.encode(log).unsqueeze(dim=0)
                length = torch.tensor(len(log)).unsqueeze(dim=0).cpu()
                logist, mean, log_var = model(instance, length)
                rec = torch.argmax(torch.softmax(logist, dim=2), dim=2)
                acc = torch.sum(rec == instance) / (rec.shape[0] * rec.shape[1])
                accuracies.append(acc.item())
                #print(F.mse_loss(model.get_z(instance, length), model.get_z(instance, length)))
            print(f"Average accuracy: {sum(accuracies) / len(data):.4f}")
            plt.plot(range(len(accuracies)), accuracies)
            plt.title(f"{log_file}")
            plt.ylim((0, 1.1))
            plt.savefig(path.join(image_folder, f"{model_name}_{log_file[:-4]}.png"))
            plt.clf()
