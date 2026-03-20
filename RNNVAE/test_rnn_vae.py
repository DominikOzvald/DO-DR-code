import pickle
import torch
from os import path, listdir
from embeddings import LogVocab, create_embedding_matrix
from rnn_vae import RnnVae
from data_utils import extract_with_parse, form_instances
from Drain import LogParser
import matplotlib.pyplot as plt

if __name__ == "__main__":

    data_folder = "../test_data"
    save_folder = "../trained_models"
    image_folder = "../test_images"
    step_size = 10
    frame_size = 30
    re = [r"^[\.s]+$", r'##\[\w+\]']
    log_format = "<DateTime> <Content>"
    embedding_dim = 384
    batch_size = 128
    hidden_size = 400
    latent_size = 100
    model_name = f"GRU_VAE_I_{embedding_dim}_H_{hidden_size}_L_{latent_size}"

    parser = LogParser(log_format, "./", "./", depth=4, st=0.2, rex=re, verbose=False)

    try:
        with open(path.join(save_folder, model_name + ".vcb"), "rb") as vocab_file:
            str2int = pickle.load(vocab_file)

    except:
        print("Can not load log vocab")
        exit(-1)

    log_vocab = LogVocab(str2int=str2int)
    matrix = create_embedding_matrix(log_vocab, embedding_dim)
    model = RnnVae(matrix, input_size=embedding_dim, hidden_size=hidden_size, latent_size=latent_size)
    model.load_state_dict(torch.load(path.join(save_folder, model_name + ".pt"), weights_only=True))
    model.eval()

    files = [file for file in listdir(data_folder) if file[-4:] == ".txt"]
    for log_file in files:
        print("Testing file:", log_file)
        _, data = extract_with_parse(path.join(data_folder, log_file), parser)
        instances = form_instances(data, step=step_size, frame_size=frame_size)
        with torch.no_grad():
            recon_errors = []
            for i, instance in enumerate(instances):
                instance_enc = log_vocab.encode(instance).unsqueeze(dim=0).to(torch.long)
                instance_enc = torch.transpose(instance_enc,dim0=0,dim1=1)
                logits, mean, log_var = model(instance_enc)
                rec = torch.argmax(torch.softmax(logits,dim=2),dim=2)
                acc = torch.sum(rec == instance_enc)/(rec.shape[0]*rec.shape[1])

                recon_errors.append(acc)
                print(f"Testing line {i * step_size + 1}-{i * step_size + frame_size}: {acc:.3f}")
            print()
            plt.plot(range(len(recon_errors)),recon_errors)
            plt.title(f"{log_file}")
            plt.ylim((0,1.1))
            plt.savefig(path.join(image_folder,f"{model_name}_{log_file[:-4]}.png"))
            plt.clf()

