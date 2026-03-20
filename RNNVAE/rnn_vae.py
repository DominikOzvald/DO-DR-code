import torch


class RnnVaeEnc(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, latent_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.rnn1 = torch.nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size)
        self.fc1 = torch.nn.Linear(in_features=self.hidden_size, out_features=2 * self.latent_size)

    def forward(self, x):
        h, _ = self.rnn1(x)
        mean, log_var = torch.split(self.fc1(h), self.latent_size, dim=2)

        eps = torch.normal(0, 1, size=mean.shape).to(self.fc1.weight.device)
        z = mean + torch.exp(0.5 * log_var) * eps

        return z, mean, log_var


class RnnVaeDec(torch.nn.Module):
    def __init__(self, embedding_size: int, latent_size:int,hidden_size: int, vocab_size: int):
        super().__init__()

        self.embedding_size = embedding_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.rnn1 = torch.nn.GRU(input_size=latent_size+embedding_size, hidden_size=hidden_size)
        self.fc = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, z,targets):
        rnn_input = torch.cat([targets,z],dim=2)
        h, _ = self.rnn1(rnn_input)
        recon = self.fc(h)
        return recon


class RnnVae(torch.nn.Module):

    def __init__(self, matrix, input_size: int, hidden_size: int, latent_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.matrix = matrix
        self.enc = RnnVaeEnc(input_size=self.input_size, hidden_size=self.hidden_size, latent_size=self.latent_size)
        self.dec = RnnVaeDec(embedding_size=self.input_size,latent_size=self.latent_size, hidden_size=self.hidden_size,
                             vocab_size=self.matrix.weight.shape[0])

    def forward(self, in_log):
        x = self.matrix(in_log)
        z, mean, log_var = self.enc(x)
        sos = self.matrix(2*torch.ones(in_log[0].shape,dtype=torch.long).to(self.enc.fc1.weight.device)).unsqueeze(0)
        targets = torch.cat([sos,x],dim=0)[:-1]
        recon = self.dec(z,targets)

        return recon, mean, log_var

    # bf = [[[111,112,113],[121,122,123]],
    #       [[211,212,213],[221,222,223]],
    #       [[311,312,313],[321,322,323]],
    #       [[411,412,413],[421,422,423]],
    #       [[511,512,513],[521,522,523]]]
    #
    # tf = [[[111,112,113],
    #        [211,212,213],
    #        [311,312,313],
    #        [411,412,413],
    #        [511,512,513]],
    #
    #       [[121,122,123],
    #        [221,222,223],
    #        [321,322,323],
    #        [421,422,423],
    #        [521,522,523]]]
