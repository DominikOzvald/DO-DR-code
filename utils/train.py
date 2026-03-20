import torch
import tqdm


def vae_loss(x, recon, mean, log_var, beta, criterion):
    recon_loss = criterion(recon.reshape(-1, recon.shape[-1]), x.view(-1))
    kl_loss = -0.5 * torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var))
    return recon_loss + beta * kl_loss / x.shape[0]


def vae_train_loop(model, dataloader, optimizer, epochs=100, show_every_n=10, beta=0.1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    model.to(device)
    model.train()
    losses = []
    batch_num = len(dataloader.dataset) // dataloader.batch_size + 1
    for epoch in range(epochs):
        train_loss = 0
        for batch, data in enumerate(dataloader):
            data = data.to(device)

            optimizer.zero_grad()
            rec, mean, log_var = model(torch.transpose(data, dim0=0, dim1=1))
            loss = vae_loss(data, torch.transpose(rec, dim0=0, dim1=1), mean, log_var, beta, criterion)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        if epoch % show_every_n == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Average Loss: {train_loss / batch_num}")
        losses.append(train_loss / batch_num)
    return losses


def vae_train_loop_lengths(model, dataloader, optimizer, epochs=100, show_every_n=10, beta=0.1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    model.to(device)
    model.train()
    losses = []
    batch_num = len(dataloader.dataset) // dataloader.batch_size + 1
    for epoch in tqdm.tqdm(range(epochs)):
        train_loss = 0
        for batch, (data, lengths) in enumerate(dataloader):
            data = data.to(device)
            lengths = lengths.cpu()
            optimizer.zero_grad()
            rec, mean, log_var = model(data, lengths)
            kl_loss = -0.5 * torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var)) / data.size(1)
            loss = criterion(rec.reshape(-1, rec.shape[-1]), data.reshape(-1)) + beta * kl_loss
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        if epoch % show_every_n == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Average Loss: {train_loss / batch_num}")
        losses.append(train_loss / batch_num)
    return losses
