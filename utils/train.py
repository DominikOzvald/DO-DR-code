import math
import copy
import torch
import tqdm
import torch.nn.functional as F
from utils.data import separate_last_log


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


def transformer_loss(out, z, masks):
    loss = F.mse_loss(out, z, reduction='none')
    inverse_masks = (torch.ones_like(masks).to(masks.device) - masks).float().unsqueeze(-1)
    loss = (loss * inverse_masks).sum() / inverse_masks.sum()
    return loss


def transformer_train_loop(model, embedder, optimizer, data_loader, epochs, show_every_n):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    embedder.requires_grad_(False)
    embedder.to(device)
    embedder.train()
    # model.embedder.requires_grad_(False)
    losses = []
    batch_num = len(data_loader.dataset) // data_loader.batch_size + 1
    min_loss = 100.0
    best_model = copy.deepcopy(model)
    for epoch in tqdm.tqdm(range(epochs)):
        train_loss = 0
        for batch, (data, lengths, masks) in enumerate(data_loader):
            data = data.to(device)
            masks = masks.to(device)
            lengths = lengths.cpu()
            z = embedder(data, lengths)
            sos = torch.zeros(z.size(0), 1, z.size(2)).to(device)
            z_tgt = torch.cat([sos, z[:, :-1, :]], dim=1)
            optimizer.zero_grad()
            out = model(z, z_tgt, masks)
            loss = transformer_loss(out, z, masks)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        if epoch % show_every_n == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Average Loss: {train_loss / batch_num}")
            # print("z", z[0, 0, :5])
            # print("o", out[0, 0, :5])
            # print("o", out[0, 1, :5])
        if loss.item() < min_loss:
            min_loss = loss.item()
            best_model = copy.deepcopy(model)
        losses.append(train_loss / batch_num)
    print(f"Min loss: {min_loss}")
    return losses, model


def dec_trans_train_loop(model, embedder, optimizer, data_loader, epochs, show_every_n, n_steps: int = 1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    embedder.requires_grad_(False)
    embedder.to(device)
    embedder.train()
    losses = []
    batch_num = len(data_loader.dataset) // data_loader.batch_size + 1
    for epoch in range(epochs):
        train_loss = 0
        for batch, (data, lengths, masks) in tqdm.tqdm(enumerate(data_loader)):
            data = data.to(device)
            masks = masks.to(device)
            lengths = lengths.cpu()
            optimizer.zero_grad()
            z = embedder(data, lengths)
            z, target, masks = separate_last_log(z, masks, n_steps=n_steps)
            out = model(z, masks, n_steps=n_steps)
            loss = F.mse_loss(out.reshape(-1, out.size(-1)), target.reshape(-1, out.size(-1)))
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        if epoch % show_every_n == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Average Loss: {train_loss / batch_num}")
        losses.append(train_loss / batch_num)
    return losses


def conv_lstm_train_loop(model, optimizer, data_loader, epochs: int = 200, show_every_n: int = 10,
                         milestones: list = [300, 500, 900]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    losses = []
    batch_num = len(data_loader.dataset) // data_loader.batch_size + 1
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.7)
    max_acc = 0
    best_model = copy.deepcopy(model)
    for epoch in tqdm.tqdm(range(epochs)):
        train_loss = 0
        for data, lengths in data_loader:
            data = data.to(device)
            lengths = lengths.cpu()
            optimizer.zero_grad()
            z = model(data, lengths)
            loss = criterion(z.reshape(-1, z.size(-1)), data.reshape(-1))
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        with torch.no_grad():
            predict = torch.argmax(torch.softmax(z, dim=-1), dim=-1)
            acc = torch.sum(data == predict) / torch.sum(lengths)
            if acc > max_acc:
                max_acc = acc
                best_model = copy.deepcopy(model)
            if epoch % show_every_n == 0 or epoch == epochs - 1:
                print(f"Accuracy:{acc:.5f}")

        losses.append(train_loss / batch_num)
        scheduler.step()

    print(f"Max Accuracy:{max_acc:.5f}")
    return losses, best_model
