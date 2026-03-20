import torch.nn.functional as f


def compute_recon_loss(x, recon):
    recon_loss = f.cross_entropy(recon.reshape(-1, recon.shape[-1]), x.view(-1))
    return recon_loss
