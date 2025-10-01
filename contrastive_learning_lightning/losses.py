from torch import nn
import torch    
import torch.nn.functional as F

def nt_xent_loss(z_i, z_j, temperature=0.1, return_individual_losses=False):
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # [2N, D]
    z = nn.functional.normalize(z, dim=1)

    similarity_matrix = torch.matmul(z, z.T)

    mask = torch.eye(2 * batch_size, device=z.device).bool()
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
    similarity_matrix = similarity_matrix / temperature

    if torch.isnan(similarity_matrix).any():
        print("[Loss Debug] NaNs in similarity matrix")
    
    positives = torch.cat([
        torch.arange(batch_size, device=z.device) + batch_size,
        torch.arange(batch_size, device=z.device)
    ], dim=0)

    log_prob = F.log_softmax(similarity_matrix, dim=1)
    loss = -log_prob[torch.arange(2 * batch_size), positives]

    if return_individual_losses:
        return loss  
    else:
        return loss.mean()