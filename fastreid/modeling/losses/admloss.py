import torch
import torch.nn.functional as F

def admloss(logits, labels, beta,alpha, h="log_sigmoid"):

    logits_on_labels = logits.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)  # [batch_size]
    logits_diff = logits - logits_on_labels.unsqueeze(1)  
   
    if h == "linear":
        weights = torch.ones_like(logits_diff)
    elif h == "log_sigmoid":
        weights =torch.sigmoid(alpha * logits_diff)
    else:
        raise ValueError(h)
    
    gene_log_probs = F.log_softmax(logits, dim=1)
    q_probs = torch.exp(F.log_softmax(logits / beta, dim=1)).detach()

    real_log_probs = gene_log_probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
    loss = -torch.sum(q_probs * weights * (real_log_probs.unsqueeze(1) - gene_log_probs), dim=1).mean()

    return loss

