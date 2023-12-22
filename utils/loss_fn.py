import torch
import torch.nn.functional as F


def cross_entropy_loss_fn(logits, targets, reduction='mean'):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)
    return loss

def focal_loss_fn(logits, targets, alpha=0.25, gamma=2.0):
    # Calculate the cross entropy loss
    ce_loss = cross_entropy_loss_fn(logits, targets, reduction=None)
    # Calculate the probabilities of the targets
    p_t = torch.exp(-ce_loss)
    # Calculate the focal loss
    focal_loss = alpha * (1 - p_t) ** gamma * ce_loss
    return focal_loss.mean()


def ghmc_loss_fn(pred, target, batch_size, bins=10, momentum=0):
    """
    Gradient Harmonized Single-stage Detector Loss Function.

    Args:
    pred (torch.Tensor): The direct prediction of classification layer.
    target (torch.Tensor): Binary class target for each sample.
    batch_size (int): The size of the batch.
    bins (int): Number of bins to compute density, default is 10.
    momentum (float): Momentum for updating the moving average of valid bins.

    Returns:
    torch.Tensor: Computed GHM loss.
    """
    assert pred.dim() == target.dim(), \
        'Inconsistent dimensions between output and target!'

    target = target.float()
    edges = [float(x) / bins for x in range(bins + 1)]
    edges[-1] += 1e-6
    acc_sum = [0.0 for _ in range(bins)] if momentum > 0 else None

    # Gradient length
    g = torch.abs(pred.detach() - target)

    tot = batch_size
    n = 0  # n valid bins
    weights = torch.zeros_like(pred)
    for i in range(bins):
        inds = (g >= edges[i]) & (g < edges[i + 1])
        num_in_bin = inds.sum().item()
        if num_in_bin > 0:
            if momentum > 0:
                acc_sum[i] = momentum * acc_sum[i] + (1 - momentum) * num_in_bin
                weights[inds] = tot / acc_sum[i]
            else:
                weights[inds] = tot / num_in_bin
            n += 1

    if n > 0:
        weights /= n

    loss = F.binary_cross_entropy(pred, target, weights, reduction='sum') / tot
    return loss


def fringe_loss_fn(input, target, eps=0.25, reduction='mean'):
    """
    Fringe Loss Function. Focusing on the simple samples for better generalization.

    Args:
    input (torch.Tensor): Predictions from the model.
    target (torch.Tensor): Ground truth labels.
    eps (float): Epsilon value for the loss calculation.
    reduction (str): Reduction method to apply, default is 'mean'.

    Returns:
    torch.Tensor: Computed Fringe loss.
    """
    device = input.device
    u_k = torch.normal(mean=0.5, std=0.15, size=target.shape).to(device)
    H_pq = F.binary_cross_entropy(input, target)
    H_pn = F.binary_cross_entropy(input, u_k)
    p = torch.exp(-H_pq)
    loss = p * ((1 - eps) * H_pq + eps * H_pn)

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss
