import torch

def cumsum(x, exclusive=True):
    r"""Computes the cumulative sum of the elements of x along the last dimension.

    Args:
        x (torch.Tensor): tensor of shape :math:`[..., Ns, 1]`.
        exclusive (bool): whether to compute exclusive cumsum or not. (default: True)

    Returns:
        torch.Tensor: tensor of shape :math:``[..., Ns, 1]`.

    """
    x = x.squeeze(-1)
    if exclusive:
        c = torch.cumsum(x, dim=-1)
        # "Roll" the elements along dimension 'dim' by 1 element.
        c = torch.roll(c, 1, dims=-1)
        # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
        c[..., 0] = 0.0
        return c[..., None]
    else:
        return torch.cumsum(x, dim=-1)[..., None]


def exponential_integration(feats, tau, depth, exclusive=True):
    r"""Exponential transmittance integration across packs using the optical thickness (tau).

    Exponential transmittance is derived from the Beer-Lambert law. Typical implementations of
    exponential transmittance is calculated with :func:`cumprod`, but the exponential allows a reformulation
    as a :func:`cumsum` which its gradient is more stable and faster to compute. We opt to use the :func:`cumsum`
    formulation.

    For more details, we recommend "Monte Carlo Methods for Volumetric Light Transport" by Novak et al.

    Args:
        feats (torch.FloatTensor): features of shape [..., Ns, num_feats].
        tau (torch.FloatTensor): optical thickness of shape [..., Ns, 1].
        depth (torch.FloatTensor): depth of shape [..., Ns, 1].
        exclusive (bool): Compute exclusive exponential integration if true. (default: True)

    Returns:
        (torch.FloatTensor, torch.FloatTensor)
        - Integrated features of shape [..., num_feats].
        - Weights of shape [..., Ns, 1].

    """
    # TODO(ttakikawa): This should be a fused kernel... we're iterating over packs, so might as well
    #                  also perform the integration in the same manner.
    alpha = 1.0 - torch.exp(-tau.contiguous())
    # Uses the reformulation as a cumsum and not a cumprod (faster and more stable gradients)
    transmittance = torch.exp(-1.0 * cumsum(tau.contiguous(), exclusive=exclusive))
    weights = transmittance * alpha
    feats_out = (weights * feats).sum(dim=-2)
    depth_out = (depth* weights).sum(dim=-2)
    alpha_out = weights.sum(dim=-2)
    return feats_out, depth_out, alpha_out
