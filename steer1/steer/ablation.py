import torch
import einops

def directional_ablation(
    x: torch.Tensor,
    r: torch.Tensor,
    coefficient: float = 1,
):
    '''
        Apply directional ablation to the input tensor x.

        Args:
            x (torch.Tensor): The input tensor to be ablated.
            r (torch.Tensor): The direction of ablation as a tensor of shape (1, 1, d_model).
            coefficient (float): The scaling factor for the ablation.

        Returns:
            torch.Tensor: The ablated tensor.
    '''

    batch_size, seq_len, d_model = x.shape

    r = r / torch.norm(r)
    r = r.to(x.dtype)

    proj = einops.einsum(x, r.unsqueeze(-1), '... d_model, d_model single -> ... single') * r

    return x - proj * coefficient
