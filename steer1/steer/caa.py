import torch

def caa(
    x: torch.Tensor,
    r: torch.Tensor,
    coefficient: float = 1,
):
    '''
        Apply caa to the input tensor x.

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

    # expand r from d to 1, 1, d
    r = r.unsqueeze(0).unsqueeze(0)

    x = x + r * coefficient
    return x