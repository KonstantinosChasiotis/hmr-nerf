import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """PyTorch implementation of positional embedding.
    """
    def __init__(self, num_freq, max_freq_log2, log_sampling=True, include_input=True, input_dim=3):
        """Initialize the module.

        Args:
            num_freq (int): The number of frequency bands to sample. 
            max_freq_log2 (int): The maximum frequency. The bands will be sampled between [0, 2^max_freq_log2].
            log_sampling (bool): If true, will sample frequency bands in log space.
            include_input (bool): If true, will concatenate the input.
            input_dim (int): The dimension of the input coordinate space.

        Returns:
            (void): Initializes the encoding.
        """
        super().__init__()

        self.num_freq = num_freq
        self.max_freq_log2 = max_freq_log2
        self.log_sampling = log_sampling
        self.include_input = include_input
        self.out_dim = 0
        if include_input:
            self.out_dim += input_dim

        if self.log_sampling:
            bands = 2.0**torch.linspace(0.0, max_freq_log2, steps=num_freq)
        else:
            bands = torch.linspace(1, 2.0**max_freq_log2, steps=num_freq)

        # The out_dim is really just input_dim + num_freq * input_dim * 2 (for sin and cos)
        self.out_dim = 0
        if include_input:
            self.out_dim += input_dim
        self.out_dim += bands.shape[0] * input_dim * 2 
        self.register_buffer("bands", bands)
    
    def forward(self, coords):
        """Embded the coordinates.

        Args:
            coords (torch.FloatTensor): Coordinates of shape [N, input_dim]

        Returns:
            (torch.FloatTensor): Embeddings of shape [N, input_dim + out_dim] or [N, out_dim].
        """
        N = coords.shape[0]
        bands = self.bands.to(coords.device)
        winded = (coords[:,None] * bands[None,:,None]).reshape(N, -1)
        encoded = torch.cat([torch.sin(winded), torch.cos(winded)], dim=-1)
        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)
        return encoded

