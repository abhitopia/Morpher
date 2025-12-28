import torch
import torch.nn as nn


class CastedSparseEmbedding(nn.Module):
    """
    Sparse embedding for efficient puzzle embedding training.
    
    Uses SignSGD-friendly structure where only accessed embeddings get gradients.
    Based on HRM's implementation.
    """
    
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        batch_size: int,
        dtype: torch.dtype,
        init_std: float = 0.0
    ):
        super().__init__()
        self.dtype = dtype
        
        # Real weights (persistent buffer)
        weights = torch.empty(num_embeddings, embedding_dim)
        if init_std > 0:
            nn.init.trunc_normal_(weights, std=init_std, a=-2*init_std, b=2*init_std)
        else:
            nn.init.zeros_(weights)
        self.register_buffer("weights", weights, persistent=True)
        
        # Local weights for training (with gradient, not persistent)
        self.register_buffer("local_weights", torch.zeros(batch_size, embedding_dim, requires_grad=True), persistent=False)
        self.register_buffer("local_ids", torch.zeros(batch_size, dtype=torch.int32), persistent=False)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            # Eval mode: direct lookup
            return self.weights[inputs].to(self.dtype)
        
        # Training mode: copy to local, return local (for gradient)
        with torch.no_grad():
            self.local_weights.copy_(self.weights[inputs])
            self.local_ids.copy_(inputs)
        
        return self.local_weights.to(self.dtype)
