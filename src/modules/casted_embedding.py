import torch
import torch.nn as nn


class CastedEmbedding(nn.Module):
    """Embedding with truncated normal init and dtype casting."""
    
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        dtype: torch.dtype,
        init_std: float = 0.02
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.dtype = dtype
        # Truncated normal init
        nn.init.trunc_normal_(self.embedding.weight, std=init_std, a=-2*init_std, b=2*init_std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x).to(self.dtype)
