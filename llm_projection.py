"""
LLM Projection Module for Mamba4Rec Fusion

This module projects LLM embeddings (typically 768 or 1024 dimensions from
sentence transformers) into the Mamba hidden space (64 dimensions by default).

The projection uses a multi-layer architecture with:
- Bottleneck design to compress information
- Layer normalization for training stability
- Residual connections where dimensionally possible
- Dropout for regularization

Training Strategy:
- Phase 1: Train vanilla Mamba4Rec (this module not used)
- Phase 2: Train this module with frozen Mamba layers
- Phase 3: Joint fine-tuning with low learning rate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LLMProjection(nn.Module):
    """
    Projects LLM embeddings to Mamba hidden space.

    Args:
        llm_dim: Dimension of input LLM embeddings (default: 768 for sentence-transformers)
        hidden_size: Dimension of Mamba hidden space (default: 64)
        dropout: Dropout rate (default: 0.1)
        use_layer_norm: Whether to use layer normalization (default: True)
    """

    def __init__(self, llm_dim=768, hidden_size=64, dropout=0.1, use_layer_norm=True):
        super().__init__()
        self.llm_dim = llm_dim
        self.hidden_size = hidden_size

        # Intermediate dimension (bottleneck)
        intermediate_dim = max(hidden_size * 2, llm_dim // 4)

        # Main projection path
        self.proj = nn.Sequential(
            nn.Linear(llm_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_size),
            nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

        # Final normalization to ensure output is well-scaled
        self.output_norm = nn.LayerNorm(hidden_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use smaller initialization to prevent dominating Mamba outputs initially
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, llm_emb):
        """
        Project LLM embedding to Mamba space.

        Args:
            llm_emb: Tensor of shape [B, llm_dim]

        Returns:
            Tensor of shape [B, hidden_size]
        """
        projected = self.proj(llm_emb)
        return self.output_norm(projected)


class LLMProjectionWithAlignment(nn.Module):
    """
    Extended LLM Projection with contrastive alignment capability.

    This version includes methods for computing alignment loss during training,
    which helps the projection learn a better mapping to the Mamba item space.

    Args:
        llm_dim: Dimension of input LLM embeddings
        hidden_size: Dimension of Mamba hidden space
        dropout: Dropout rate
        temperature: Temperature for contrastive loss (default: 0.07)
    """

    def __init__(self, llm_dim=768, hidden_size=64, dropout=0.1, temperature=0.07):
        super().__init__()
        self.llm_dim = llm_dim
        self.hidden_size = hidden_size
        self.temperature = temperature

        # Base projection (same as LLMProjection)
        intermediate_dim = max(hidden_size * 2, llm_dim // 4)

        self.encoder = nn.Sequential(
            nn.Linear(llm_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # Projection head for contrastive learning (discarded after training)
        self.contrastive_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Final output layer
        self.output_layer = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, llm_emb, return_contrastive=False):
        """
        Project LLM embedding to Mamba space.

        Args:
            llm_emb: Tensor of shape [B, llm_dim]
            return_contrastive: If True, also return contrastive head output

        Returns:
            If return_contrastive is False: Tensor of shape [B, hidden_size]
            If return_contrastive is True: Tuple of (output, contrastive_output)
        """
        encoded = self.encoder(llm_emb)
        output = self.output_layer(encoded)

        if return_contrastive:
            contrastive_out = self.contrastive_head(encoded)
            return output, contrastive_out
        return output

    def compute_alignment_loss(self, llm_emb, item_emb):
        """
        Compute contrastive alignment loss between projected LLM embeddings
        and item embeddings.

        This loss encourages the projection to map movie descriptions to
        positions near their corresponding item embeddings in Mamba space.

        Args:
            llm_emb: Tensor of shape [B, llm_dim] - LLM embeddings of movie descriptions
            item_emb: Tensor of shape [B, hidden_size] - Corresponding item embeddings

        Returns:
            Scalar alignment loss
        """
        # Get contrastive representations
        _, contrastive_proj = self.forward(llm_emb, return_contrastive=True)

        # Normalize for cosine similarity
        contrastive_proj = F.normalize(contrastive_proj, dim=-1)
        item_emb = F.normalize(item_emb, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(contrastive_proj, item_emb.T) / self.temperature

        # Labels are diagonal (each row should match its corresponding column)
        batch_size = llm_emb.shape[0]
        labels = torch.arange(batch_size, device=llm_emb.device)

        # Symmetric contrastive loss
        loss_llm_to_item = F.cross_entropy(logits, labels)
        loss_item_to_llm = F.cross_entropy(logits.T, labels)

        return (loss_llm_to_item + loss_item_to_llm) / 2

    def compute_geometry_preservation_loss(self, llm_emb_1, llm_emb_2):
        """
        Compute a regularization loss that preserves relative distances in LLM space.

        This prevents the projection from collapsing semantically similar inputs
        to identical points.

        Args:
            llm_emb_1: Tensor of shape [B, llm_dim]
            llm_emb_2: Tensor of shape [B, llm_dim]

        Returns:
            Scalar geometry preservation loss
        """
        # Original distances in LLM space
        original_dist = F.pairwise_distance(llm_emb_1, llm_emb_2)

        # Projected distances
        proj_1 = self.forward(llm_emb_1)
        proj_2 = self.forward(llm_emb_2)
        projected_dist = F.pairwise_distance(proj_1, proj_2)

        # Normalize distances to [0, 1] range for fair comparison
        original_dist_norm = original_dist / (original_dist.max() + 1e-8)
        projected_dist_norm = projected_dist / (projected_dist.max() + 1e-8)

        return F.mse_loss(projected_dist_norm, original_dist_norm)
