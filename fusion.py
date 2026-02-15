"""
Preference Fusion Module for Mamba4Rec

This module implements gated fusion between:
- s_mamba: Sequential preference signal from Mamba layers
- m_current: Current mood/intent signal from LLM
- p_profile: Long-term user profile signal

The fusion learns to adaptively weight these signals based on their content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreferenceFusion(nn.Module):
    """
    Gated fusion of Mamba sequence output with LLM-derived signals.

    Implements the fusion formula:
        fused = α * s_mamba + (1 - α) * (m_current + p_profile)

    where α is learned from the concatenated inputs.

    Args:
        hidden_size: Dimension of all input vectors
        dropout: Dropout rate (default: 0.1)
        vector_gate: If True, use per-dimension gating. If False, use scalar gate.
    """

    def __init__(self, hidden_size, dropout=0.1, vector_gate=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.vector_gate = vector_gate

        # Output dimension of gate: 1 for scalar, hidden_size for vector
        gate_out_dim = hidden_size if vector_gate else 1

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, gate_out_dim),
            nn.Sigmoid()
        )

        # Output normalization
        self.norm = nn.LayerNorm(hidden_size)

        # Initialize gate bias to favor Mamba output initially
        # This ensures stable behavior when LLM signals are weak
        self._init_gate_bias()

    def _init_gate_bias(self):
        """Initialize gate to slightly favor Mamba output (α ≈ 0.6)."""
        with torch.no_grad():
            # Access the final Linear layer's bias
            final_linear = self.gate[-2]  # Second to last (before Sigmoid)
            if hasattr(final_linear, 'bias') and final_linear.bias is not None:
                # Sigmoid(0.4) ≈ 0.6, so we bias toward Mamba
                final_linear.bias.fill_(0.4)

    def forward(
        self,
        s_mamba,        # [B, d]
        m_current=None, # [B, d]
        p_profile=None  # [B, d]
    ):
        """
        Fuse Mamba sequence output with LLM-derived signals.

        Args:
            s_mamba: Sequence output from Mamba4Rec [B, hidden_size]
            m_current: LLM-conditioned current intent/mood vector [B, hidden_size]
            p_profile: Long-term user profile vector [B, hidden_size]

        Returns:
            Fused preference vector [B, hidden_size]
        """
        B, d = s_mamba.shape

        # Default to zeros if signals are missing
        if m_current is None:
            m_current = torch.zeros_like(s_mamba)
        if p_profile is None:
            p_profile = torch.zeros_like(s_mamba)

        # Concatenate all three signals
        fusion_input = torch.cat([s_mamba, m_current, p_profile], dim=-1)  # [B, 3d]

        # Compute gate (scalar or vector per user)
        alpha = self.gate(fusion_input)  # [B, 1] or [B, d]

        # Combine LLM signals
        llm_combined = m_current + p_profile

        # Weighted fusion
        if self.vector_gate:
            # Per-dimension weighting
            fused = alpha * s_mamba + (1 - alpha) * llm_combined
        else:
            # Scalar weighting (broadcast across dimensions)
            fused = alpha * s_mamba + (1 - alpha) * llm_combined

        return self.norm(fused)

    def get_gate_value(self, s_mamba, m_current=None, p_profile=None):
        """
        Get the gate value(s) for interpretability.

        Returns the α value(s) that show how much the model trusts
        the Mamba sequence vs. the LLM signals.

        Returns:
            alpha: Gate value(s) [B, 1] or [B, hidden_size]
        """
        B, d = s_mamba.shape

        if m_current is None:
            m_current = torch.zeros_like(s_mamba)
        if p_profile is None:
            p_profile = torch.zeros_like(s_mamba)

        fusion_input = torch.cat([s_mamba, m_current, p_profile], dim=-1)
        return self.gate(fusion_input)


class AdaptivePreferenceFusion(nn.Module):
    """
    Advanced fusion with attention-based signal weighting.

    Instead of a simple gate, this uses a small attention mechanism
    to weight the three signals, allowing for more nuanced combinations.

    Args:
        hidden_size: Dimension of all input vectors
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_dim = hidden_size // num_heads

        # Signal type embeddings (learnable identifiers for each signal type)
        self.signal_embeddings = nn.Parameter(torch.randn(3, hidden_size) * 0.02)

        # Query projection (from concatenated signals)
        self.query_proj = nn.Linear(hidden_size * 3, hidden_size)

        # Key and value projections (per signal)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )

        # Residual gate (learnable interpolation between input and attention output)
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        s_mamba,        # [B, d]
        m_current=None, # [B, d]
        p_profile=None  # [B, d]
    ):
        """
        Fuse signals using attention-based weighting.

        Args:
            s_mamba: Sequence output from Mamba4Rec [B, hidden_size]
            m_current: LLM-conditioned current intent/mood vector [B, hidden_size]
            p_profile: Long-term user profile vector [B, hidden_size]

        Returns:
            Fused preference vector [B, hidden_size]
        """
        B, d = s_mamba.shape

        # Default to zeros if signals are missing
        if m_current is None:
            m_current = torch.zeros_like(s_mamba)
        if p_profile is None:
            p_profile = torch.zeros_like(s_mamba)

        # Stack signals: [B, 3, d]
        signals = torch.stack([s_mamba, m_current, p_profile], dim=1)

        # Add signal type embeddings
        signals = signals + self.signal_embeddings.unsqueeze(0)

        # Compute query from concatenated signals
        concat_signals = torch.cat([s_mamba, m_current, p_profile], dim=-1)  # [B, 3d]
        query = self.query_proj(concat_signals).unsqueeze(1)  # [B, 1, d]

        # Compute keys and values
        keys = self.key_proj(signals)    # [B, 3, d]
        values = self.value_proj(signals)  # [B, 3, d]

        # Reshape for multi-head attention
        # query: [B, num_heads, 1, head_dim]
        # keys, values: [B, num_heads, 3, head_dim]
        query = query.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B, 3, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, 3, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(query, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # [B, num_heads, 1, 3]

        # Weighted combination
        attended = torch.matmul(attn_weights, values)  # [B, num_heads, 1, head_dim]

        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(B, d)  # [B, d]

        # Output projection
        output = self.output_proj(attended)

        # Residual connection with learnable gate
        gate_input = torch.cat([s_mamba, output], dim=-1)
        gate = self.residual_gate(gate_input)
        fused = gate * s_mamba + (1 - gate) * output

        return fused

    def get_attention_weights(self, s_mamba, m_current=None, p_profile=None):
        """
        Get attention weights for interpretability.

        Returns attention weights showing how much each signal contributes.

        Returns:
            weights: Attention weights [B, num_heads, 1, 3]
                     Last dimension: [mamba_weight, current_weight, profile_weight]
        """
        B, d = s_mamba.shape

        if m_current is None:
            m_current = torch.zeros_like(s_mamba)
        if p_profile is None:
            p_profile = torch.zeros_like(s_mamba)

        signals = torch.stack([s_mamba, m_current, p_profile], dim=1)
        signals = signals + self.signal_embeddings.unsqueeze(0)

        concat_signals = torch.cat([s_mamba, m_current, p_profile], dim=-1)
        query = self.query_proj(concat_signals).unsqueeze(1)

        keys = self.key_proj(signals)

        query = query.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B, 3, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        return F.softmax(scores, dim=-1)


class TemporalPreferenceFusion(nn.Module):
    """
    Fusion that considers temporal dynamics of mood signals.

    This variant can process a sequence of mood vectors from a conversation,
    using a small transformer to aggregate temporal context before fusion.

    Args:
        hidden_size: Dimension of all input vectors
        max_mood_history: Maximum number of mood vectors to consider
        num_layers: Number of transformer layers for mood aggregation
        dropout: Dropout rate
    """

    def __init__(self, hidden_size, max_mood_history=5, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_mood_history = max_mood_history

        # Positional encoding for mood history
        self.position_emb = nn.Embedding(max_mood_history, hidden_size)

        # Small transformer to aggregate mood history
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True
        )
        self.mood_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Standard fusion after aggregation
        self.fusion = PreferenceFusion(hidden_size, dropout=dropout, vector_gate=True)

    def forward(
        self,
        s_mamba,           # [B, d]
        mood_history=None, # [B, T, d] where T <= max_mood_history
        p_profile=None     # [B, d]
    ):
        """
        Fuse with temporally-aggregated mood signal.

        Args:
            s_mamba: Sequence output from Mamba4Rec [B, hidden_size]
            mood_history: History of mood vectors [B, T, hidden_size]
            p_profile: Long-term user profile vector [B, hidden_size]

        Returns:
            Fused preference vector [B, hidden_size]
        """
        B, d = s_mamba.shape

        if mood_history is None:
            m_current = None
        else:
            T = mood_history.shape[1]

            # Add positional encoding
            positions = torch.arange(T, device=mood_history.device)
            mood_history = mood_history + self.position_emb(positions).unsqueeze(0)

            # Aggregate mood history with transformer
            aggregated = self.mood_encoder(mood_history)

            # Take the last position as the current aggregated mood
            m_current = aggregated[:, -1, :]  # [B, d]

        return self.fusion(s_mamba, m_current, p_profile)
