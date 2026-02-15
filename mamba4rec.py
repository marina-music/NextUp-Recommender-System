"""
Mamba4Rec with LLM Fusion

This module extends the original Mamba4Rec model to support fusion with
LLM-derived mood and profile vectors for enhanced personalization.

Key modifications from original:
1. Integration of PreferenceFusion module
2. Integration of LLMProjection module
3. Fusion is applied in calculate_loss() for training
4. Support for phased training with layer freezing
5. Additional methods for interpretability
"""

import torch
from torch import nn
from mamba_ssm import Mamba
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

from fusion import PreferenceFusion, AdaptivePreferenceFusion
from llm_projection import LLMProjection, LLMProjectionWithAlignment


class Mamba4RecFusion(SequentialRecommender):
    """
    Mamba4Rec with LLM Fusion for mood-aware recommendations.

    This model combines:
    - Sequential behavior modeling via Mamba layers
    - Current mood/intent from LLM embeddings
    - Long-term user profiles

    Args:
        config: RecBole config dictionary
        dataset: RecBole dataset object
    """

    def __init__(self, config, dataset):
        super(Mamba4RecFusion, self).__init__(config, dataset)

        # Core hyperparameters
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]

        # Mamba hyperparameters
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]

        # LLM Fusion hyperparameters
        self.use_llm_fusion = config.get("use_llm_fusion", False)
        self.llm_dim = config.get("llm_dim", 768)
        self.fusion_dropout = config.get("fusion_dropout", 0.1)
        self.vector_gate = config.get("vector_gate", False)

        # Item embeddings
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )

        # Pre-Mamba normalization
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])

        # LLM Fusion components (only created if enabled)
        if self.use_llm_fusion:
            # LLM Projection: maps LLM embeddings to hidden_size
            self.llm_projection = LLMProjectionWithAlignment(
                llm_dim=self.llm_dim,
                hidden_size=self.hidden_size,
                dropout=self.fusion_dropout
            )

            # Preference Fusion: combines Mamba output with LLM signals
            self.fusion = PreferenceFusion(
                hidden_size=self.hidden_size,
                dropout=self.fusion_dropout,
                vector_gate=self.vector_gate
            )
        else:
            self.llm_projection = None
            self.fusion = None

        # Loss function
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # Initialize weights
        self.apply(self._init_weights)

        # Training phase tracking
        self._current_phase = 1

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        """
        Forward pass through Mamba layers.

        Args:
            item_seq: Item sequence tensor [B, L]
            item_seq_len: Sequence lengths [B]

        Returns:
            seq_output: Sequence representation [B, hidden_size]
        """
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)

        for i in range(self.num_layers):
            item_emb = self.mamba_layers[i](item_emb)

        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output

    def _apply_fusion(self, seq_output, interaction):
        """
        Apply LLM fusion if enabled and signals are available.

        Args:
            seq_output: Mamba sequence output [B, hidden_size]
            interaction: Interaction dict potentially containing LLM embeddings

        Returns:
            fused_output: Fused representation [B, hidden_size]
        """
        if not self.use_llm_fusion or self.fusion is None:
            return seq_output

        # Get raw LLM embeddings from interaction
        raw_mood_emb = interaction.get("LLM_MOOD_EMB", None)
        raw_profile_emb = interaction.get("LLM_PROFILE_EMB", None)

        # Project raw LLM embeddings to hidden_size
        m_current = None
        p_profile = None

        if raw_mood_emb is not None:
            m_current = self.llm_projection(raw_mood_emb)

        if raw_profile_emb is not None:
            p_profile = self.llm_projection(raw_profile_emb)

        # Apply fusion
        return self.fusion(
            s_mamba=seq_output,
            m_current=m_current,
            p_profile=p_profile
        )

    def calculate_loss(self, interaction):
        """
        Calculate training loss with optional fusion.

        In Phase 1 (vanilla Mamba training), fusion is not applied.
        In Phase 2+, fusion is applied if LLM embeddings are present.

        Args:
            interaction: RecBole interaction dict

        Returns:
            loss: Scalar loss value
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)

        # Apply fusion during training (Phase 2+)
        if self._current_phase >= 2:
            seq_output = self._apply_fusion(seq_output, interaction)

        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # CE loss
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def calculate_loss_with_alignment(self, interaction, llm_item_emb, item_ids):
        """
        Calculate loss with additional alignment regularization.

        Used during Phase 2 to align LLM projections with item embeddings.

        Args:
            interaction: RecBole interaction dict
            llm_item_emb: LLM embeddings of item descriptions [B, llm_dim]
            item_ids: Corresponding item IDs [B]

        Returns:
            total_loss: Combined recommendation and alignment loss
        """
        # Standard recommendation loss
        rec_loss = self.calculate_loss(interaction)

        # Alignment loss (only if LLM fusion is enabled)
        if self.use_llm_fusion and self.llm_projection is not None:
            item_emb = self.item_embedding(item_ids)
            align_loss = self.llm_projection.compute_alignment_loss(
                llm_item_emb, item_emb
            )
            total_loss = rec_loss + 0.1 * align_loss
        else:
            total_loss = rec_loss

        return total_loss

    def predict(self, interaction):
        """
        Predict scores for specific items.

        Args:
            interaction: RecBole interaction dict with ITEM_ID

        Returns:
            scores: Prediction scores [B]
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = self._apply_fusion(seq_output, interaction)

        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        """
        Predict scores for all items.

        Args:
            interaction: RecBole interaction dict

        Returns:
            scores: Scores for all items [B, n_items]
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = self._apply_fusion(seq_output, interaction)

        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores

    # ================== Phased Training Methods ==================

    def set_training_phase(self, phase):
        """
        Set the current training phase and configure layer freezing.

        Phase 1: Vanilla Mamba training (no fusion, all trainable)
        Phase 2: LLM alignment (freeze Mamba, train fusion + projection)
        Phase 3: Joint fine-tuning (unfreeze top Mamba layer)

        Args:
            phase: Training phase (1, 2, or 3)
        """
        self._current_phase = phase

        if phase == 1:
            # All parameters trainable, no fusion
            self._unfreeze_all()
        elif phase == 2:
            # Freeze Mamba components, train fusion
            self._freeze_for_phase2()
        elif phase == 3:
            # Unfreeze top Mamba layer for fine-tuning
            self._configure_phase3()
        else:
            raise ValueError(f"Invalid phase: {phase}. Must be 1, 2, or 3.")

    def _unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def _freeze_for_phase2(self):
        """Freeze Mamba components for Phase 2 training."""
        # Freeze item embeddings
        self.item_embedding.weight.requires_grad = False

        # Freeze Mamba layers
        for layer in self.mamba_layers:
            for param in layer.parameters():
                param.requires_grad = False

        # Freeze LayerNorm
        for param in self.LayerNorm.parameters():
            param.requires_grad = False

        # Keep fusion and projection trainable
        if self.fusion is not None:
            for param in self.fusion.parameters():
                param.requires_grad = True

        if self.llm_projection is not None:
            for param in self.llm_projection.parameters():
                param.requires_grad = True

    def _configure_phase3(self):
        """Configure for Phase 3 fine-tuning."""
        # Start from Phase 2 configuration
        self._freeze_for_phase2()

        # Unfreeze only the top Mamba layer
        if self.num_layers > 0:
            for param in self.mamba_layers[-1].parameters():
                param.requires_grad = True

    def get_trainable_params(self):
        """Get count of trainable parameters for logging."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_frozen_params(self):
        """Get count of frozen parameters for logging."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    # ================== Interpretability Methods ==================

    def get_fusion_weights(self, interaction):
        """
        Get fusion gate values for interpretability.

        Args:
            interaction: RecBole interaction dict

        Returns:
            alpha: Gate values showing Mamba vs LLM trust
        """
        if not self.use_llm_fusion or self.fusion is None:
            return None

        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)

        raw_mood_emb = interaction.get("LLM_MOOD_EMB", None)
        raw_profile_emb = interaction.get("LLM_PROFILE_EMB", None)

        m_current = None
        p_profile = None

        if raw_mood_emb is not None:
            m_current = self.llm_projection(raw_mood_emb)

        if raw_profile_emb is not None:
            p_profile = self.llm_projection(raw_profile_emb)

        return self.fusion.get_gate_value(seq_output, m_current, p_profile)


class MambaLayer(nn.Module):
    """
    Single Mamba layer with residual connection and FFN.
    """

    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:
            # One Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:
            # Stacked Mamba layers with residual connections
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    """

    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


# Alias for backward compatibility
Mamba4Rec = Mamba4RecFusion
