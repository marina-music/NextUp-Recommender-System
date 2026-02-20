"""
Mamba4Rec - Sequential Recommendation with Mamba

Pure sequential recommender using Mamba (selective state-space) layers.
Predicts next items from user interaction history.
"""

import torch
from torch import nn
from torch.nn import functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class MambaPureTorch(nn.Module):
    """
    Pure PyTorch implementation of the Mamba selective scan block.
    Used as a fallback when mamba_ssm is not installed.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        # Input projection: x -> (z, x)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise conv
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True
        )

        # SSM projections
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # SSM parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """x: (B, L, D)"""
        B, L, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Conv
        x_inner = x_inner.transpose(1, 2)  # (B, d_inner, L)
        x_inner = self.conv1d(x_inner)[:, :, :L]  # causal: trim to L
        x_inner = x_inner.transpose(1, 2)  # (B, L, d_inner)
        x_inner = F.silu(x_inner)

        # SSM parameters from input
        x_ssm = self.x_proj(x_inner)  # (B, L, d_state*2 + 1)
        B_param = x_ssm[:, :, :self.d_state]  # (B, L, d_state)
        C_param = x_ssm[:, :, self.d_state:2*self.d_state]  # (B, L, d_state)
        dt = x_ssm[:, :, -1:]  # (B, L, 1)
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Selective scan (sequential)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            dt_t = dt[:, t]  # (B, d_inner)
            B_t = B_param[:, t]  # (B, d_state)
            C_t = C_param[:, t]  # (B, d_state)
            x_t = x_inner[:, t]  # (B, d_inner)

            # h = exp(A * dt) * h + dt * B * x
            dA = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))  # (B, d_inner, d_state)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, d_inner, d_state)
            h = dA * h + dB * x_t.unsqueeze(-1)  # (B, d_inner, d_state)

            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        y = y + x_inner * self.D.unsqueeze(0).unsqueeze(0)

        # Gate and output
        y = y * F.silu(z)
        return self.out_proj(y)


class Mamba4Rec(SequentialRecommender):
    """
    Mamba4Rec - Sequential recommendation with Mamba layers.

    Uses selective state-space (Mamba) layers to model user interaction
    sequences and predict next items.

    Args:
        config: RecBole config dictionary
        dataset: RecBole dataset object
    """

    def __init__(self, config, dataset):
        super(Mamba4Rec, self).__init__(config, dataset)

        # Core hyperparameters
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]

        # Mamba hyperparameters
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]

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

        # Loss function
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # Initialize weights
        self.apply(self._init_weights)

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

    def calculate_loss(self, interaction):
        """
        Calculate training loss.

        Args:
            interaction: RecBole interaction dict

        Returns:
            loss: Scalar loss value
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)

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

        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores

    def get_trainable_params(self):
        """Get count of trainable parameters for logging."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_frozen_params(self):
        """Get count of frozen parameters for logging."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


class MambaLayer(nn.Module):
    """
    Single Mamba layer with residual connection and FFN.
    """

    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        MambaBlock = Mamba if Mamba is not None else MambaPureTorch
        self.mamba = MambaBlock(
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
