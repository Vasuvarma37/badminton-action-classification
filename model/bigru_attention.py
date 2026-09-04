"""
model/bigru_attention.py

Bidirectional LSTM with Additive Attention for badminton action classification.

Architecture

    Input (B, T, F)           B=batch, T=30 frames, F=26 features (13 joints  2)
        
    LayerNorm(F)              stabilise inputs across feature dim
        
    BiGRU 3 stacked         each layer: hidden=128, bidirectional  out=256
        
    Attention                 additive (Bahdanau-style), context over time steps
        
    FC(512  128) + GELU
        
    Dropout(p)
        
    FC(128  6)               logits for 6 action classes

Why BiGRU over LSTM?
    A forward LSTM only sees past frames; a BiGRU additionally sees future
    frames in the reverse pass, giving a richer hidden state that captures
    the full motion arc of each shot  crucial for strokes like forehand
    clear vs drive that share early motion but diverge in follow-through.

Why 30 frames?
    10 frames captures only a slice of a stroke; 30 frames spans the full
    swing arc from preparation to recovery, giving the model temporal
    context to distinguish similar-looking strokes (e.g., forehand_lift
    vs forehand_clear).

Why Attention?
    Not all 30 frames carry equal information (e.g., the impact frame is
    more discriminative than a recovery frame). Attention learns a soft
    weighting over time steps, yielding a weighted context vector that
    focuses the classifier on the most informative frames.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
    DROPOUT, FC_HIDDEN, NUM_CLASSES, FRAMES_PER_VIDEO,
)


# 
# Additive Attention (Bahdanau-style)
# 

class AttentionLayer(nn.Module):
    """
    Soft attention over a sequence of hidden states.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the input hidden states.

    Input  : (B, T, H)    batch, time-steps, hidden dim
    Output : (B, H)       weighted context vector
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.energy = nn.Linear(hidden_dim, 1,          bias=False)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden : (B, T, H)

        Returns
        -------
        context : (B, H)          weighted sum of hidden states
        weights : (B, T)          attention weights (for inspection)
        """
        # Query: use mean-pooled state as global context
        q = self.query(hidden.mean(dim=1, keepdim=True))   # (B, 1, H)
        k = self.key(hidden)                               # (B, T, H)
        scores = self.energy(torch.tanh(q + k))            # (B, T, 1)
        weights = F.softmax(scores, dim=1)                 # (B, T, 1)
        context = (weights * hidden).sum(dim=1)            # (B, H)
        return context, weights.squeeze(-1)


# 
# Main Model
# 

class BadmintonBiGRU(nn.Module):
    """
    3-layer Bidirectional LSTM + Attention classifier.

    Parameters
    ----------
    input_size  : int   number of features per frame (default 66)
    hidden_size : int   LSTM hidden units per direction (default 128)
    num_layers  : int   stacked BiGRU layers (default 3)
    num_classes : int   output action categories (default 4)
    dropout     : float
    fc_hidden   : int   intermediate FC dimension
    """

    def __init__(
        self,
        input_size:  int   = INPUT_SIZE,
        hidden_size: int   = HIDDEN_SIZE,
        num_layers:  int   = NUM_LAYERS,
        num_classes: int   = NUM_CLASSES,
        dropout:     float = DROPOUT,
        fc_hidden:   int   = FC_HIDDEN,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        #  Input normalisation 
        self.input_norm = nn.LayerNorm(input_size)

        #  Bidirectional LSTM stack 
        # bidirectional=True doubles the effective hidden dim  *2
        self.bigru = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )

        bigru_out_dim = hidden_size * 2   # 256 with default settings

        #  Attention 
        self.attention = AttentionLayer(bigru_out_dim)

        #  Classifier head 
        self.classifier = nn.Sequential(
            nn.Linear(bigru_out_dim, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        for name, param in self.bigru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # LSTM forget-gate bias trick: initialise to 1 for better
                # gradient flow in early training
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x               : (B, T, F)
        return_attention: bool  if True, also return attention weights

        Returns
        -------
        logits : (B, num_classes)
        weights (optional) : (B, T)
        """
        x = self.input_norm(x)                       # (B, T, F)
        out, _ = self.bigru(x)                      # (B, T, 2*H)
        context, weights = self.attention(out)        # (B, 2*H), (B, T)
        logits = self.classifier(context)             # (B, C)

        if return_attention:
            return logits, weights
        return logits

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 
# Quick sanity check
# 
if __name__ == "__main__":
    model = BadmintonBiGRU()
    dummy = torch.randn(8, FRAMES_PER_VIDEO, INPUT_SIZE)   # batch=8, T=30 frames, F=26
    logits, attn = model(dummy, return_attention=True)
    print(f"Model   : BadmintonBiGRU")
    print(f"Params  : {model.count_parameters():,}")
    print(f"Input   : {dummy.shape}")
    print(f"Output  : {logits.shape}")      # (8, 6)
    print(f"Attention: {attn.shape}")       # (8, 30)
    print("[OK] Model forward pass successful.")
