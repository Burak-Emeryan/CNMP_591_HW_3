"""Conditional Neural Movement Primitive (CNMP) model.

Architecture  as 
  Encoder : MLP(5 → 128 → 128 → 128, ReLU)
  Aggregator : mean pool over context points → r ∈ R^128
  Decoder : MLP(130 → 128 → 128 → 8, ReLU)
    Input = [target_t (1) | r (128) | h (1)] = 130
    Output = [mean (4) | log_var (4)] = 8

The condition h (object height) is fed ONLY to the decoder.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encodes each context point [t, e_y, e_z, o_y, o_z] into a latent vector."""

    def __init__(self, in_dim: int = 5, hidden_dim: int = 128, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (..., 5)

        Returns
        -------
        Tensor of shape (..., 128)
        """
        return self.net(x)


class Decoder(nn.Module):
    """Decodes [target_t, r, h] into predicted means and log-variances."""

    def __init__(self, in_dim: int = 130, hidden_dim: int = 128, out_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (..., 130)

        Returns
        -------
        Tensor of shape (..., 8)
        """
        return self.net(x)


class CNMP(nn.Module):
    """Conditional Neural Movement Primitive.

    Forward pass:
      1. Encode each context point → (B, N_ctx, r_dim)
      2. Mean-pool over context → r of shape (B, r_dim)
      3. Expand r and h to match N_tgt, concatenate [target_t, r, h] → (B, N_tgt, 130)
      4. Decode → (B, N_tgt, 8), split into mean (4) and log_var (4)
      5. Clamp log_var to [-10, 2]
    """

    def __init__(self, r_dim: int = 128) -> None:
        super().__init__()
        self.encoder = Encoder(in_dim=5, hidden_dim=128, out_dim=r_dim)
        self.decoder = Decoder(in_dim=1 + r_dim + 1, hidden_dim=128, out_dim=8)

    def forward(
        self,
        context: torch.Tensor,
        target_t: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple:
        """
        Parameters
        ----------
        context : Tensor (B, N_ctx, 5)
            Context points [t, e_y, e_z, o_y, o_z].
        target_t : Tensor (B, N_tgt, 1)
            Query time steps for target predictions.
        h : Tensor (B, 1)
            Object height condition (one scalar per batch item).

        Returns
        -------
        mean : Tensor (B, N_tgt, 4)
            Predicted means for [e_y, e_z, o_y, o_z].
        log_var : Tensor (B, N_tgt, 4)
            Predicted log-variances, clamped to [-10, 2].
        """
        n_tgt = target_t.shape[1]

        # 1. Encode context points: (B, N_ctx, 5) -> (B, N_ctx, 128)
        encoded = self.encoder(context)

        # 2. Meanpool over context dimension: (B, N_ctx, 128) -> (B, 128)
        r = encoded.mean(dim=1)

        # 3. Expand r and h to match target count
        r_expanded = r.unsqueeze(1).expand(-1, n_tgt, -1)   # (B, N_tgt, 128)
        h_expanded = h.unsqueeze(1).expand(-1, n_tgt, -1)   # (B, N_tgt, 1)

        # 4. Concatenate [target_t, r_expanded, h_expanded] -> (B, N_tgt, 130)
        decoder_input = torch.cat([target_t, r_expanded, h_expanded], dim=-1)

        # 5. Decode -> (B, N_tgt, 8)
        out = self.decoder(decoder_input)

        # 6. Split into mean and log_var
        mean = out[..., :4]
        log_var = out[..., 4:]

        # 7. Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-10.0, max=2.0)

        return mean, log_var


if __name__ == "__main__":
    model = CNMP()
    B, N_ctx, N_tgt = 2, 5, 10
    ctx = torch.randn(B, N_ctx, 5)
    tgt_t = torch.randn(B, N_tgt, 1)
    h = torch.randn(B, 1)
    mean, log_var = model(ctx, tgt_t, h)
    assert mean.shape == (B, N_tgt, 4), mean.shape
    assert log_var.shape == (B, N_tgt, 4), log_var.shape
    print("shape test passed")
    print(f"  model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  mean range:    [{mean.min().item():.4f}, {mean.max().item():.4f}]")
    print(f"  log_var range: [{log_var.min().item():.4f}, {log_var.max().item():.4f}]")
