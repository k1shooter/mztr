
from __future__ import annotations

import torch


class DepthStratifiedSchedule:
    """
    Depth-stratified (piecewise linear) schedule used to:
    - control how quickly deeper nodes/slots "turn on"
    - provide psi(t,d) = kappa_dot/(1-kappa) used by EditFlow loss

    kappa(t,d) = clamp((t - t_start(d))/width, 0, 1)
    where t_start(d) = d / max_depth.

    This is intentionally simple and easy to debug.
    """

    def __init__(self, max_depth: int, width: float = 0.5, max_psi: float = 200.0):
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if not (0.0 < width <= 1.0):
            raise ValueError("width must be in (0, 1]")
        self.max_depth = int(max_depth)
        self.width = float(width)
        self.max_psi = float(max_psi)

    def kappa(self, t: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        """
        t: [B] in [0,1]
        depths: [B,N] or [B,N,K] integer depths
        return kappa: same shape as depths
        """
        if t.dim() != 1:
            raise ValueError("t must be shape [B]")
        t = t.to(depths.device).view(-1, *([1] * (depths.dim() - 1))).expand_as(depths).float()
        d = depths.float()
        t_start = d / float(self.max_depth)
        progress = (t - t_start) / self.width
        return torch.clamp(progress, 0.0, 1.0)

    def psi(self, t: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        """
        psi(t,d) = kappa_dot / (1 - kappa)
        with kappa_dot = 1/width in the active window and 0 outside.
        We clip psi to max_psi for numerical stability.
        """
        if t.dim() != 1:
            raise ValueError("t must be shape [B]")
        t_b = t.to(depths.device).view(-1, *([1] * (depths.dim() - 1))).expand_as(depths).float()
        d = depths.float()
        t_start = d / float(self.max_depth)
        t_end = t_start + self.width
        in_window = (t_b >= t_start) & (t_b < t_end)

        # kappa in the window is linear; outside it's clamped.
        kappa = self.kappa(t, depths)
        denom = torch.clamp(1.0 - kappa, min=1e-6)
        kdot = (1.0 / self.width) * in_window.float()
        psi = kdot / denom
        psi = torch.clamp(psi, 0.0, self.max_psi)
        # Explicitly 0 outside the window
        psi = psi * in_window.float()
        return psi

    def kappa_and_psi(self, t: torch.Tensor, depths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kappa(t, depths), self.psi(t, depths)
