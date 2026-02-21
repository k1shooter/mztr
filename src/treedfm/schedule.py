
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal, Optional
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

        NOTE: clipping psi breaks the identity psi = kappa_dot/(1-kappa) unless you
            also change kappa accordingly. In practice this can destabilize learning
            near the end of the ramp (kappa -> 1). Prefer an exponential-ramp schedule
            (ExpProfiledDepthSchedule) which keeps psi bounded *by construction*.
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

@dataclass
class DepthProfile:
    """Summary statistics used to build a profiled depth schedule.

    Attributes
    ----------
    counts:
        A length-(max_depth+1) tensor. counts[d] is the (estimated) number of
        *relevant coordinates* at depth d.

        - If count_mode == "nodes": number of nodes at depth d.
        - If count_mode == "all_slots": number of parent-slots that *could* create
          a child at depth d, i.e. k * (#nodes at depth d-1).

    max_depth:
        Max depth used to clip counts.
    """

    counts: torch.Tensor
    max_depth: int


class ProfiledDepthSchedule:
    """Piecewise-linear depth schedule with *depth-dependent* windows.

    This schedule is designed to be "plug-compatible" with DepthStratifiedSchedule
    (implements kappa/psi/kappa_and_psi and exposes max_depth).

    For each depth d we define a start time s[d] and width w[d] (both in [0,1]):

        kappa_d(t) = clamp((t - s[d]) / w[d], 0, 1)

    with the convention that if w[d] is very small the transition is almost
    instantaneous.

    psi(t,d) is computed as kappa_dot/(1-kappa) inside the active window
    [s[d], s[d] + w[d]) and clipped for numerical stability.
    """

    def __init__(
        self,
        *,
        starts: torch.Tensor,
        widths: torch.Tensor,
        max_psi: float = 200.0,
        mode: str = "linear",   # "linear" (default) or "exp"
        exp_eps: float = 1e-3,  # exp mode: kappa reaches ~1-exp_eps at window end (before clamp)
    ):
        if starts.dim() != 1 or widths.dim() != 1 or starts.numel() != widths.numel():
            raise ValueError("starts/widths must be 1D tensors of the same length")
        if (widths <= 0).any():
            raise ValueError("all widths must be > 0")

        self.starts = starts.detach().float().clone()  # [D+1]
        self.widths = widths.detach().float().clone()  # [D+1]
        self.max_depth = int(self.starts.numel() - 1)
        self.max_psi = float(max_psi)
        if mode not in ("linear", "exp"):
            raise ValueError(f"Unknown mode: {mode} (expected 'linear' or 'exp')")
        self.mode = mode
        self.exp_eps = float(exp_eps)
        if not (0.0 < self.exp_eps < 1.0):
            raise ValueError("exp_eps must be in (0,1)")

    def _gather(self, depths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather start/width per depth index and broadcast to `depths` shape."""
        device = depths.device
        d = depths.clamp(min=0, max=self.max_depth).long()
        s = self.starts.to(device)[d]
        w = self.widths.to(device)[d]
        return s, w

    def kappa(self, t: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        if t.dim() != 1:
            raise ValueError("t must be shape [B]")
        # Broadcast t to match depths
        t_b = t.to(depths.device).view(-1, *([1] * (depths.dim() - 1))).expand_as(depths).float()
        s, w = self._gather(depths)
        if self.mode == "linear":
            progress = (t_b - s) / w
            return torch.clamp(progress, 0.0, 1.0)

        # exp mode: constant-hazard ramp inside [s, s+w), clamp to 1 afterwards.
        dt = t_b - s
        before = dt < 0
        after = dt >= w
        in_window = (~before) & (~after)

        log_inv_eps = torch.log(torch.tensor(1.0 / self.exp_eps, device=depths.device, dtype=torch.float32))
        beta = log_inv_eps / torch.clamp(w, min=1e-6)   # hazard
        beta = torch.clamp(beta, max=self.max_psi)

        k_win = 1.0 - torch.exp(-beta * torch.clamp(dt, min=0.0))
        k = torch.zeros_like(dt)
        k = torch.where(in_window, k_win, k)
        k = torch.where(after, torch.ones_like(k), k)
        return k

    def psi(self, t: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        if t.dim() != 1:
            raise ValueError("t must be shape [B]")
        t_b = t.to(depths.device).view(-1, *([1] * (depths.dim() - 1))).expand_as(depths).float()
        s, w = self._gather(depths)
        t_end = s + w
        in_window = (t_b >= s) & (t_b < t_end)

        if self.mode == "linear":
            kappa = self.kappa(t, depths)
            denom = torch.clamp(1.0 - kappa, min=1e-6)
            kdot = (1.0 / w) * in_window.float()
            psi = kdot / denom
            psi = torch.clamp(psi, 0.0, self.max_psi)
            psi = psi * in_window.float()
            return psi

        # exp mode: psi is constant hazard beta inside the window
        log_inv_eps = torch.log(torch.tensor(1.0 / self.exp_eps, device=depths.device, dtype=torch.float32))
        beta = log_inv_eps / torch.clamp(w, min=1e-6)
        beta = torch.clamp(beta, max=self.max_psi)
        return beta * in_window.float()

    def kappa_and_psi(self, t: torch.Tensor, depths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kappa(t, depths), self.psi(t, depths)

class SplitKappaSchedule:
    """Wrap two schedules: one for existence (blank vs token), one for type correctness.

    This is a small utility so you can keep the rest of the code mostly unchanged.

    - kappa()/psi() are aliased to the *existence* schedule for backward compat.
    - kappa_exist/psi_exist and kappa_type/psi_type are exposed explicitly.
    """

    def __init__(self, exist: TreeSchedule, typ: TreeSchedule | None = None):
        self.exist = exist
        self.typ = typ if typ is not None else exist
        self.max_depth = int(getattr(exist, "max_depth", 128))

    # Back-compat
    def kappa(self, t: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        return self.exist.kappa(t, depths)

    def psi(self, t: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        return self.exist.psi(t, depths)

    # Explicit split
    def kappa_exist(self, t: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        return self.exist.kappa(t, depths)

    def psi_exist(self, t: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        return self.exist.psi(t, depths)

    def kappa_type(self, t: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        return self.typ.kappa(t, depths)

    def psi_type(self, t: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        return self.typ.psi(t, depths)

    def psi_ops(self, t: torch.Tensor, depths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(psi_ins, psi_del, psi_sub) at (t,depth). Default: ins/del use exist, sub uses type."""
        psi_e = self.psi_exist(t, depths)
        psi_y = self.psi_type(t, depths)
        return psi_e, psi_e, psi_y

CountMode = Literal["nodes", "all_slots"]


def estimate_depth_profile(
    samples: Iterable[torch.Tensor],
    *,
    max_depth: int,
    k: int = 3,
    count_mode: CountMode = "nodes",
    include_root: bool = False,
) -> DepthProfile:
    """Estimate depth counts n_d from a dataset or list of [Ni,4] tensors.

    Parameters
    ----------
    samples:
        Iterable of tensors shaped [Ni,4] with columns (depth, rank, type, parent_idx).
        This is exactly what MazeTreeDataset returns.

    max_depth:
        Maximum depth to count (depths are clipped to this).

    k:
        Branching factor (only used for count_mode="all_slots").

    count_mode:
        - "nodes": counts[d] = #nodes at depth d
        - "all_slots": counts[d] = k * #nodes at depth (d-1)
          (the number of parent-slots that could generate a child at depth d)

    include_root:
        Whether to include depth 0 in the returned counts.

    Returns
    -------
    DepthProfile(counts, max_depth)
    """

    md = int(max_depth)
    counts = torch.zeros((md + 1,), dtype=torch.float64)

    # First pass: count nodes per depth
    for x in samples:
        if x.dim() != 2 or x.size(-1) != 4:
            raise ValueError(f"Expected sample [N,4], got {tuple(x.shape)}")
        d = x[:, 0].clamp(min=0, max=md).to(torch.long)
        # In these maze datasets, all rows are real nodes (no padding here).
        # Still, if someone passes padded tensors, we can optionally ignore type==0.
        if (x[:, 2] == 0).any():
            d = d[x[:, 2] > 0]
        if d.numel() == 0:
            continue
        binc = torch.bincount(d, minlength=md + 1).to(torch.float64)
        counts += binc

    # IMPORTANT: `include_root=False` should *not* erase the influence of the root on
    # depth-1 slot counts when `count_mode="all_slots"`.
    #
    # We therefore keep a separate `node_counts` (with the root included), then apply
    # `include_root` only to the "nodes" view.
    node_counts = counts.clone()
    if count_mode == "nodes":
        counts = node_counts
        if not include_root:
            counts[0] = 0.0
    elif count_mode == "all_slots":
        # slot_mass[d] = k * node_mass[d-1]
        counts = torch.zeros_like(node_counts)
        counts[1:] = float(k) * node_counts[:-1]
        counts[0] = 0.0
        # `include_root` is irrelevant for slot_mass (depth 0 has no parent-slots),
        # but the root *must* contribute to depth-1.
    else:
        raise ValueError(f"Unknown count_mode: {count_mode}")


    return DepthProfile(counts=counts.to(torch.float32), max_depth=md)


def make_profiled_sequential_schedule(
    profile: DepthProfile,
    *,
    smoothing: float = 1.0,
    power: float = 1.0,
    min_width: float = 1e-3,
    max_psi: float = 200.0,
    t0: float = 0.0,
    t1: float = 1.0,
    include_root: bool = False,
    overlap: float = 0.0,
    mode: str = "linear",
    exp_eps: float = 1e-3,
) -> ProfiledDepthSchedule:
    """Build a *sequential* depth schedule from a depth profile.

    The idea is to allocate a time window to each depth proportionally to the
    depth "mass" n_d so that the total expected flux is roughly uniform:

        sum_d n_d * kappa_dot_d(t) \approx const.

    We construct piecewise-linear ramps per depth:
      - depth d is inactive before its start s[d]
      - ramps linearly to 1 over width w[d]
      - stays at 1 afterwards

    Parameters
    ----------
    smoothing:
        Additive smoothing added to counts before normalization.
        Useful to avoid zero-width depths.

    power:
        Optional tempering on the counts: w_d \propto (n_d + smoothing)^power.
        power < 1 compresses the dynamic range.

    min_width:
        Lower bound on each width (in time units). Ensures finite kappa_dot.

    include_root:
        If False (default), depth 0 gets a tiny width and start=0 but does not
        affect normalization. Root is typically forced to exist anyway.

    overlap:
        Fraction in [0,1). If > 0, we allow adjacent depth windows to overlap.
        This can soften the curriculum at the cost of increased projection error.

    Returns
    -------
    ProfiledDepthSchedule
    """

    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0,1)")
    if t1 <= t0:
        raise ValueError("t1 must be > t0")
    md = int(profile.max_depth)
    counts = profile.counts.detach().float().clone()
    if counts.numel() != md + 1:
        raise ValueError("profile.counts must have length max_depth+1")

    # Depths we actually schedule (by default: 1..max_depth)
    sched_mask = torch.ones_like(counts, dtype=torch.bool)
    if not include_root:
        sched_mask[0] = False

    # ---- NEW: trim depths beyond the last depth that actually appears in the dataset ----
    # If we don't do this, profiled schedules can waste time budget on depths that never occur,
    # shrinking early-depth windows (depth 1/2) and making insertion learning too rare.
    raw = counts.clone()
    if not include_root:
        raw[0] = 0.0
    nz = torch.nonzero(raw > 0, as_tuple=False)
    if nz.numel() > 0:
        last = int(nz[-1].item())
        # deactivate depths > last
        sched_mask[(last + 1):] = False

    # Smoothing + tempering
    w = torch.zeros_like(counts)
    w[sched_mask] = torch.clamp(counts[sched_mask] + float(smoothing), min=0.0) ** float(power)

    total = float(w[sched_mask].sum().item())
    if total <= 0:
        # Fallback: uniform over depths 1..md
        w[sched_mask] = 1.0
        total = float(w[sched_mask].sum().item())

    # Normalize to time budget
    budget = float(t1 - t0)
    w = w / total
    widths = torch.clamp(w * budget, min=float(min_width))

    # Renormalize widths to exactly match budget (preserving min_width constraints).
    # If min_width makes it impossible (too many active depths), we still do a best-effort scale.
    active = sched_mask & (widths > 0)
    sum_w = float(widths[active].sum().item())
    if sum_w > 0:
        scale = budget / sum_w
        widths[active] = widths[active] * scale

    # Build start times sequentially
    starts = torch.zeros_like(widths)
    cur = float(t0)
    for d in range(md + 1):
        if not active[d]:
            starts[d] = float(t1)  # effectively never turns on
            continue
        starts[d] = cur
        cur += float(widths[d].item())

    # Force the last scheduled depth to end exactly at t1
    # (numerical drift from float ops can leave a small gap).
    last_d = int(torch.nonzero(active, as_tuple=False)[-1].item())
    end_last = float(starts[last_d].item() + widths[last_d].item())
    if abs(end_last - float(t1)) > 1e-6:
        widths[last_d] = max(float(min_width), widths[last_d].item() + (float(t1) - end_last))

    # Optional overlap: shift starts backward by overlap fraction of previous width.
    if overlap > 0:
        prev_width = 0.0
        for d in range(md + 1):
            if not active[d]:
                continue
            starts[d] = max(float(t0), float(starts[d].item()) - overlap * prev_width)
            prev_width = float(widths[d].item())

    # Root handling: if excluded, keep it "always on" by setting start=0 and width=min_width.
    if not include_root:
        starts[0] = float(t0)
        widths[0] = float(min_width)

    return ProfiledDepthSchedule(starts=starts, widths=widths, max_psi=max_psi, mode=mode, exp_eps=exp_eps)

class TimeOnlySchedule:
    """Depth-agnostic (time-only) schedule.

    This ablation schedule ignores `depths` and applies the same kappa(t) / psi(t)
    to every coordinate, then relies on the same *projection* step to enforce
    tree closure (if parent absent => child absent).

    We provide both:
      - linear mode: kappa(t)=clamp(t/width,0,1) with psi = kdot/(1-kappa) in-window
      - exp mode: constant-hazard ramp inside [0,width): kappa(t)=1-exp(-beta*t),
                  psi(t)=beta (bounded by construction), and kappa=1 after width.

    Notes
    -----
    - `max_depth` is only used to set the projection loop count (for compat).
    - For fair comparisons with depth-aware profiled schedules, consider setting
      width=1.0 so the mixture spans the full [0,1] time interval.
    """

    def __init__(
        self,
        *,
        max_depth: int,
        width: float = 1.0,
        max_psi: float = 200.0,
        mode: str = "linear",   # "linear" or "exp"
        exp_eps: float = 1e-3,
    ):
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if not (0.0 < float(width) <= 1.0):
            raise ValueError("width must be in (0,1]")
        if mode not in ("linear", "exp"):
            raise ValueError("mode must be 'linear' or 'exp'")
        if not (0.0 < float(exp_eps) < 1.0):
            raise ValueError("exp_eps must be in (0,1)")
        self.max_depth = int(max_depth)
        self.width = float(width)
        self.max_psi = float(max_psi)
        self.mode = str(mode)
        self.exp_eps = float(exp_eps)

    def _broadcast_t(self, t, ref):
        if t.dim() != 1:
            raise ValueError("t must be shape [B]")
        return t.to(ref.device).view(-1, *([1] * (ref.dim() - 1))).expand_as(ref).float()

    def kappa(self, t, depths):
        import torch
        t_b = self._broadcast_t(t, depths)
        if self.mode == "linear":
            return torch.clamp(t_b / self.width, 0.0, 1.0)

        # exp mode: constant hazard beta inside [0,width), then clamp to 1.
        dt = t_b
        after = dt >= self.width
        in_window = (~after)

        log_inv_eps = torch.log(torch.tensor(1.0 / self.exp_eps, device=depths.device, dtype=torch.float32))
        beta = log_inv_eps / torch.clamp(torch.tensor(self.width, device=depths.device), min=1e-6)
        beta = torch.clamp(beta, max=self.max_psi)

        k_win = 1.0 - torch.exp(-beta * torch.clamp(dt, min=0.0))
        k = torch.where(in_window, k_win, torch.zeros_like(dt))
        k = torch.where(after, torch.ones_like(k), k)
        return k

    def psi(self, t, depths):
        import torch
        t_b = self._broadcast_t(t, depths)
        in_window = (t_b >= 0.0) & (t_b < self.width)

        if self.mode == "linear":
            kappa = self.kappa(t, depths)
            denom = torch.clamp(1.0 - kappa, min=1e-6)
            kdot = (1.0 / self.width) * in_window.float()
            psi = kdot / denom
            psi = torch.clamp(psi, 0.0, self.max_psi)
            return psi * in_window.float()

        # exp mode: psi is constant hazard beta inside the window
        log_inv_eps = torch.log(torch.tensor(1.0 / self.exp_eps, device=depths.device, dtype=torch.float32))
        beta = log_inv_eps / torch.clamp(torch.tensor(self.width, device=depths.device), min=1e-6)
        beta = torch.clamp(beta, max=self.max_psi)
        return beta * in_window.float()

    def kappa_and_psi(self, t, depths):
        return self.kappa(t, depths), self.psi(t, depths)