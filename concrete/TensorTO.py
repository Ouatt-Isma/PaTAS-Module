import numpy as np
from typing import Literal

try:
    import torch
except ImportError:  # torch optional: numpy-only environments still work
    torch = None

from patas_module.subjective_logic import (
    averaging_fusion_vec,
    cumulative_fusion_vec,
    weighted_belief_fusion_vec,
    multiply_vec,
    deduce_vec,
    discount_vec,
    _normalize_vec,
)

# We assume base rate a=0.5 everywhere (your code asserts/uses this a lot).
A0 = 0.5

FuseMethod = Literal["average", "cumulative", "weighted"]
fMethod = "average"  # default fuse method for TensorArrayTO; can be overridden per-instance

# Aliases so internal code can use the short names unchanged.
_normalize = _normalize_vec
normalize_tensor = _normalize_vec


# ------------------------------------------------------------------ #
#  Backend helpers (numpy <-> torch)                                  #
# ------------------------------------------------------------------ #

def _is_torch(v) -> bool:
    return torch is not None and isinstance(v, torch.Tensor)


def default_device():
    """Preferred torch device (CUDA when available), or None without torch."""
    if torch is None:
        return None
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def as_tensor(value, device=None, dtype=None):
    """
    Convert a numpy (...,3) opinion array (or an existing torch tensor) to a
    float32 torch tensor on `device`. Returns the input unchanged when torch
    is not installed, so the numpy code path keeps working.
    """
    if torch is None:
        return value
    if device is None:
        device = default_device()
    if dtype is None:
        dtype = torch.float32
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(np.ascontiguousarray(value), dtype=dtype, device=device)


def to_numpy(value) -> np.ndarray:
    """Bring a torch tensor back to a numpy array (no-op for numpy input)."""
    if _is_torch(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _t102(v):
    """Transpose the first two axes of a (n, m, 3) opinion tensor."""
    if _is_torch(v):
        return v.permute(1, 0, 2)
    return v.transpose(1, 0, 2)


def _clip_min(x, lo: float):
    if _is_torch(x):
        return torch.clamp(x, min=lo)
    return np.clip(x, lo, None)


def _stack_last(parts):
    if _is_torch(parts[0]):
        return torch.stack(parts, -1)
    return np.stack(parts, axis=-1)


def _minimum(a, b):
    return torch.minimum(a, b) if _is_torch(a) else np.minimum(a, b)


def _maximum(a, b):
    return torch.maximum(a, b) if _is_torch(a) else np.maximum(a, b)


def check_valid(v, name="tensor", tol=1e-5):
    """
    Check that a trust opinion tensor is well-formed: no NaN/Inf,
    all components in [0,1], and b+d+u ≈ 1.
    Prints a detailed diagnostic and raises ValueError on first violation.
    """
    v = to_numpy(v)
    if np.any(np.isnan(v)):
        idx = np.argwhere(np.isnan(v))
        raise ValueError(f"[check_valid] NaN in '{name}' at indices {idx[:5]}")
    if np.any(np.isinf(v)):
        idx = np.argwhere(np.isinf(v))
        raise ValueError(f"[check_valid] Inf in '{name}' at indices {idx[:5]}")
    if np.any(v < -tol) or np.any(v > 1.0 + tol):
        lo, hi = v.min(), v.max()
        bad = np.argwhere((v < -tol) | (v > 1.0 + tol))
        raise ValueError(
            f"[check_valid] '{name}' out of [0,1]: min={lo:.6f}, max={hi:.6f}, "
            f"first bad index {bad[0].tolist()} = {v[tuple(bad[0])]:.6f}"
        )
    sums = v.sum(axis=-1)
    if np.any(np.abs(sums - 1.0) > tol):
        bad = np.argwhere(np.abs(sums - 1.0) > tol)
        raise ValueError(
            f"[check_valid] '{name}' b+d+u != 1 at {bad[0].tolist()}: sum={sums[tuple(bad[0])]:.6f}"
        )


def fill(shape, method="vacuous", dtype=np.float32, device=None):
    """
    Build an opinion tensor of shape `shape + (3,)`.
    Returns numpy by default; pass `device` to get a torch tensor there.
    """
    if method == "trust":
        v = np.zeros(shape + (3,), dtype=dtype); v[..., 0] = 1.0
    elif method == "distrust":
        v = np.zeros(shape + (3,), dtype=dtype); v[..., 1] = 1.0
    elif method == "vacuous" or method == "one":
        v = np.zeros(shape + (3,), dtype=dtype); v[..., 2] = 1.0
    elif method == "vacuous2":
        v = np.zeros(shape + (3,), dtype=dtype); v[..., 0] = 0.25; v[..., 1] = 0.25; v[..., 2] = 0.5
    else:
        raise ValueError(f"Unknown method={method}")
    if device is not None:
        return as_tensor(v, device=device)
    return v


def discount(w, x):
    """
    Vectorized probability-sensitive trust discounting.
    w : weight opinion (..., 3)  [b, d, u]
    x : input opinion  (..., 3)
    Delegates to subjective_logic.discount_vec (uses p = b_w + u_w·a).
    """
    return discount_vec(w, x)


def av_fuse_gen(opinions, axis=0):
    """
    Arithmetic mean of opinion masses over `axis` (used for batch aggregation).
    Not the SL ABF formula — this is an element-wise average, same as
    TrustOpinion.avFuseGen.
    """
    m = opinions.mean(axis)
    return _normalize(m)


def av_fuse_pair(op1, op2):
    """
    Vectorized 2-source Averaging Belief Fusion (ABF).
    Delegates to subjective_logic.averaging_fusion_vec.
    Formula: denom = u1+u2,  u = 2·u1·u2/denom.
    """
    return averaging_fusion_vec(op1, op2)


def cum_fuse_pair(op1, op2):
    """
    Vectorized 2-source aleatory Cumulative Belief Fusion (CBF).
    Delegates to subjective_logic.cumulative_fusion_vec.
    Formula: denom = u1+u2−u1·u2,  u = u1·u2/denom.
    """
    return cumulative_fusion_vec(op1, op2)


def weighted_fuse_pair(op1, op2):
    """
    Vectorized 2-source Weighted Belief Fusion (WBF).
    Delegates to subjective_logic.weighted_belief_fusion_vec.
    """
    return weighted_belief_fusion_vec(op1, op2)


def fuse_pair(op1, op2, method: FuseMethod = fMethod):
    """
    Dispatch to av_fuse_pair, cum_fuse_pair, or weighted_fuse_pair.

    Parameters
    ----------
    op1, op2 : ndarray or torch.Tensor (..., 3)
    method   : "average" | "cumulative" | "weighted"
    """
    if method == "average":
        return av_fuse_pair(op1, op2)
    elif method == "cumulative":
        return cum_fuse_pair(op1, op2)
    elif method == "weighted":
        return weighted_fuse_pair(op1, op2)
    else:
        raise ValueError(f"Unknown fuse method: {method!r}. Choose 'average', 'cumulative', or 'weighted'.")


def bin_mult(op1, op2):
    """
    Vectorized binomial multiplication ⊙ (base rates assumed 0.5).
    Delegates to subjective_logic.multiply_vec.
    op1, op2 : (..., 3)
    """
    return multiply_vec(op1, op2)


def fast_deduction(op_x, op_y_given_x, op_y_given_not_x):
    """
    Vectorized approximate inferential deduction.
    - Exact when op_x is fully trust or fully distrust (early-exit cases).
    - Otherwise blends via projected probability, delegating to
      subjective_logic.deduce_vec.
    """
    xt = op_x[..., 0]
    xd = op_x[..., 1]
    if _is_torch(op_x):
        is_trust = torch.round(xt, decimals=3) == 1.0
        is_distr = torch.round(xd, decimals=3) == 1.0
        blended = deduce_vec(op_x, op_y_given_x, op_y_given_not_x)
        return torch.where(is_trust[..., None], op_y_given_x,
               torch.where(is_distr[..., None], op_y_given_not_x, blended))
    is_trust = (np.round(xt, 3) == 1.0)
    is_distr = (np.round(xd, 3) == 1.0)
    blended = deduce_vec(op_x, op_y_given_x, op_y_given_not_x)
    return np.where(is_trust[..., None], op_y_given_x,
           np.where(is_distr[..., None], op_y_given_not_x, blended))


def normalize_tensor(op, eps=1e-12):
    return _normalize_vec(op, eps)


def theta_given_y(delta, epsilon_low, dtype=np.float32):
    # delta: (in+1, out) float — numpy or torch
    cond = abs(delta) < epsilon_low
    W = 2.0
    if _is_torch(delta):
        r = cond.sum(0).to(delta.dtype)
        s = (~cond).sum(0).to(delta.dtype)
    else:
        r = cond.sum(axis=0).astype(dtype)
        s = (~cond).sum(axis=0).astype(dtype)
        W = dtype(2.0)
    b = r / (r + s + W)
    d = s / (r + s + W)
    u = W / (r + s + W)
    out = _stack_last([b, d, u])            # (out,3)
    return normalize_tensor(out)[None, ...] # (1,out,3)


def theta_given_not_y(delta, epsilon_up=None, dtype=np.float32):
    # Your code returns vacuous when epsilon_up is None.
    out_dim = delta.shape[1]
    if _is_torch(delta):
        out = torch.zeros((1, out_dim, 3), dtype=delta.dtype, device=delta.device)
    else:
        out = np.zeros((1, out_dim, 3), dtype=dtype)
    out[..., 2] = 1.0
    return out


def op_theta(weights, opinion_theta_y, method: FuseMethod = fMethod):
    """
    Vectorized version of ArrayTO.op_theta:
      for each column j: fuse(weights[:,j], opinion_theta_y[0,j])
    weights: (n,k,3)
    opinion_theta_y: (1,k,3)
    """
    return fuse_pair(weights, opinion_theta_y, method=method)


def update(weights, prev, lr, method: FuseMethod = fMethod):
    """
    Vectorized version of ArrayTO.update:
      fuse( binMult(prev, lr), weights )
    """
    w_lr = bin_mult(prev, lr)
    return fuse_pair(weights, w_lr, method=method)


def update_2(weights, Tx, Ty):
    """
    Vectorized version of ArrayTO.update_2 (intended behavior).
    weights: (ni1, no, 3)
    Tx:      (ni, 1, 3)  where ni=ni1-1
    Ty:      (3,) scalar — same opinion for all output columns
          OR (no, 3)     — per-column opinion (used when output classes have
                           different trust, e.g. poisoning detection)
    """
    ni1, no, _ = weights.shape
    ni, one, _ = Tx.shape
    assert one == 1 and ni == ni1 - 1, f"Tx shape mismatch: Tx={Tx.shape}, weights={weights.shape}"

    tx_t = Tx[:, 0, 0]          # (ni,)
    tx_d = Tx[:, 0, 1]          # (ni,)

    out = weights.clone() if _is_torch(weights) else weights.copy()

    if Ty.ndim == 1:
        # scalar: same Ty for every output column
        ty_t, ty_d = Ty[0], Ty[1]
        b = _minimum(tx_t, ty_t)
        d = _maximum(tx_d, ty_d)
        u = 1.0 - (b + d)
        myOp = normalize_tensor(_stack_last([b, d, u]))[:, None, :]  # (ni,1,3)
        out[:ni] = bin_mult(out[:ni], myOp)
        out[ni]  = bin_mult(out[ni],  Ty)
    else:
        # per-column: Ty shape (no, 3) — each output column gets its own label opinion
        ty_t = Ty[:, 0]          # (no,)
        ty_d = Ty[:, 1]          # (no,)
        b = _minimum(tx_t[:, None], ty_t[None, :])   # (ni, no)
        d = _maximum(tx_d[:, None], ty_d[None, :])   # (ni, no)
        u = 1.0 - (b + d)
        myOp = normalize_tensor(_stack_last([b, d, u]))  # (ni, no, 3)
        out[:ni] = bin_mult(out[:ni], myOp)
        out[ni]  = bin_mult(out[ni],  Ty)              # (no, 3) broadcast over bias row

    return out


class TensorArrayTO:
    """
    A drop-in style container: value is a float32 tensor (...,3),
    either a numpy ndarray (legacy) or a torch.Tensor (GPU-capable).

    Parameters
    ----------
    value       : ndarray or torch.Tensor (...,3)
    fuse_method : "average" | "cumulative" | "weighted" — controls all
                  internal fusion calls.
    """

    def __init__(self, value, fuse_method: FuseMethod = fMethod):
        if value.ndim < 2 or value.shape[-1] != 3:
            raise ValueError("TensorArrayTO expects shape (...,3)")
        self.value = value
        self.fuse_method = fuse_method

    def _fuse(self, op1, op2):
        """Internal shortcut that uses self.fuse_method."""
        return fuse_pair(op1, op2, method=self.fuse_method)

    def get_shape(self):
        return tuple(self.value.shape[:-1])

    def to(self, device):
        """Return a torch-backed copy on `device` (no-op without torch)."""
        return TensorArrayTO(as_tensor(self.value, device=device), fuse_method=self.fuse_method)

    def to_numpy(self) -> np.ndarray:
        """Return the underlying opinion tensor as a numpy array."""
        return to_numpy(self.value)

    @property
    def T(self):
        # transpose the first 2 dims (matrix), keep last dim=3
        v = self.value
        if v.ndim != 3:
            raise ValueError("T only supported for 2D matrices (n,m,3)")
        return TensorArrayTO(_t102(v), fuse_method=self.fuse_method)

    def fuse_batch(self):
        """
        Match ArrayTO.fuse_batch() behavior:
        input  (batch, dim, 3)
        output (dim, 1, 3)
        """
        v = self.value
        if v.ndim != 3:
            raise ValueError("fuse_batch expects (batch, dim, 3)")
        bsz, dim, _ = v.shape
        if bsz == 1:
            # (1, dim, 3) -> (dim, 1, 3)
            return TensorArrayTO(_t102(v), fuse_method=self.fuse_method)
        fused = av_fuse_gen(v, axis=0)         # (dim, 3)  — gen always uses mean
        return TensorArrayTO(fused[:, None, :], fuse_method=self.fuse_method)

    @staticmethod
    def dot(A: "TensorArrayTO", B: "TensorArrayTO"):
        """
        A: (batch, K, 3)
        B: (K, J, 3)
        Output: (batch, J, 3)

        Equivalent to mean_k discount(B[k,j], A[b,k]) but uses two matmuls
        instead of a (batch,K,J,3) intermediate tensor.

        discount(w, a) = (p*b_a, p*d_a, 1 - p*(b_a+d_a))  where p = b_w + 0.5*u_w
        av_fuse_gen over k = arithmetic mean, which distributes over the linear terms.
        """
        a = A.value   # (batch, K, 3)
        w = B.value   # (K, J, 3)

        # Coerce mixed backends (e.g. numpy trust input against torch omegas)
        if _is_torch(w) and not _is_torch(a):
            a = as_tensor(a, device=w.device)
        elif _is_torch(a) and not _is_torch(w):
            w = as_tensor(w, device=a.device)

        # Projected probability of each weight: p_kj = b_kj + 0.5 * u_kj  →  (K, J)
        p = w[..., 0] + 0.5 * w[..., 2]

        # mean_k(p_kj * a_bk[c]) = (a[...,c] @ p) / K  for c in {belief, disbelief}
        K = a.shape[1]
        b_out = (a[..., 0] @ p) / K   # (batch, J)
        d_out = (a[..., 1] @ p) / K   # (batch, J)
        u_out = _clip_min(1.0 - b_out - d_out, 0.0)
        out = _stack_last([b_out, d_out, u_out])  # (batch, J, 3)
        return TensorArrayTO(_normalize_vec(out), fuse_method=A.fuse_method)

    def update(self, prev: "TensorArrayTO", lr):
        """
        lr is a scalar opinion tensor (3,) or broadcastable to weights shape.
        Equivalent of: weights = fuse( binMult(weights, lr), prev )
        Uses self.fuse_method.
        """
        w_lr = bin_mult(self.value, lr)
        fused = self._fuse(w_lr, prev.value)
        return TensorArrayTO(fused, fuse_method=self.fuse_method)
