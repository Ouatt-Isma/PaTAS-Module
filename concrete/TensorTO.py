import numpy as np
from typing import Literal

# We assume base rate a=0.5 everywhere (your code asserts/uses this a lot).
A0 = 0.5

FuseMethod = Literal["average", "cumulative", "weighted"]
fMethod = "average"  # default fuse method for TensorArrayTO; can be overridden per-instance

def _normalize(v, eps=1e-12):
    s = v.sum(axis=-1, keepdims=True)
    return v / np.clip(s, eps, None)


def fill(shape, method="vacuous", dtype=np.float32):
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
    return v


def discount(w, x):
    """
    Vectorized version of TrustOpinion.discount where:
    - w is the weight opinion (...,3) with (t,d,u)
    - x is the input opinion (...,3)
    Matches TrustOpinion.discount: p = t + u*a; t'=p*op2.t; d'=p*op2.d; u'=1-(t'+d')
    """
    wt, wd, wu = w[..., 0], w[..., 1], w[..., 2]
    xt, xd = x[..., 0], x[..., 1]
    p = wt + wu * A0
    t = p * xt
    d = p * xd
    u = 1.0 - (t + d)
    out = np.stack([t, d, u], axis=-1)
    return _normalize(out)


def av_fuse_gen(opinions, axis=0):
    """
    Vectorized TrustOpinion.avFuseGen over 'axis'
    """
    m = opinions.mean(axis=axis)
    return _normalize(m)


def av_fuse_pair(op1, op2):
    """
    Vectorized TrustOpinion.avFuse (denom = u1+u2-u1*u2).
    Falls back to arithmetic mean when both uncertainties are 0.
    """
    b1, d1, u1 = op1[..., 0], op1[..., 1], op1[..., 2]
    b2, d2, u2 = op2[..., 0], op2[..., 1], op2[..., 2]

    denom = (u1 + u2 - u1 * u2)
    b = np.where(denom != 0, (b1 * u2 + b2 * u1) / denom, 0.5 * (b1 + b2))
    u = np.where(denom != 0, (u1 * u2) / denom, 0.0)
    d = 1.0 - u - b
    out = np.stack([b, d, u], axis=-1)
    return _normalize(out)


def cum_fuse_pair(op1, op2):
    """
    Vectorized TrustOpinion.cumFuse.

    Same b/u formula as av_fuse_pair, but adds:
      - d clamped to max(0, 1 - b - u)  (mirrors the scalar cumFuse guard)
      - b adjusted to 1 - u when d would have gone negative

    Base rate a is fixed at A0=0.5, so the dynamic-a branch collapses
    to the constant and has no effect on (b, d, u).
    """
    b1, d1, u1 = op1[..., 0], op1[..., 1], op1[..., 2]
    b2, d2, u2 = op2[..., 0], op2[..., 1], op2[..., 2]

    denom = u1 + u2 - u1 * u2
    both_certain = (denom == 0)

    b = np.where(both_certain, 0.5 * (b1 + b2), (b1 * u2 + b2 * u1) / np.where(both_certain, 1.0, denom))
    u = np.where(both_certain, 0.0,              (u1 * u2)            / np.where(both_certain, 1.0, denom))

    # cumFuse clamp: d must be >= 0; if not, cap b at 1-u
    d_raw = 1.0 - b - u
    d = np.maximum(d_raw, 0.0)
    b = np.where(d_raw < 0.0, 1.0 - u, b)   # adjust b when d was clamped

    out = np.stack([b, d, u], axis=-1)
    return _normalize(out)

def weighted_fuse_pair(op1, op2):
    """
    Vectorized TrustOpinion.weighted_belief_fusion.

    Three cases (matching the scalar implementation exactly):
      Case I:  at least one uncertainty != 0, and not both == 1
               b = (b1*(1-u1)*u2 + b2*(1-u2)*u1) / (u1 + u2)
               u = (2-u1-u2)*u1*u2 / (u1 + u2 - 2*u1*u2)
               a = (a1*(1-u1) + a2*(1-u2)) / 2   [collapsed to A0 below]

      Case II: both uncertainties == 0
               b = 0.5*(b1+b2), u = 0

      Case III: both uncertainties == 1
               b = 0, u = 1

    Since A0 is fixed at 0.5, the base-rate output has no effect on (b,d,u),
    so it is not stored. d = 1 - b - u (clamped to 0 like cumFuse).
    """
    b1, d1, u1 = op1[..., 0], op1[..., 1], op1[..., 2]
    b2, d2, u2 = op2[..., 0], op2[..., 1], op2[..., 2]

    both_zero = (u1 == 0) & (u2 == 0)        # Case II
    both_one  = (u1 == 1) & (u2 == 1)        # Case III
    case_i    = ~both_zero & ~both_one        # Case I

    # ---------- Case I numerators / denominators ----------
    # Guard denominator against division by zero outside Case I
    denom_b = np.where(case_i, u1 + u2,               1.0)
    denom_u = np.where(case_i, u1 + u2 - 2*u1*u2,    1.0)

    b_i = (b1*(1-u1)*u2 + b2*(1-u2)*u1) / denom_b
    u_i = (2 - u1 - u2) * u1 * u2 / denom_u

    # ---------- Case II ----------
    b_ii = 0.5 * (b1 + b2)
    u_ii = np.zeros_like(u1)

    # ---------- Case III ----------
    b_iii = np.zeros_like(b1)
    u_iii = np.ones_like(u1)

    # ---------- Select ----------
    b = np.where(both_zero, b_ii,
        np.where(both_one,  b_iii, b_i))
    u = np.where(both_zero, u_ii,
        np.where(both_one,  u_iii, u_i))

    # Clamp d >= 0 (same guard as cumFuse)
    d_raw = 1.0 - b - u
    d = np.maximum(d_raw, 0.0)
    b = np.where(d_raw < 0.0, 1.0 - u, b)

    return _normalize(np.stack([b, d, u], axis=-1))

def fuse_pair(op1, op2, method: FuseMethod = fMethod):
    """
    Dispatch to av_fuse_pair or cum_fuse_pair based on `method`.

    Parameters
    ----------
    op1, op2 : ndarray (..., 3)
    method   : "average" | "cumulative"
    """
    # print(f"Fusing opinions with method={method}")
    if method == "average":
        return av_fuse_pair(op1, op2)
    elif method == "cumulative":
        return cum_fuse_pair(op1, op2)
    elif method == "weighted":
        return weighted_fuse_pair(op1, op2)
    else:
        raise ValueError(f"Unknown fuse method: {method!r}. Choose 'average' or 'cumulative'.")


def bin_mult(op1, op2):
    """
    Vectorized TrustOpinion.binMult (base-rates assumed 0.5)
    op1/op2 (...,3)
    """
    t1, d1, u1 = op1[..., 0], op1[..., 1], op1[..., 2]
    t2, d2, u2 = op2[..., 0], op2[..., 1], op2[..., 2]
    a1 = A0; a2 = A0

    denom = (1.0 - a1 * a2)  # = 0.75
    t = t1 * t2 + (((1-a1)*a2*t1*u2 + (1-a2)*a1*t2*u1) / denom)
    d = d1 + d2 - d1 * d2
    u = u1 * u2 + (((1-a1)*t2*u1 + (1-a2)*t1*u2) / denom)

    out = np.stack([t, d, u], axis=-1)
    return _normalize(out)


def fast_deduction(op_x, op_y_given_x, op_y_given_not_x):
    """
    FAST APPROX DEDUCTION (vectorized):
    - exact when op_x is fully trust or fully distrust (matches your early returns)
    - otherwise blend using projected probability ex = t + a*u

    This replaces the very branch-heavy TrustOpinion.deduction with a tensor-friendly version.
    """
    xt, xd, xu = op_x[..., 0], op_x[..., 1], op_x[..., 2]
    ex = xt + A0 * xu  # projected prob

    # handle extremes exactly like your deduction() early exit
    is_trust = (np.round(xt, 3) == 1.0)
    is_distr = (np.round(xd, 3) == 1.0)

    blended = ex[..., None] * op_y_given_x + (1.0 - ex[..., None]) * op_y_given_not_x
    out = np.where(is_trust[..., None], op_y_given_x,
          np.where(is_distr[..., None], op_y_given_not_x, blended))
    return _normalize(out)


def normalize_tensor(op, eps=1e-12):
    s = op.sum(axis=-1, keepdims=True)
    return op / np.clip(s, eps, None)


def theta_given_y(delta, epsilon_low, dtype=np.float32):
    # delta: (in+1, out) float
    cond = np.abs(delta) < epsilon_low
    r = cond.sum(axis=0).astype(dtype)
    s = (~cond).sum(axis=0).astype(dtype)
    W = dtype(2.0)
    b = r / (r + s + W)
    d = s / (r + s + W)
    u = W / (r + s + W)
    out = np.stack([b, d, u], axis=-1)      # (out,3)
    return normalize_tensor(out)[None, ...] # (1,out,3)


def theta_given_not_y(delta, epsilon_up=None, dtype=np.float32):
    # Your code returns vacuous when epsilon_up is None.
    out_dim = delta.shape[1]
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
    Ty:      (3,) scalar opinion
    """
    ni1, no, _ = weights.shape
    ni, one, _ = Tx.shape
    assert one == 1 and ni == ni1 - 1, f"Tx shape mismatch: Tx={Tx.shape}, weights={weights.shape}"

    ty_t, ty_d = Ty[0], Ty[1]

    tx_t = Tx[:, 0, 0]          # (ni,)
    tx_d = Tx[:, 0, 1]          # (ni,)

    b = np.minimum(tx_t, ty_t)
    d = np.maximum(tx_d, ty_d)
    u = 1.0 - (b + d)
    myOp = np.stack([b, d, u], axis=-1)           # (ni,3)
    myOp = normalize_tensor(myOp)[:, None, :]     # (ni,1,3) broadcastable

    out = weights.copy()
    # first ni rows: binMult with myOp (broadcast to (ni,no,3))
    out[:ni, :, :] = bin_mult(out[:ni, :, :], myOp)
    # last row: binMult with Ty
    out[ni, :, :] = bin_mult(out[ni, :, :], Ty)
    return out


class TensorArrayTO:
    """
    A drop-in style container: value is float32 tensor (...,3).

    Parameters
    ----------
    value       : ndarray (...,3)
    fuse_method : "average" | "cumulative"  — controls all internal fusion calls.
    """

    def __init__(self, value: np.ndarray, fuse_method: FuseMethod = fMethod):
        if value.ndim < 2 or value.shape[-1] != 3:
            raise ValueError("TensorArrayTO expects shape (...,3)")
        self.value = value
        self.fuse_method = fuse_method

    def _fuse(self, op1, op2):
        """Internal shortcut that uses self.fuse_method."""
        return fuse_pair(op1, op2, method=self.fuse_method)

    def get_shape(self):
        return self.value.shape[:-1]

    @property
    def T(self):
        # transpose the first 2 dims (matrix), keep last dim=3
        v = self.value
        if v.ndim != 3:
            raise ValueError("T only supported for 2D matrices (n,m,3)")
        return TensorArrayTO(v.transpose(1, 0, 2), fuse_method=self.fuse_method)

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
            return TensorArrayTO(v.transpose(1, 0, 2), fuse_method=self.fuse_method)
        fused = av_fuse_gen(v, axis=0)         # (dim, 3)  — gen always uses mean
        return TensorArrayTO(fused[:, None, :], fuse_method=self.fuse_method)

    @staticmethod
    def dot(A: "TensorArrayTO", B: "TensorArrayTO"):
        """
        A: (batch, K, 3)
        B: (K, J, 3)
        Output: (batch, J, 3)

        fuse_method is inherited from A.
        """
        a = A.value
        w = B.value

        # broadcast to (batch, K, J, 3)
        a4 = a[:, :, None, :]
        w4 = w[None, :, :, :]

        # discount each weight by corresponding input opinion: (batch,K,J,3)
        dw = discount(w4, a4)

        # fuse across K (mean fusion like your TrustOpinion.add/avFuseGen)
        out = av_fuse_gen(dw, axis=1)
        return TensorArrayTO(out, fuse_method=A.fuse_method)

    def update(self, prev: "TensorArrayTO", lr: np.ndarray):
        """
        lr is a scalar opinion tensor (3,) or broadcastable to weights shape.
        Equivalent of: weights = fuse( binMult(weights, lr), prev )
        Uses self.fuse_method.
        """
        w_lr = bin_mult(self.value, lr)
        fused = self._fuse(w_lr, prev.value)
        return TensorArrayTO(fused, fuse_method=self.fuse_method)