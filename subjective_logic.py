"""
Canonical implementation of Subjective Logic (SL) operators for patas_module.

This file lives inside patas_module/ and is imported by the rest of the package
(e.g. concrete/TrustOpinion.py).  The root-level subjective_logic.py is a
backward-compatibility shim for scripts (patas.py) that pre-date the integration.

Subjective Logic (SL) operators and binomial opinion data structure.

Implements the operators required by PaTAS-TP (Chapter 7 of the dissertation):
    * Trust discounting          (⊗)
    * Cumulative belief fusion   (⊕ cumulative)    [CBF / aCBF]
    * Averaging belief fusion    (⊕ averaging)     [ABF]  — used for NN sums per §7.8.1
    * Weighted belief fusion     (⊕ weighted)      [WBF]  — Definition 4 of [van der Heijden 2018]
    * Consensus & Compromise     (⊕ ccf)           [CCF]  — Definition 5 of [van der Heijden 2018]
    * Trust revision             (⊖)
    * Binomial multiplication    (⊙)
    * Conservative division      (⊘) — Eq. (7.7)
    * Inferential deduction      (⊚)
    * BPQ / EWQ / CUQ quantification schemes (Eq. 6.3-6.5)

All opinions are *binomial* and stored as a 4-tuple (b, d, u, a) with
b + d + u = 1 and a ∈ [0, 1] being the base rate (default 0.5).

Multi-source fusion (fuse_many / consensus_compromise_fusion / weighted_belief_fusion)
-----------------------------------------------------------------------
All multi-source operators follow the *direct* N-source formulas from:

    A. Jøsang, D. Wang, J. Zhang, "Multi-Source Fusion in Subjective Logic,"
    FUSION 2017.

and the corrections / extensions in:

    R. W. van der Heijden, H. Kopp, F. Kargl,
    "Multi-Source Fusion Operations in Subjective Logic," arXiv 1805.01388, 2018.

Key correctness notes
---------------------
* ABF is **not** associative, so sequential binary application is WRONG for N>2
  sources; this module uses the direct N-source formula (verified against
  Table I of [van der Heijden 2018]).
* CBF *is* effectively associative in the non-dogmatic case, and the direct
  N-source formula is used here for consistency.
* Dogmatic-handling edge cases follow the corrections in §III-A of
  [van der Heijden 2018]: when ≥2 sources are dogmatic (u=0) and ≥1 is
  non-dogmatic, the non-dogmatic opinions are discarded and only the
  dogmatic ones are averaged.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List
import numpy as np

EPS = 1e-12


@dataclass
class Opinion:
    """Binomial subjective opinion ω = (b, d, u, a)."""
    b: float = 0.0
    d: float = 0.0
    u: float = 1.0
    a: float = 0.5

    def __post_init__(self):
        # numerical safety + normalisation
        self.b = max(0.0, min(1.0, float(self.b)))
        self.d = max(0.0, min(1.0, float(self.d)))
        self.u = max(0.0, min(1.0, float(self.u)))
        s = self.b + self.d + self.u
        if s > 0 and abs(s - 1.0) > 1e-6:
            self.b, self.d, self.u = self.b / s, self.d / s, self.u / s

    # projected probability (probability expectation)
    @property
    def P(self) -> float:
        return self.b + self.a * self.u

    def as_tuple(self):
        return (self.b, self.d, self.u, self.a)

    def __repr__(self):
        return f"ω(b={self.b:.3f}, d={self.d:.3f}, u={self.u:.3f}, a={self.a:.2f})"


# ------------------------------------------------------------------ #
#  Canonical opinion shortcuts                                       #
# ------------------------------------------------------------------ #
def trusted(a: float = 0.5)    -> Opinion: return Opinion(1.0, 0.0, 0.0, a)
def distrusted(a: float = 0.5) -> Opinion: return Opinion(0.0, 1.0, 0.0, a)
def vacuous(a: float = 0.5)    -> Opinion: return Opinion(0.0, 0.0, 1.0, a)


# ------------------------------------------------------------------ #
#  Quantification schemes (Chapter 6, Eq. 6.3-6.5)                   #
# ------------------------------------------------------------------ #
def bpq(r: float, s: float, W: float = 2.0, a: float = 0.5) -> Opinion:
    """Baseline-Prior Quantification."""
    denom = W + r + s
    return Opinion(r / denom, s / denom, W / denom, a)


def ewq(r: float, s: float, w: float, a: float = 0.5) -> Opinion:
    """Evidence-Weighted Quantification."""
    denom = w + r + s + EPS
    return Opinion(r / denom, s / denom, w / denom, a)


def cuq(r: float, s: float, U: float, a: float = 0.5) -> Opinion:
    """Constant-Uncertainty Quantification."""
    U = max(EPS, min(1.0, U))
    total = r + s
    if total <= 0:
        return Opinion(0.0, 0.0, 1.0, a)
    gamma = (1.0 - U) / total
    return Opinion(gamma * r, gamma * s, U, a)


# ------------------------------------------------------------------ #
#  Trust discounting  (⊗) -- referral discounting                    #
#  ω_x^{A;B} = ω_A^B ⊗ ω_x^B                                         #
# ------------------------------------------------------------------ #
def discount(omega_A: Opinion, omega_x: Opinion) -> Opinion:
    """
    Standard probability-sensitive trust discounting.
    Trust mass of A scales the (b,d) of x; the rest goes to uncertainty.
    """
    p = omega_A.b  # use trust mass of A as the discount factor
    b = p * omega_x.b
    d = p * omega_x.d
    u = 1.0 - b - d
    return Opinion(b, d, u, omega_x.a)


# ------------------------------------------------------------------ #
#  Binary fusion operators (two-source, exact)                       #
# ------------------------------------------------------------------ #
def cumulative_fusion(w1: Opinion, w2: Opinion) -> Opinion:
    """
    Aleatory cumulative belief fusion (aCBF) for two sources [Jøsang].

    Correct per [van der Heijden 2018] §III-A for the binary case:
    - Both dogmatic (u1=u2=0): average.
    - Otherwise: standard CBF formula.
    """
    u1, u2 = w1.u, w2.u
    if u1 < EPS and u2 < EPS:                      # Case I: both dogmatic
        b = 0.5 * (w1.b + w2.b)
        d = 0.5 * (w1.d + w2.d)
        return Opinion(b, d, max(0.0, 1.0 - b - d), 0.5 * (w1.a + w2.a))
    denom = u1 + u2 - u1 * u2                      # Case II: ≥1 non-dogmatic
    b = (w1.b * u2 + w2.b * u1) / denom
    d = (w1.d * u2 + w2.d * u1) / denom
    u = u1 * u2 / denom
    return Opinion(b, d, u, 0.5 * (w1.a + w2.a))


def averaging_fusion(w1: Opinion, w2: Opinion) -> Opinion:
    """
    Averaging belief fusion (ABF) for two sources [Jøsang].

    Correct formula:
    - Both dogmatic (u1=u2=0): average.
    - Otherwise: b=(b1u2+b2u1)/(u1+u2), u=2u1u2/(u1+u2).

    Note: the denominator for BOTH b and u is (u1+u2), NOT (u1+u2−2u1u2).
    The latter is the WBF denominator and is a common mis-transcription.
    """
    u1, u2 = w1.u, w2.u
    if (u1 + u2) < EPS:                            # Case I: both dogmatic
        b = 0.5 * (w1.b + w2.b)
        d = 0.5 * (w1.d + w2.d)
        return Opinion(b, d, max(0.0, 1.0 - b - d), 0.5 * (w1.a + w2.a))
    b = (w1.b * u2 + w2.b * u1) / (u1 + u2)       # Case II: ≥1 non-dogmatic
    d = (w1.d * u2 + w2.d * u1) / (u1 + u2)
    u = 2.0 * u1 * u2 / (u1 + u2)
    return Opinion(b, d, u, 0.5 * (w1.a + w2.a))


# ------------------------------------------------------------------ #
#  Multi-source fusion helpers                                       #
# ------------------------------------------------------------------ #
def _prod_except(us: List[float], k: int) -> float:
    """∏_{j ≠ k} us[j]  (product of all elements except index k)."""
    result = 1.0
    for j, u in enumerate(us):
        if j != k:
            result *= u
    return result


def _fuse_cbf_multi(opinions: List[Opinion]) -> Opinion:
    """
    Direct N-source aleatory CBF (van der Heijden 2018, §III-A corrected).

    Three cases, matching the corrected multi-source definition:
    - Case I  (all dogmatic, u=0):            average all opinions.
    - Case II (≥1 dogmatic, ≥1 non-dogmatic): discard non-dogmatic, average dogmatic.
    - Case III (all non-dogmatic):            direct N-source formula.

    Formula (Case III, binomial):
        denom = Σ_A [∏_{A'≠A} u_A'] − (N−1)·∏_A u_A
        b     = Σ_A [b_A · ∏_{A'≠A} u_A'] / denom
        d     = Σ_A [d_A · ∏_{A'≠A} u_A'] / denom
        u     = ∏_A u_A / denom
    """
    n = len(opinions)
    us = [w.u for w in opinions]

    # Case I: all dogmatic
    if all(u < EPS for u in us):
        b = sum(w.b for w in opinions) / n
        d = sum(w.d for w in opinions) / n
        a = sum(w.a for w in opinions) / n
        return Opinion(b, d, max(0.0, 1.0 - b - d), a)

    # Case II: mixed dogmatic/non-dogmatic — discard non-dogmatic
    dog_ops = [w for w in opinions if w.u < EPS]
    if dog_ops:
        nd = len(dog_ops)
        b = sum(w.b for w in dog_ops) / nd
        d = sum(w.d for w in dog_ops) / nd
        a = sum(w.a for w in dog_ops) / nd
        return Opinion(b, d, 0.0, a)

    # Case III: all non-dogmatic
    prod_u      = float(np.prod(us))
    prods_excl  = [_prod_except(us, k) for k in range(n)]
    denom       = sum(prods_excl) - (n - 1) * prod_u
    if denom < EPS:
        return vacuous()
    b = sum(opinions[k].b * prods_excl[k] for k in range(n)) / denom
    d = sum(opinions[k].d * prods_excl[k] for k in range(n)) / denom
    u = prod_u / denom
    a = sum(w.a for w in opinions) / n
    return Opinion(b, d, max(0.0, u), a)


def _fuse_abf_multi(opinions: List[Opinion]) -> Opinion:
    """
    Direct N-source ABF (van der Heijden 2018, §III-A corrected).

    ABF is *not* associative, so sequential binary application gives wrong
    results for N > 2.  This uses the correct direct formula.

    Three cases (same structure as CBF):
    - Case I  (all dogmatic):                 average all.
    - Case II (≥1 dogmatic, ≥1 non-dogmatic): discard non-dogmatic, average dogmatic.
    - Case III (all non-dogmatic):            direct N-source formula.

    Formula (Case III, binomial):
        denom = Σ_A [∏_{A'≠A} u_A']
        b     = Σ_A [b_A · ∏_{A'≠A} u_A'] / denom
        d     = Σ_A [d_A · ∏_{A'≠A} u_A'] / denom
        u     = N · ∏_A u_A / denom
        a     = (1/N) Σ_A a_A          (simple average)
    """
    n = len(opinions)
    us = [w.u for w in opinions]

    # Case I: all dogmatic
    if all(u < EPS for u in us):
        b = sum(w.b for w in opinions) / n
        d = sum(w.d for w in opinions) / n
        a = sum(w.a for w in opinions) / n
        return Opinion(b, d, max(0.0, 1.0 - b - d), a)

    # Case II: mixed — discard non-dogmatic
    dog_ops = [w for w in opinions if w.u < EPS]
    if dog_ops:
        nd = len(dog_ops)
        b = sum(w.b for w in dog_ops) / nd
        d = sum(w.d for w in dog_ops) / nd
        a = sum(w.a for w in dog_ops) / nd
        return Opinion(b, d, 0.0, a)

    # Case III: all non-dogmatic
    prods_excl  = [_prod_except(us, k) for k in range(n)]
    denom       = sum(prods_excl)
    if denom < EPS:
        return vacuous()
    b = sum(opinions[k].b * prods_excl[k] for k in range(n)) / denom
    d = sum(opinions[k].d * prods_excl[k] for k in range(n)) / denom
    u = n * float(np.prod(us)) / denom
    a = sum(w.a for w in opinions) / n
    return Opinion(b, d, max(0.0, u), a)


# ------------------------------------------------------------------ #
#  Weighted Belief Fusion  (WBF) — Definition 4                      #
# ------------------------------------------------------------------ #
def weighted_belief_fusion(opinions: Iterable[Opinion]) -> Opinion:
    """
    Direct N-source weighted belief fusion (WBF) — Definition 4 of
    [van der Heijden 2018].

    Three cases:
    - Case 1 (all u_A ≠ 0, ∃A: u_A ≠ 1):  direct N-source formula.
    - Case 2 (∃A: u_A = 0):                combine only dogmatic opinions
                                            with equal weights (γ_A = 1/|A^dog|).
    - Case 3 (∀A: u_A = 1):                fully vacuous result.

    Formula (Case 1, binomial):
        denom = [Σ_A ∏_{A'≠A} u_A'] − N · ∏_A u_A
        b     = Σ_A [b_A·(1−u_A)·∏_{A'≠A} u_A'] / denom
        d     = Σ_A [d_A·(1−u_A)·∏_{A'≠A} u_A'] / denom
        u     = (N − Σ_A u_A) · ∏_A u_A / denom
        a     = Σ_A [a_A·(1−u_A)] / (N − Σ_A u_A)   (confidence-weighted)

    Verified against Table I of [van der Heijden 2018] (3-source example):
        A1=(0.10,0.30,0.60), A2=(0.40,0.20,0.40), A3=(0.70,0.10,0.20)
        → WBF: b≈0.562, d≈0.146, u≈0.292  ✓
    """
    ops = list(opinions)
    n   = len(ops)
    if n == 0:
        return vacuous()
    if n == 1:
        return ops[0]

    us = [w.u for w in ops]

    # Case 3: all vacuous
    if all(abs(u - 1.0) < EPS for u in us):
        return Opinion(0.0, 0.0, 1.0, sum(w.a for w in ops) / n)

    # Case 2: ≥1 dogmatic — combine dogmatic opinions with equal weights
    dog_ops = [w for w in ops if w.u < EPS]
    if dog_ops:
        nd = len(dog_ops)
        b  = sum(w.b for w in dog_ops) / nd
        d  = sum(w.d for w in dog_ops) / nd
        a  = sum(w.a for w in dog_ops) / nd
        return Opinion(b, d, 0.0, a)

    # Case 1: all non-dogmatic, not all vacuous
    prod_u      = float(np.prod(us))
    prods_excl  = [_prod_except(us, k) for k in range(n)]
    sum_prods   = sum(prods_excl)
    denom       = sum_prods - n * prod_u

    if abs(denom) < EPS:
        # Degenerate: all uncertainties identical → fall back to ABF
        return _fuse_abf_multi(ops)

    bw = [(1.0 - us[k]) * prods_excl[k] for k in range(n)]
    b  = sum(ops[k].b * bw[k] for k in range(n)) / denom
    d  = sum(ops[k].d * bw[k] for k in range(n)) / denom
    u  = (n - sum(us)) * prod_u / denom

    # Confidence-weighted base rate
    conf_sum = n - sum(us)   # = Σ(1 − u_A)
    a = sum(ops[k].a * (1.0 - us[k]) for k in range(n)) / max(EPS, conf_sum)

    return Opinion(b, d, max(0.0, u), a)


# ------------------------------------------------------------------ #
#  Consensus & Compromise Fusion  (CCF) — Definition 5              #
# ------------------------------------------------------------------ #
def consensus_compromise_fusion(opinions: Iterable[Opinion]) -> Opinion:
    """
    Direct N-source consensus & compromise fusion (CCF) for *binomial*
    opinions — Definition 5 of [van der Heijden 2018].

    For binomial opinions with domain X = {x, x̄}, the CC-fusion reduces
    to a three-step procedure operating on (b, d, u) directly.

    Step 1 — Consensus:
        b_cons = min_A b_A            (minimum common positive belief)
        d_cons = min_A d_A            (minimum common negative belief)
        b_res_A = b_A − b_cons        (residual positive belief of A)
        d_res_A = d_A − d_cons        (residual negative belief of A)

    Step 2 — Compromise (derived for binomial X = {x, x̄}):
        b_comp(x)  = Σ_A b_res_A·∏_{A'≠A} u_A' + ∏_A b_res_A
        b_comp(x̄) = Σ_A d_res_A·∏_{A'≠A} u_A' + ∏_A d_res_A
        b_comp(X)  = ∏_A(b_res_A+d_res_A) − ∏_A b_res_A − ∏_A d_res_A
                     (composite belief → transferred to uncertainty)
        u_pre      = ∏_A u_A

    Step 3 — Normalization:
        b_comp_total = b_comp(x) + b_comp(x̄) + b_comp(X)
        η  = (1 − b_cons_total − u_pre) / b_comp_total
        b  = b_cons + η·b_comp(x)
        d  = d_cons + η·b_comp(x̄)
        u  = u_pre  + η·b_comp(X)

    Verified against Table I of [van der Heijden 2018] (3-source example):
        A1=(0.10,0.30,0.60), A2=(0.40,0.20,0.40), A3=(0.70,0.10,0.20)
        → CCF: b≈0.629, d≈0.182, u≈0.189  ✓
    """
    ops = list(opinions)
    n   = len(ops)
    if n == 0:
        return vacuous()
    if n == 1:
        return ops[0]

    us   = [w.u for w in ops]
    bs   = [w.b for w in ops]
    ds   = [w.d for w in ops]
    a    = sum(w.a for w in ops) / n   # simple average of base rates

    # ---- Step 1: Consensus ----
    b_cons = min(bs)
    d_cons = min(ds)
    b_cons_total = b_cons + d_cons

    b_res = [bs[k] - b_cons for k in range(n)]
    d_res = [ds[k] - d_cons for k in range(n)]

    # ---- Step 2: Compromise ----
    prods_excl = [_prod_except(us, k) for k in range(n)]

    # b_comp(x): component 1 (one source residual, others uncertain)
    #          + component 2 (all sources provide residual simultaneously, ∩=x)
    # For binomial: component 2 = ∏_A b_res_A  (since a(x|x)=1)
    b_comp_x   = (sum(b_res[k] * prods_excl[k] for k in range(n))
                  + float(np.prod(b_res)))

    b_comp_xbar = (sum(d_res[k] * prods_excl[k] for k in range(n))
                   + float(np.prod(d_res)))

    # b_comp(X): composite belief (all mixed tuples with ≥1 x and ≥1 x̄)
    # = ∏_A(b_res_A+d_res_A) − ∏_A b_res_A − ∏_A d_res_A
    prod_sum_res = float(np.prod([b_res[k] + d_res[k] for k in range(n)]))
    prod_b_res   = float(np.prod(b_res))
    prod_d_res   = float(np.prod(d_res))
    b_comp_X     = max(0.0, prod_sum_res - prod_b_res - prod_d_res)

    u_pre        = float(np.prod(us))
    b_comp_total = b_comp_x + b_comp_xbar + b_comp_X

    # ---- Step 3: Normalization ----
    if b_comp_total < EPS:
        # No residual compromise — consensus opinion dominates
        b_out = b_cons
        d_out = d_cons
        u_out = max(0.0, 1.0 - b_out - d_out)
        return Opinion(b_out, d_out, u_out, a)

    eta   = (1.0 - b_cons_total - u_pre) / b_comp_total
    b_out = b_cons + eta * b_comp_x
    d_out = d_cons + eta * b_comp_xbar
    u_out = u_pre  + eta * b_comp_X

    return Opinion(b_out, d_out, max(0.0, u_out), a)


# ------------------------------------------------------------------ #
#  Unified multi-source fuse_many                                    #
# ------------------------------------------------------------------ #
def fuse_many(opinions: Iterable[Opinion], how: str = "averaging") -> Opinion:
    """
    Fuse an arbitrary number of opinions using the specified operator.

    Parameters
    ----------
    opinions : iterable of Opinion
    how      : one of
        "averaging"  — N-source averaging belief fusion (ABF).
                       NOTE: ABF is not associative, so the direct N-source
                       formula is used (not sequential binary application).
        "cumulative" — N-source cumulative belief fusion (CBF).
        "weighted"   — N-source weighted belief fusion (WBF).
        "ccf"        — N-source consensus & compromise fusion (CCF).

    Returns
    -------
    Opinion
        The fused opinion. Returns vacuous() for an empty iterable.
    """
    ops = list(opinions)
    if not ops:
        return vacuous()
    if len(ops) == 1:
        return ops[0]

    if how == "cumulative":
        return _fuse_cbf_multi(ops)
    elif how == "weighted":
        return weighted_belief_fusion(ops)
    elif how == "ccf":
        return consensus_compromise_fusion(ops)
    else:   # "averaging" (default)
        return _fuse_abf_multi(ops)


# ------------------------------------------------------------------ #
#  Trust revision (⊖)                                                #
#  Used to refine parameter trust: ω' = ω ⊖ evidence                 #
#  Implemented as conservative averaging fusion                      #
# ------------------------------------------------------------------ #
def revise(omega: Opinion, evidence: Opinion) -> Opinion:
    return averaging_fusion(omega, evidence)


# ------------------------------------------------------------------ #
#  Binomial multiplication (⊙) and conservative division (⊘)         #
# ------------------------------------------------------------------ #
def multiply(w1: Opinion, w2: Opinion) -> Opinion:
    """Binomial multiplication (AND) - Jøsang."""
    a1, a2 = w1.a, w2.a
    a = a1 * a2
    b = w1.b * w2.b + (
        ((1.0 - a1) * a2 * w1.b * w2.u + (1.0 - a2) * a1 * w1.u * w2.b)
        / max(EPS, 1.0 - a)
    )
    d = w1.d + w2.d - w1.d * w2.d
    u = max(0.0, 1.0 - b - d)
    return Opinion(b, d, u, a)


def conservative_div(w1: Opinion, w2: Opinion) -> Opinion:
    """Eq. (7.7) - conservative trust 'division' for parameter updates."""
    b = min(w1.b, w2.b)
    d = max(w1.d, w2.d)
    u = max(0.0, 1.0 - (b + d))
    return Opinion(b, d, u, 0.5 * (w1.a + w2.a))


# ------------------------------------------------------------------ #
#  Inferential deduction (⊚)                                         #
#  ω_{n||y} = ω_y ⊚ (ω_{n|y}, ω_{n|¬y})                              #
# ------------------------------------------------------------------ #
def deduce(omega_y: Opinion,
           omega_n_given_y: Opinion,
           omega_n_given_not_y: Opinion) -> Opinion:
    """
    Simplified binomial inferential deduction.
    Returns the trust in n marginalised over y / ¬y.
    """
    Py     = omega_y.P
    notPy  = 1.0 - Py
    b = Py * omega_n_given_y.b + notPy * omega_n_given_not_y.b
    d = Py * omega_n_given_y.d + notPy * omega_n_given_not_y.d
    u = Py * omega_n_given_y.u + notPy * omega_n_given_not_y.u
    s = b + d + u
    if s > 0:
        b, d, u = b / s, d / s, u / s
    return Opinion(b, d, u, omega_y.a)


# ------------------------------------------------------------------ #
#  Convenience: NumPy <-> Opinion grid helpers                       #
# ------------------------------------------------------------------ #
def opinions_to_array(grid) -> np.ndarray:
    """Convert a nested list / array of Opinions to a stacked ndarray
    of shape (..., 4) holding (b, d, u, a)."""
    if isinstance(grid, Opinion):
        return np.array(grid.as_tuple())
    arr = np.array([opinions_to_array(g) for g in grid])
    return arr


# ================================================================== #
#  Vectorized (NumPy) variants                                        #
#                                                                     #
#  Each function operates on opinion arrays of shape (..., 3) where  #
#  the last axis stores [b/trust, d/disbelief, u/uncertainty].        #
#  Base rate  a = 0.5  is assumed throughout (consistent with the     #
#  rest of patas_module).                                             #
#                                                                     #
#  These implement the SAME mathematical formulas as the scalar       #
#  operators above and are used by TensorTO for batch computation.    #
#  Having them here keeps the formulas in one place and avoids drift. #
# ================================================================== #

_EPS_VEC: float = 1e-12


def _normalize_vec(v: np.ndarray, eps: float = _EPS_VEC) -> np.ndarray:
    """Normalise an (..., 3) opinion array so  b + d + u = 1  along axis=-1."""
    s = v.sum(axis=-1, keepdims=True)
    return v / np.clip(s, eps, None)


def averaging_fusion_vec(op1: np.ndarray, op2: np.ndarray,
                         eps: float = _EPS_VEC) -> np.ndarray:
    """
    Vectorized 2-source Averaging Belief Fusion (ABF).
    op1, op2 : (..., 3) arrays  [b, d, u].
    Implements the same formula as averaging_fusion() for scalar Opinion.

    Case II (≥1 non-dogmatic):
        denom = u1 + u2
        b = (b1·u2 + b2·u1) / denom
        d = (d1·u2 + d2·u1) / denom
        u = 2·u1·u2 / denom

    Case I (both dogmatic, u1=u2=0): simple average.
    """
    b1, d1, u1 = op1[..., 0], op1[..., 1], op1[..., 2]
    b2, d2, u2 = op2[..., 0], op2[..., 1], op2[..., 2]
    both_dog = (u1 < eps) & (u2 < eps)
    denom = np.where(both_dog, 1.0, u1 + u2)
    b = np.where(both_dog, 0.5*(b1+b2), (b1*u2 + b2*u1) / denom)
    d = np.where(both_dog, 0.5*(d1+d2), (d1*u2 + d2*u1) / denom)
    u = np.where(both_dog, 0.0,          2.0*u1*u2      / denom)
    return _normalize_vec(np.stack([b, d, u], axis=-1))


def cumulative_fusion_vec(op1: np.ndarray, op2: np.ndarray,
                          eps: float = _EPS_VEC) -> np.ndarray:
    """
    Vectorized 2-source aleatory Cumulative Belief Fusion (aCBF).
    Implements cumulative_fusion() for scalar Opinion.

    Case II (≥1 non-dogmatic):
        denom = u1 + u2 − u1·u2
        b = (b1·u2 + b2·u1) / denom
        d = (d1·u2 + d2·u1) / denom
        u = u1·u2 / denom

    Case I (both dogmatic): simple average.
    """
    b1, d1, u1 = op1[..., 0], op1[..., 1], op1[..., 2]
    b2, d2, u2 = op2[..., 0], op2[..., 1], op2[..., 2]
    both_dog = (u1 < eps) & (u2 < eps)
    raw_denom = u1 + u2 - u1*u2
    denom = np.where(both_dog, 1.0, np.where(np.abs(raw_denom) < eps, eps, raw_denom))
    b = np.where(both_dog, 0.5*(b1+b2), (b1*u2 + b2*u1) / denom)
    d = np.where(both_dog, 0.5*(d1+d2), (d1*u2 + d2*u1) / denom)
    u = np.where(both_dog, 0.0,          u1*u2          / denom)
    return _normalize_vec(np.stack([b, d, u], axis=-1))


def weighted_belief_fusion_vec(op1: np.ndarray, op2: np.ndarray,
                                eps: float = _EPS_VEC) -> np.ndarray:
    """
    Vectorized 2-source Weighted Belief Fusion (WBF).
    Implements weighted_belief_fusion() for scalar Opinion.

    Case 1 (non-dogmatic, not fully vacuous):
        denom = u1 + u2 − 2·u1·u2
        b = (b1·(1−u1)·u2 + b2·(1−u2)·u1) / denom
        d = (d1·(1−u1)·u2 + d2·(1−u2)·u1) / denom
        u = (2 − u1 − u2)·u1·u2 / denom

    Case 2 (≥1 dogmatic): equal-weight average of dogmatic opinions.
    Case 3 (both vacuous): fully vacuous result.
    """
    b1, d1, u1 = op1[..., 0], op1[..., 1], op1[..., 2]
    b2, d2, u2 = op2[..., 0], op2[..., 1], op2[..., 2]
    both_dog = (u1 < eps) & (u2 < eps)                   # Case 2
    both_vac = (u1 > 1.0 - eps) & (u2 > 1.0 - eps)      # Case 3
    case1    = ~both_dog & ~both_vac
    raw_denom = u1 + u2 - 2.0*u1*u2
    denom = np.where(case1, np.where(np.abs(raw_denom) < eps, eps, raw_denom), 1.0)
    bw1 = (1.0 - u1) * u2
    bw2 = (1.0 - u2) * u1
    b = np.where(both_dog, 0.5*(b1+b2),
        np.where(both_vac, 0.0,
                 (b1*bw1 + b2*bw2) / denom))
    d = np.where(both_dog, 0.5*(d1+d2),
        np.where(both_vac, 0.0,
                 (d1*bw1 + d2*bw2) / denom))
    u = np.where(both_dog, 0.0,
        np.where(both_vac, 1.0,
                 (2.0 - u1 - u2) * u1*u2 / denom))
    return _normalize_vec(np.stack([b, d, u], axis=-1))


def ccf_fusion_vec(op1: np.ndarray, op2: np.ndarray,
                   eps: float = _EPS_VEC) -> np.ndarray:
    """
    Vectorized 2-source Consensus & Compromise Fusion (CCF).
    Implements consensus_compromise_fusion() for scalar Opinion.

    Reference: van der Heijden et al., arXiv:1805.01388, 2018, §III-C / Definition 5.
    Binomial case (X = {x, x̄}).

    Step 1 — Consensus:
        b_cons = min(b1, b2),   d_cons = min(d1, d2)
        b_res_k = b_k − b_cons, d_res_k = d_k − d_cons

    Step 2 — Compromise:
        b_comp_x    = b_res1·u2 + b_res2·u1 + b_res1·b_res2
        b_comp_xbar = d_res1·u2 + d_res2·u1 + d_res1·d_res2
        b_comp_X    = max(0, (b_res1+d_res1)·(b_res2+d_res2) − b_res1·b_res2 − d_res1·d_res2)
        u_pre       = u1·u2
        b_comp_total = b_comp_x + b_comp_xbar + b_comp_X + u_pre

    Step 3 — Normalisation (η = residual mass distributed to compromise):
        η = (1 − b_cons − d_cons − u_pre) / b_comp_total   (0 when b_comp_total ≈ 0)
        b_out = b_cons + η·b_comp_x
        d_out = d_cons + η·b_comp_xbar
        u_out = u_pre  + η·b_comp_X
    """
    b1, d1, u1 = op1[..., 0], op1[..., 1], op1[..., 2]
    b2, d2, u2 = op2[..., 0], op2[..., 1], op2[..., 2]

    # Step 1: consensus
    b_cons = np.minimum(b1, b2)
    d_cons = np.minimum(d1, d2)
    b_res1 = b1 - b_cons;  b_res2 = b2 - b_cons
    d_res1 = d1 - d_cons;  d_res2 = d2 - d_cons

    # Step 2: compromise masses
    b_comp_x    = b_res1*u2 + b_res2*u1 + b_res1*b_res2
    b_comp_xbar = d_res1*u2 + d_res2*u1 + d_res1*d_res2
    b_comp_X    = np.maximum(0.0,
                      (b_res1 + d_res1)*(b_res2 + d_res2)
                      - b_res1*b_res2 - d_res1*d_res2)
    u_pre       = u1 * u2
    b_comp_total = b_comp_x + b_comp_xbar + b_comp_X + u_pre

    # Step 3: normalisation factor η
    numerator = 1.0 - b_cons - d_cons - u_pre
    eta = np.where(b_comp_total < eps, 0.0, numerator / np.where(b_comp_total < eps, 1.0, b_comp_total))

    b_out = b_cons + eta * b_comp_x
    d_out = d_cons + eta * b_comp_xbar
    u_out = u_pre  + eta * b_comp_X
    return _normalize_vec(np.stack([b_out, d_out, u_out], axis=-1))


def constraint_fusion_vec(op1: np.ndarray, op2: np.ndarray,
                          eps: float = _EPS_VEC) -> np.ndarray:
    """
    Vectorized 2-source Belief Constraint Fusion (BCF / Dempster's rule).

    Reference: van der Heijden et al., arXiv:1805.01388, 2018, §III-E / Definition 6.
    Equivalent to Dempster-Shafer combination restricted to binomial SL.

    Conflict:
        K = b1·d2 + d1·b2   (mass assigned to the empty set)

    Non-conflicting case (K < 1):
        b_out = (b1·b2 + b1·u2 + u1·b2) / (1 − K)
        d_out = (d1·d2 + d1·u2 + u1·d2) / (1 − K)
        u_out = u1·u2 / (1 − K)

    Total-conflict case (K ≥ 1 − eps): vacuous opinion (u=1).
    """
    b1, d1, u1 = op1[..., 0], op1[..., 1], op1[..., 2]
    b2, d2, u2 = op2[..., 0], op2[..., 1], op2[..., 2]

    K     = b1*d2 + d1*b2
    total_conflict = K > 1.0 - eps
    safe_denom = np.where(total_conflict, 1.0, 1.0 - K)

    b_num = b1*b2 + b1*u2 + u1*b2
    d_num = d1*d2 + d1*u2 + u1*d2
    u_num = u1*u2

    b_out = np.where(total_conflict, 0.0, b_num / safe_denom)
    d_out = np.where(total_conflict, 0.0, d_num / safe_denom)
    u_out = np.where(total_conflict, 1.0, u_num / safe_denom)
    return _normalize_vec(np.stack([b_out, d_out, u_out], axis=-1))


def multiply_vec(op1: np.ndarray, op2: np.ndarray,
                 a: float = 0.5) -> np.ndarray:
    """
    Vectorized binomial multiplication (⊙), base rates assumed equal to a=0.5.
    Implements multiply() for scalar Opinion.

    With a1 = a2 = a:
        b = b1·b2 + [(1−a)·a·b1·u2 + (1−a)·a·b2·u1] / (1 − a²)
        d = d1 + d2 − d1·d2
        u = max(0, 1 − b − d)
    """
    b1, d1, u1 = op1[..., 0], op1[..., 1], op1[..., 2]
    b2, d2, u2 = op2[..., 0], op2[..., 1], op2[..., 2]
    denom = max(EPS, 1.0 - a * a)          # 0.75 when a = 0.5
    b = b1*b2 + ((1.0-a)*a*b1*u2 + (1.0-a)*a*b2*u1) / denom
    d = d1 + d2 - d1*d2
    u = np.clip(1.0 - b - d, 0.0, None)
    return _normalize_vec(np.stack([b, d, u], axis=-1))


def discount_vec(omega_A: np.ndarray, omega_x: np.ndarray,
                 a: float = 0.5) -> np.ndarray:
    """
    Vectorized probability-sensitive trust discounting.
    omega_A, omega_x : (..., 3) arrays  [b, d, u].

    Uses projected probability  P = b_A + a·u_A  as the discount factor.
    This matches TrustOpinion.discount() (projected-probability variant)
    rather than the belief-only subjective_logic.discount() scalar function.

        t = P · b_x
        d = P · d_x
        u = 1 − (t + d)
    """
    wt, wu = omega_A[..., 0], omega_A[..., 2]
    xt, xd = omega_x[..., 0], omega_x[..., 1]
    p = wt + wu * a
    t = p * xt
    d = p * xd
    u = np.clip(1.0 - t - d, 0.0, None)
    return _normalize_vec(np.stack([t, d, u], axis=-1))


def bpq_vec(r: np.ndarray, s: np.ndarray,
            W: float = 2.0, a: float = 0.5) -> np.ndarray:
    """
    Vectorized Baseline-Prior Quantification (BPQ).
    Implements bpq() for scalar Opinion, applied element-wise.

    r, s : arrays of evidence counts (positive / negative)
    Returns : array of shape (*r.shape, 3)  [b, d, u]
    """
    denom = W + r + s
    b = r / denom
    d = s / denom
    u = np.full_like(b, W) / denom
    return np.stack([b, d, u], axis=-1)


def deduce_vec(op_x: np.ndarray,
               op_n_given_y: np.ndarray,
               op_n_given_not_y: np.ndarray,
               a: float = 0.5) -> np.ndarray:
    """
    Vectorized simplified inferential deduction.
    Implements deduce() for scalar Opinion, applied element-wise.
    op_x, op_n_given_y, op_n_given_not_y : (..., 3) arrays.

    Uses projected probability  P(x) = b_x + a·u_x  as the blending weight:
        out = P · op_n_given_y + (1−P) · op_n_given_not_y

    This is the fast/approximate form used in TensorTO (fast_deduction).
    For the exact 8-case Jøsang algorithm see TrustOpinion.deduction().
    """
    xt, xu = op_x[..., 0], op_x[..., 2]
    Py    = (xt + a * xu)[..., None]   # broadcast over last dim
    notPy = 1.0 - Py
    blended = Py * op_n_given_y + notPy * op_n_given_not_y
    return _normalize_vec(blended)
