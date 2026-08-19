"""PTASConv — PaTAS trust assessment for convolutional networks (prototype).

Mirrors a ConvNet (see NN/convNN.py) with one opinion matrix per weight layer:

    conv layer   →  omega (c_in·k² + 1, c_out)
    dense layer  →  omega (in + 1, out)

Opinion feedforward uses the *uniform-trust reduction*: the input pixel
opinions are fused into per-channel opinions and each conv layer is applied
as the standard dense discount-fusion rule on the flattened kernel matrix
(inputs tiled k² times, matching the kernel-grad flattening on the NN side).
This is exact when the input trust is spatially uniform (trust / vacuous /
distrust fills) and an approximation otherwise (e.g. patch-distrust inputs).

Additional operator mappings:
    ReLU / pooling      → identity in opinion space
    residual skip add   → averaging-belief fusion of block input and output
    IPTA (path pruning) → activity-weighted: under weight sharing a path
                          selects filter APPLICATIONS rather than weights,
                          so each channel's contribution is weighted by the
                          fraction of spatial positions where it actually
                          fired (exact reduction to MLP row pruning for
                          binary activities; see GenIPTA).

Trust revision reuses the base PTAS machinery unchanged: each conv layer's
gradient arrives flattened to (c_in·k² + 1, c_out), exactly like a dense
layer's delta.
"""

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from NN.PTAStemplate import PTAS
from NN.convNN import spec_omega_shapes
from concrete.TensorTO import (
    TensorArrayTO,
    fill as tfill,
    as_tensor,
    av_fuse_pair,
    normalize_tensor,
    _is_torch,
    _t102,
)


class PTASConv(PTAS):
    def __init__(self, layer_specs, nn_interface, trust_assessment_func,
                 epsilon_low=0.05, epsilon_up=None, learning_rate=None,
                 eval=False, no_round=None, device=None):
        shapes = spec_omega_shapes(layer_specs)
        omegas = [TensorArrayTO(tfill(shape, method="vacuous")) for shape in shapes]
        kwargs = {}
        if learning_rate is not None:
            kwargs["learning_rate"] = learning_rate
        super().__init__(
            omegas,
            operator_mapping=None,
            nn_interface=nn_interface,
            trust_assessment_func=trust_assessment_func,
            structure=layer_specs,
            epsilon_low=epsilon_low,
            epsilon_up=epsilon_up,
            eval=eval,
            no_round=no_round,
            device=device,
            **kwargs,
        )
        self.layer_specs = layer_specs
        self._tx_tiled_history = None

    # ── hooks used by the base class ─────────────────────────────────────────
    def _output_dim(self):
        return self.layer_specs[-1]["out"]

    def _tx_for_layer(self, layer):
        return self._tx_tiled_history[layer]

    def GenIPTA(self, inference_path):
        """Conv IPTA: activity-weighted path-conditioned trust propagation.

        ``inference_path`` is the list of per-omega channel activity vectors
        returned by ConvNet.forward(getactivated=True) for ONE sample: one
        (cout,)-shaped array per weight layer except the classifier head,
        holding the fraction of spatial positions where each channel's
        post-ReLU output was positive.

        Under weight sharing a path cannot select weights, only
        applications of the filter, so the MLP's binary row pruning
        generalizes to weighting each channel's contribution by its
        participation degree (TensorArrayTO.weighted_dot): with binary
        activities this reduces exactly to the MLP GenIPTA semantics, and
        channels the real forward pass never activated contribute nothing.
        """
        acts = [np.asarray(a, dtype=np.float32).reshape(-1) for a in inference_path]
        n_omegas = len(self.omega_thetas)
        if len(acts) != n_omegas - 1:
            raise ValueError(
                f"expected {n_omegas - 1} activity vectors "
                f"(one per weight layer except the classifier head), "
                f"got {len(acts)}")

        def IPTA(Tx):
            return self._feedforward_impl(Tx, activities=acts, tmp=False)

        return IPTA

    # ── opinion feedforward ──────────────────────────────────────────────────
    def apply_feedforward(self, Tx, tmp=True):
        """
        Tx : opinions over the input pixels (batch, d, 3) — any d; they are
             fused per input channel (uniform-trust reduction).
        Returns the output-class opinions (batch, n_classes, 3).
        """
        return self._feedforward_impl(Tx, activities=None, tmp=tmp)

    def _feedforward_impl(self, Tx, activities=None, tmp=True):
        """Shared trust feedforward.  With ``activities`` (one channel
        participation vector per omega except the last), each layer's input
        contributions are weighted by the producing channel's activity —
        the conv IPTA path conditioning; input channels and bias rows keep
        weight 1, mirroring the MLP GenIPTA which keeps all input rows."""
        Tx_t = self._to_tensor_opinions(Tx)
        v = Tx_t.value
        batch = v.shape[0]
        cin0 = self.layer_specs[0]["cin"]

        # pixels → per-channel opinions (exact under spatially uniform trust)
        per_ch = normalize_tensor(v.reshape(batch, cin0, -1, 3).mean(2))  # (batch, cin, 3)
        cur = TensorArrayTO(per_ch)

        one = tfill((batch, 1), method="one", device=self.device)  # bias opinion (vacuous)
        tiled_inputs = []      # per omega: the tiled input opinions (batch, rows-1, 3)
        outputs = [cur]        # per omega: the output opinions       (batch, cols, 3)

        def _cat1(a, b):
            if _is_torch(a):
                return torch.cat([a, b], dim=1)
            return np.concatenate([a, b], axis=1)

        def _tile(val, times):
            if times == 1:
                return val
            if _is_torch(val):
                return torch.repeat_interleave(val, times, dim=1)
            return np.repeat(val, times, axis=1)

        w = 0
        def _apply_omega(cur, tile):
            nonlocal w
            tiled = _tile(cur.value, tile)
            tiled_inputs.append(TensorArrayTO(tiled))
            with_bias = TensorArrayTO(_cat1(tiled, one))
            if activities is not None and w > 0:
                # rows follow the c·k²+u·k+v flattening: repeat the producing
                # channel's activity over its k² (or spatial) rows; bias = 1.
                row_w = np.concatenate([np.repeat(activities[w - 1], tile),
                                        np.ones(1, dtype=np.float32)])
                out = TensorArrayTO.weighted_dot(with_bias, self.omega_thetas[w], row_w)
            else:
                out = TensorArrayTO.dot(with_bias, self.omega_thetas[w])
            w += 1
            outputs.append(out)
            return out

        for sp in self.layer_specs:
            if sp["kind"] == "conv":
                cur = _apply_omega(cur, sp["k"] ** 2)
            elif sp["kind"] == "resconv":
                block_in = cur
                cur = _apply_omega(cur, sp["k"] ** 2)
                cur = _apply_omega(cur, sp["k"] ** 2)
                # skip connection: fuse block input with block output
                fused = TensorArrayTO(av_fuse_pair(block_in.value, cur.value))
                outputs[-1] = fused          # block output seen by the next layer
                cur = fused
            elif sp["kind"] == "dense":
                cur = _apply_omega(cur, sp["spatial"])
            else:
                raise ValueError(f"Unknown spec kind: {sp['kind']}")

        if tmp:
            def _to_hist(t):
                if self.batch_size == 1:
                    return TensorArrayTO(_t102(t.value))
                return t.fuse_batch()
            # history[i+1] = output of omega i (y_batch for its revision);
            # final output stays batch-shaped like the base class.
            hist = [_to_hist(o) for o in outputs[:-1]] + [outputs[-1]]
            self.Typrime_layers_history = hist
            self._tx_tiled_history = [_to_hist(t) for t in tiled_inputs]

        return cur
