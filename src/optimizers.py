"""
SOAP optimizer (PyTorch).

SOAP = "ShampoO with Adam in the Preconditioner's eigenbasis"
(Vyas et al. 2024, arXiv:2409.11321).  Quasi-second-order method shown to
resolve directional gradient conflicts in PINN training with 2-10x accuracy
gains over Adam (Wang et al. 2025, arXiv:2502.00604).

Algorithm per 2D parameter W with gradient G:
  1. Maintain Shampoo statistics  L = EMA(G G^T),  R = EMA(G^T G).
  2. Every `precondition_frequency` steps refresh eigenbases Q_L, Q_R (eigh).
  3. Run Adam in the rotated space:
       m   = EMA_b1(G)                       (kept in original space)
       G'  = Q_L^T G Q_R,   m' = Q_L^T m Q_R
       v'  = EMA_b2(G'^2)                    (kept in rotated space)
       N'  = m'_hat / (sqrt(v'_hat) + eps)
       W  -= lr * Q_L N' Q_R^T
Parameters with ndim != 2 (biases, scalars like PirateNet alpha) fall back to
plain Adam — they have no meaningful Kronecker structure.
"""
import torch
from torch.optim.optimizer import Optimizer


class SOAP(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.95, 0.95), shampoo_beta=0.95,
                 eps=1e-8, weight_decay=0.0, precondition_frequency=10):
        defaults = dict(lr=lr, betas=betas, shampoo_beta=shampoo_beta, eps=eps,
                        weight_decay=weight_decay,
                        precondition_frequency=precondition_frequency)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr    = group["lr"]
            b1, b2 = group["betas"]
            sb    = group["shampoo_beta"]
            eps   = group["eps"]
            wd    = group["weight_decay"]
            freq  = group["precondition_frequency"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    if p.ndim == 2:
                        m_dim, n_dim = p.shape
                        state["L"]  = torch.zeros(m_dim, m_dim,
                                                  device=p.device, dtype=p.dtype)
                        state["R"]  = torch.zeros(n_dim, n_dim,
                                                  device=p.device, dtype=p.dtype)
                        state["QL"] = None
                        state["QR"] = None

                state["step"] += 1
                t = state["step"]

                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                if p.ndim != 2:
                    # Plain Adam for vectors / scalars
                    state["m"].mul_(b1).add_(g, alpha=1 - b1)
                    state["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                    m_hat = state["m"] / (1 - b1 ** t)
                    v_hat = state["v"] / (1 - b2 ** t)
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
                    continue

                # ── 2D parameter: SOAP path ─────────────────────────────────
                L, R = state["L"], state["R"]
                L.mul_(sb).add_(g @ g.T, alpha=1 - sb)
                R.mul_(sb).add_(g.T @ g, alpha=1 - sb)

                if state["QL"] is None or (t - 1) % freq == 0:
                    jitter_L = eps * torch.eye(L.shape[0], device=p.device, dtype=p.dtype)
                    jitter_R = eps * torch.eye(R.shape[0], device=p.device, dtype=p.dtype)
                    # eigh in float32+ for stability
                    state["QL"] = torch.linalg.eigh(L + jitter_L).eigenvectors
                    state["QR"] = torch.linalg.eigh(R + jitter_R).eigenvectors
                QL, QR = state["QL"], state["QR"]

                # First moment in original space, projected for the update
                state["m"].mul_(b1).add_(g, alpha=1 - b1)
                g_rot = QL.T @ g @ QR
                m_rot = QL.T @ state["m"] @ QR

                # Second moment in rotated space
                state["v"].mul_(b2).addcmul_(g_rot, g_rot, value=1 - b2)

                m_hat = m_rot / (1 - b1 ** t)
                v_hat = state["v"] / (1 - b2 ** t)
                n_rot = m_hat / (v_hat.sqrt() + eps)

                p.add_(QL @ n_rot @ QR.T, alpha=-lr)

        return loss
