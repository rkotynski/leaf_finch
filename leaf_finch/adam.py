from __future__ import annotations

import math
from typing import Any

import torch


class TensorAdam:
    """Adam optimizer specialized for the single dense logits tensor.

    The update is implemented with ordinary eager PyTorch operations and works
    on CPU, CUDA, and ROCm. Keeping this small optimizer local avoids optional
    compiler dependencies and makes checkpoint contents explicit.
    """

    def __init__(
        self,
        parameter: torch.Tensor,
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if not parameter.is_floating_point():
            raise TypeError("TensorAdam requires a real floating-point parameter tensor")
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        beta1, beta2 = (float(betas[0]), float(betas[1]))
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("Adam beta values must be in [0, 1)")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.parameter = parameter
        self.lr = float(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = float(eps)
        self.step_count = 0
        self.exp_avg = torch.zeros_like(parameter, memory_format=torch.preserve_format)
        self.exp_avg_sq = torch.zeros_like(parameter, memory_format=torch.preserve_format)

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        grad = self.parameter.grad
        if grad is None:
            return
        if set_to_none:
            self.parameter.grad = None
        else:
            grad.detach_()
            grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        grad = self.parameter.grad
        if grad is None:
            return
        if grad.is_sparse:
            raise RuntimeError("TensorAdam does not support sparse gradients")
        self.step_count += 1
        self.exp_avg.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
        self.exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

        bias_correction1 = 1.0 - self.beta1**self.step_count
        bias_correction2 = 1.0 - self.beta2**self.step_count
        step_size = self.lr / bias_correction1

        denominator = self.exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(self.eps)
        self.parameter.addcdiv_(self.exp_avg, denominator, value=-step_size)

    def state_dict(self, *, cpu: bool = False) -> dict[str, Any]:
        def tensor(value: torch.Tensor) -> torch.Tensor:
            result = value.detach()
            return result.to("cpu") if cpu else result.clone()

        return {
            "step_count": int(self.step_count),
            "lr": float(self.lr),
            "beta1": float(self.beta1),
            "beta2": float(self.beta2),
            "eps": float(self.eps),
            "exp_avg": tensor(self.exp_avg),
            "exp_avg_sq": tensor(self.exp_avg_sq),
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Any], *, keep_current_lr: bool = True) -> None:
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if not isinstance(exp_avg, torch.Tensor) or not isinstance(exp_avg_sq, torch.Tensor):
            raise ValueError("Invalid TensorAdam state")
        if tuple(exp_avg.shape) != tuple(self.parameter.shape) or tuple(exp_avg_sq.shape) != tuple(self.parameter.shape):
            raise ValueError("Optimizer state shape does not match the loaded model")
        self.step_count = int(state.get("step_count", 0))
        if not keep_current_lr:
            self.lr = float(state.get("lr", self.lr))
        self.beta1 = float(state.get("beta1", self.beta1))
        self.beta2 = float(state.get("beta2", self.beta2))
        self.eps = float(state.get("eps", self.eps))
        self.exp_avg.copy_(exp_avg.to(device=self.parameter.device, dtype=self.parameter.dtype))
        self.exp_avg_sq.copy_(exp_avg_sq.to(device=self.parameter.device, dtype=self.parameter.dtype))
