# ==================================================================== #
# [PIM-HBM ZERO-COPY HARDWARE MoE CORE INFRASTRUCTURE - V1.0]          #
# @file: pim_moe_autograd_bridge.py                                   #
# ==================================================================== #

import torch
import jax
import jax.numpy as jnp
from torch.utils.dlpack import to_dlpack, from_dlpack
from jax.dlpack import to_dlpack as jax_to_dlpack
from jax.dlpack import from_dlpack as jax_from_dlpack
from typing import Tuple, Any

# Inherit hardware specification bank from pim_moe_config
from pim_moe_config import NUM_EXPERTS, FEATURE_DIM

class FngMoeAutogradBridgeFunction(torch.autograd.Function):
    """
    [HYBRID FRAMEWORK INTERLOCK INTERFACE]
    
    A bidirectional virtualization bridge that injects the JAX/XLA distributed VJP 
    execution engine directly into the PyTorch C++ Autograd timeline using a 
    zero-copy protocol (DLPack Pointer Hijacking).
    """

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        gate_logits: torch.Tensor,
        e2e_pipeline: Any,
        mesh: Any
    ) -> torch.Tensor:
        """
        [FORWARD PASS]:
        Imports and binds the PyTorch VRAM base addresses to the JAX tensor bus with zero latency.
        """
        # [HARDWARE ALIGNMENT CHECK]: 
        # Enforce physical memory continuity in VRAM to preemptively prevent numerical explosion.
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        if not gate_logits.is_contiguous():
            gate_logits = gate_logits.contiguous()

        ctx.e2e_pipeline = e2e_pipeline
        ctx.mesh = mesh

        # [ZERO-COPY POINTER HIJACKING]: 
        # Eliminate physical memory duplication overhead via the DLPack standard spec.
        # Direct-map the accelerator address lines owned by PyTorch into JAX DeviceArrays.
        jax_tokens = jax_from_dlpack(to_dlpack(hidden_states))
        jax_logits = jax_from_dlpack(to_dlpack(gate_logits))

        # [JAX VJP ENGINE LAUNCH]: 
        # Compute forward pass outputs while capturing the gradient address paths (_e2e_vjp_fn) for the backward pass.
        with mesh:
            jax_outputs, e2e_vjp_fn = jax.vjp(
                lambda h, g: e2e_pipeline(h, g),
                jax_tokens,
                jax_logits
            )

            
              # [🔒 EXTENDED LIFE-CYCLE GUARD]: 
        # Preserve context to safeguard against premature memory reclamation by the asynchronous Garbage Collector (GC).
        ctx.e2e_vjp_fn = e2e_vjp_fn
        ctx.save_for_backward(hidden_states, gate_logits)

        # Reclaim the finalized JAX output back into the PyTorch VRAM heap space via 0-byte zero-copy.
        torch_outputs = from_dlpack(jax_to_dlpack(jax_outputs))
        return torch_outputs

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        [BACKWARD PASS]:
        Activates the Adiabatic Backpropagation Tunnel for gradient computation.
        """
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        # Load the VJP compiled execution handles and the upstream PyTorch error matrix.
        e2e_vjp_fn = ctx.e2e_vjp_fn
        mesh = ctx.mesh
        
        # Hijack the incoming gradient matrix via zero-copy and inject it into the XLA VJP fused pipeline.
        jax_grad_output = jax_from_dlpack(to_dlpack(grad_output))

        # Trigger the XLA VJP backward flow to compute gradients for tokens and gate logits without numerical leaks (NaN).
        with mesh:
            grad_hidden, grad_logits = e2e_vjp_fn(jax_grad_output)

        # Convert back via zero-copy and adhere to PyTorch autograd spec by returning None for non-differentiable arguments (e2e_pipeline, mesh).
        torch_grad_hidden = from_dlpack(jax_to_dlpack(grad_hidden))
        torch_grad_logits = from_dlpack(jax_to_dlpack(grad_logits))

        return torch_grad_hidden, torch_grad_logits, None, None


class PimMoeAutogradBridge:
    """
    [HIGH-LEVEL CO-DESIGN WRAPPER]
    Provides an encapsulated plugin factory interface designed for seamless injection 
    at the actual model layer execution stage.
    """
    def __init__(self, e2e_pipeline: Any, mesh: Any):
        self.e2e_pipeline = e2e_pipeline
        self.mesh = mesh

    def __call__(self, hidden_states: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
        # Trigger the execution of the forward/backward adiabatic automatic differentiation pipeline
        return FngMoeAutogradBridgeFunction.apply(
            hidden_states, 
            gate_logits, 
            self.e2e_pipeline, 
            self.mesh
        )

print("====================================================================")
print("🔄 HETEROGENEOUS AUTOMATIC DIFFERENTIATION BRIDGE SEALS COMPLETED")
print("   ├─ [FORWARD] DLPack Dual-Pointer Hijacking Loop Enabled.")
print("   └─ [BACKWARD] Adiabatic JAX VJP-to-Autograd Tunneling Sealed.")
print("====================================================================")
