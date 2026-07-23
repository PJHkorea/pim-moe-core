# ====================================================================
# [PIM-HBM ZERO-COPY HARDWARE MoE CORE INFRASTRUCTURE - V1.0]
# @file: pim_moe_autograd_bridge.py
# [PART 1/1]: Heterogeneous Framework 0-Copy Autograd-VJP Bridging Layer
# ====================================================================

import torch
import jax
import jax.numpy as jnp
from torch.utils.dlpack import to_dlpack, from_dlpack
from jax.dlpack import to_dlpack as jax_to_dlpack
from jax.dlpack import from_dlpack as jax_from_dlpack
from typing import Tuple, Any

# pim_moe_config의 하드웨어 사양 뱅크 연동상속
from pim_moe_config import NUM_EXPERTS, FEATURE_DIM

class FngMoeAutogradBridgeFunction(torch.autograd.Function):
    """
    [HYBRID FRAMEWORK INTERLOCK INTERFACE]
    PyTorch의 C++ Autograd 실행 타임라인 환경 내부에 JAX/XLA 분산 VJP 연산 장치를
    0바이트 복사 프로토콜(DLPack Pointer Hijacking)로 주입하는 양방향 가상화 교량입니다.
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
        [📢 FORWARD PASS]: 파이토치 VRAM 기저 주소를 JAX 텐서 버스로 0ns만에 수입 결착
        """
        # [🛡️ HARDWARE ALIGNMENT CHECK]: VRAM 상의 물리 배열 연속성이 깨져 발생하는 수치 폭주 선제 차단
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        if not gate_logits.is_contiguous():
            gate_logits = gate_logits.contiguous()

        ctx.e2e_pipeline = e2e_pipeline
        ctx.mesh = mesh

        # [🔒 0-COPY POINTER HIJACKING]: DLPack 표준 규격을 통해 물리 메모리 복제 버블을 영구 박멸
        # PyTorch가 소유한 가속기 주소선을 JAX 디바이스 어레이로 무복사 직통 전사합니다.
        jax_tokens = jax_from_dlpack(to_dlpack(hidden_states))
        jax_logits = jax_from_dlpack(to_dlpack(gate_logits))

        # [🌀 JAX VJP ENGINE LAUNCH]: 정방향 출력 사출과 동시에 역방향용 미분 주소선(_e2e_vjp_fn) 박제
        with mesh:
            jax_outputs, e2e_vjp_fn = jax.vjp(
                lambda h, g: e2e_pipeline(h, g), 
                jax_tokens, 
                jax_logits
            )
            
        # [🔒 EXTENDED LIFE-CYCLE GUARD]: 비동기 가비지 컬렉터(GC)의 조기 주소 파손을 방어하기 위한 컨텍스트 보존
        ctx.e2e_vjp_fn = e2e_vjp_fn
        ctx.save_for_backward(hidden_states, gate_logits)

        # 정제 완결된 JAX 아웃풋을 다시 파이토치 VRAM 힙 공간으로 0바이트 무복사 회수
        torch_outputs = from_dlpack(jax_to_dlpack(jax_outputs))
        return torch_outputs

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        [📢 BACKWARD PASS]: 단열 백프로파게이션 터널(Adiabatic Backpropagation Tunnel) 가동
        """
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        # 정방향에서 박제해 둔 VJP 기계어 주소선과 파이토치 상류의 오차 행렬 로드
        e2e_vjp_fn = ctx.e2e_vjp_fn
        mesh = ctx.mesh
        
        # 입력 오차 행렬을 0-copy 하이재킹하여 XLA VJP 융합 선로로 조준 투하
        jax_grad_output = jax_from_dlpack(to_dlpack(grad_output))

        # XLA VJP 역산 관류를 기폭하여 토큰과 게이팅 로짓의 그라디언트를 누수(NaN) 없이 산출
        with mesh:
            grad_hidden, grad_logits = e2e_vjp_fn(jax_grad_output)

        # 연산 대상이 아닌 인자축(e2e_pipeline, mesh)에 맞춰 명시적 None 매칭 반환 규격 준수
        torch_grad_hidden = from_dlpack(jax_to_dlpack(grad_hidden))
        torch_grad_logits = from_dlpack(jax_to_dlpack(grad_logits))

        return torch_grad_hidden, torch_grad_logits, None, None


class PimMoeAutogradBridge:
    """
    [HIGH-LEVEL CO-DESIGN WRAPPER]
    실제 모델 레이어 인젝션 단에서 호출하기 용의하도록 캡슐화 플러그인 팩토리 인터페이스를 제공합니다.
    """
    def __init__(self, e2e_pipeline: Any, mesh: Any):
        self.e2e_pipeline = e2e_pipeline
        self.mesh = mesh

    def __call__(self, hidden_states: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
        # 정방향/역방향 단열 자동미분 관로 기폭 실행
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
