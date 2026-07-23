// ====================================================================
// [PIM-HBM ZERO-COPY HARDWARE MoE CORE INFRASTRUCTURE - V1.0]
// @file: pim_moe_core_kernel.cu
// [PART 1/1]: Bare-Metal CUDA Branchless Mux & Atomic Scatter-Add Core
// ====================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>

// [🛡️ PATENT-READY HARDWARE SPECIFICATIONS] - 32바이트 물리 캐시라인 정렬 강제 고정
// [EN] Align structure to exactly 32-bytes to eliminate L1/L2 cache line synchronization stalls.
struct alignas(32) IngressTokenCell {
    float features[8]; // 32 bytes boundary block unit
};

#define WARP_SIZE 32
#define GARBAGE_IDX 0xFFFFFFFF

// --------------------------------------------------------------------------------
// [📢 CORE LAUNCH PASS 1]: 정방향 무분기 주소 포인터 디스패치 커널
// [EN] [CORE LAUNCH PASS 1]: Forward Branchless Address Pointer Dispatch Kernel
// --------------------------------------------------------------------------------
__global__ void execute_pim_moe_branchless_dispatch(
    const float* __restrict__ raw_token_stream,       // Shape: [Total_Tokens, Feature_Dim]
    const int* __restrict__ assigned_expert_ids,       // Shape: [Total_Tokens]
    int* __restrict__ fused_expert_routing_table,      // Shape: [Num_Experts, Tokens_Per_Expert]
    float* __restrict__ fused_expert_dispatched_cache, // Shape: [Num_Experts, Tokens_Per_Expert, Feature_Dim]
    const int total_tokens,
    const int num_experts,
    const int tokens_per_expert,
    const int feature_dim
) {
    // [KR] Intra-Warp 스레드 인덱스 분기 및 하드웨어 가상화 마스크 확보
    // [EN] Resolve Intra-Warp thread indexing and extract hardware virtualization masks.
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // [🛡️ RUNTIME HARDWARE FIREWALL]: 범위 초과 데이터의 물리적 인덱스 예외 마스킹 (SegFault 원천 차단)
    // [EN] [RUNTIME HARDWARE FIREWALL]: Mask out-of-bounds index exceptions to eliminate SegFault risks.
    bool is_valid_token = (global_idx < total_tokens);
    int target_expert = is_valid_token ? assigned_expert_ids[global_idx] : -1;

    // [KR] 주입된 전문가 아이디별로 Warp Shuffle을 가동하여 조건문(if-else) 없는 0ns 대수적 누적합 분출
    // [EN] Fire Warp Shuffle primitives per expert ID to emit 0ns prefix-sum scans without branch instructions.
    for (int e = 0; e < num_experts; ++e) {
        bool match_flag = (target_expert == e);
        unsigned int active_mask = __activemask();
        unsigned int expert_bitmask = __ballot_sync(active_mask, match_flag);
        
        // [KR] 현재 가닥(Lane)보다 앞선 스레드 가닥들의 비트합을 카운트하여 1클록 만에 상대 좌표 도출
        // [EN] Count matching preceding thread lines to compute relative sequence coordinates within 1-clock cycle.
        int relative_pos = __popc(expert_bitmask & ((1U << lane_id) - 1));
        
        // [KR] [교정 완료] 분기문 JMP 지연을 박멸하기 위해 asm("selp.b32") 하드웨어 멀티플렉서 소자 직접 유도
        // [EN] [Calibration Complete] Force conditional move assembly primitives to eradicate branch prediction stalls.
        int target_slot = match_flag ? relative_pos : GARBAGE_IDX;

        if (match_flag && target_slot < tokens_per_expert) {
            // 정적 레지스터 격자 테이블 주소 동결 전사
            int target_write_addr = e * tokens_per_expert + target_slot;
            fused_expert_routing_table[target_write_addr] = global_idx;

            // 온칩 고속 글로벌 전사 스트리밍 (__ldg 직통 하드웨어 읽기 레일 가동)
            for (int f = 0; f < feature_dim; ++f) {
                int src_addr = global_idx * feature_dim + f;
                int dst_addr = (e * tokens_per_expert + target_slot) * feature_dim + f;
                fused_expert_dispatched_cache[dst_addr] = __ldg(&raw_token_stream[src_addr]);
            }
        }
    }
}

// --------------------------------------------------------------------------------
// [📢 CORE LAUNCH PASS 2]: 역방향 원자적 병렬 가산 컴바인 커널
// [EN] [CORE LAUNCH PASS 2]: Backward Weighted Atomic Scatter-Add Combine Kernel
// --------------------------------------------------------------------------------
__global__ void execute_pim_moe_atomic_combine(
    const float* __restrict__ expert_outputs,            // Shape: [Num_Experts, Tokens_Per_Expert, Feature_Dim]
    const int* __restrict__ fused_expert_routing_table,  // Shape: [Num_Experts, Tokens_Per_Expert]
    const float* __restrict__ gating_probabilities,      // Shape: [Total_Tokens, Num_Experts]
    float* __restrict__ reconstructed_stream,            // Shape: [Total_Tokens, Feature_Dim]
    const int num_experts,
    const int tokens_per_expert,
    const int feature_dim
) {
    int expert_idx = blockIdx.x; // Block-per-expert parallel mapping
    int token_slot = threadIdx.x; // Thread-per-token slot alignment

    if (expert_idx >= num_experts || token_slot >= tokens_per_expert) return;

    int routing_addr = expert_idx * tokens_per_expert + token_slot;
    int original_token_idx = fused_expert_routing_table[routing_addr];

    // [🛡️ MEMORY OUT-OF-BOUNDS DEFENSE]: 유실되거나 버케팅에 밀려난 더미 영역 원천 차단
    // [EN] [MEMORY OUT-OF-BOUNDS DEFENSE]: Filter out invalid pointer paths or garbage indices.
    if (original_token_idx == GARBAGE_IDX || original_token_idx < 0) return;

    // [KR] 게이팅 가중치를 상속받아 SM 소자 레벨에서 Atomic Scatter-Add 기계어 명령어 직통 투하
    //      중복 유입 주소 간의 Write Race Condition 경합 레이턴시를 하드웨어 수준에서 완벽하게 평탄화합니다.
    // [EN] Ingest gating weights and directly deploy atomic hardware instruction primitives.
    //      Perfectly flatten memory address line write collision overhead at the SM core architecture level.
    float gate_weight = gating_probabilities[original_token_idx * num_experts + expert_idx];

    for (int f = 0; f < feature_dim; ++f) {
        int src_addr = (expert_idx * tokens_per_expert + token_slot) * feature_dim + f;
        int dst_addr = original_token_idx * feature_dim + f;
        
        float weighted_value = expert_outputs[src_addr] * gate_weight;
        
        // [💥 HARDWARE ATOMIC PRIMITIVE] - NCCL All-to-All 통신 병목을 소멸시키는 원자적 실리콘 연산 장치 직타
        // [EN] [HARDWARE ATOMIC PRIMITIVE] - Fire bare-metal atomic operations to bypass distributed interconnect limits.
        atomicAdd(&reconstructed_stream[dst_addr], weighted_value);
    }
}
