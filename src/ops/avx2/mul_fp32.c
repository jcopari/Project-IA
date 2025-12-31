#include "qorus.h"
#include <immintrin.h>
#include <stdio.h>
#include <time.h>

// Element-wise Mul AVX2: output = a * b
// Optimized Architecture:
// - AVX2 Vectorization: Processes 8 elements at a time
// - 4x Loop Unrolling: Processes 32 elements per iteration (maximizes throughput)
// - Supports in-place operation: output may alias a or b
//
// Time Complexity: O(N) - Must visit every element once
// Space Complexity: O(1) - Only AVX2 registers, no temporary buffers
//
// Reference: Similar to q_add_f32_avx2, but uses multiplication instead of addition

q_error_code q_mul_f32_avx2(
    const q_tensor* a,      // NO restrict: output may alias a or b
    const q_tensor* b,      // NO restrict: output may alias a or b
    q_tensor* output        // NO restrict: may alias a or b (in-place operation)
) {
    // STEP 0: Validation (always active)
    Q_VALIDATE_PTR_OR_RETURN(a, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(b, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(output, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(a->data, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(b->data, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(output->data, Q_ERR_INVALID_ARG);
    
    // Validate 1D tensors (explicit check)
    if (!(a->ne[1] == 1 && a->ne[2] == 1 && a->ne[3] == 1)) {
        return Q_ERR_INVALID_SIZE;
    }
    if (!(b->ne[1] == 1 && b->ne[2] == 1 && b->ne[3] == 1)) {
        return Q_ERR_INVALID_SIZE;
    }
    if (!(output->ne[1] == 1 && output->ne[2] == 1 && output->ne[3] == 1)) {
        return Q_ERR_INVALID_SIZE;
    }
    
    // Extract dimensions
    const uint32_t N = a->ne[0];
    
    // Validate shapes match
    if (N != b->ne[0]) {
        return Q_ERR_INVALID_SIZE;
    }
    if (N != output->ne[0]) {
        return Q_ERR_INVALID_SIZE;
    }
    
    // Validate types
    Q_VALIDATE_OR_RETURN(a->type == Q_F32, Q_ERR_INVALID_DTYPE);
    Q_VALIDATE_OR_RETURN(b->type == Q_F32, Q_ERR_INVALID_DTYPE);
    Q_VALIDATE_OR_RETURN(output->type == Q_F32, Q_ERR_INVALID_DTYPE);
    
    // Validate contiguity: nb[0] must equal N * sizeof(float) for 1D contiguous tensors
    if (a->nb[0] != N * sizeof(float)) {
        return Q_ERR_INVALID_SIZE;
    }
    if (b->nb[0] != N * sizeof(float)) {
        return Q_ERR_INVALID_SIZE;
    }
    if (output->nb[0] != N * sizeof(float)) {
        return Q_ERR_INVALID_SIZE;
    }
    
    // Validate alignment: AVX2 requires 32-byte alignment for aligned loads/stores
    // Check that data pointers are 32-byte aligned
    Q_VALIDATE_OR_RETURN(((uintptr_t)a->data % 32) == 0, Q_ERR_MISALIGNED);
    Q_VALIDATE_OR_RETURN(((uintptr_t)b->data % 32) == 0, Q_ERR_MISALIGNED);
    Q_VALIDATE_OR_RETURN(((uintptr_t)output->data % 32) == 0, Q_ERR_MISALIGNED);
    
    // Special case: N=0 (no-op)
    if (N == 0) {
        return Q_OK;
    }
    
    // Get data pointers (NO restrict: output may alias a or b)
    const float* a_data = (const float*)a->data;
    const float* b_data = (const float*)b->data;
    float* out_data = (float*)output->data;
    
    // AVX2 Vectorized Multiplication with 4x unrolling
    // Process 32 elements per iteration (4x8 AVX2 registers)
    uint32_t vec_end = N & ~31U; // Multiple of 32
    
    // Main loop: 4x unrolling to maximize throughput
    for (uint32_t i = 0; i < vec_end; i += 32) {
        __m256 vec_a0, vec_a1, vec_a2, vec_a3;
        __m256 vec_b0, vec_b1, vec_b2, vec_b3;
        __m256 vec_out0, vec_out1, vec_out2, vec_out3;
        
        // Load 4x8 elements from a (aligned loads: data is 32-byte aligned)
        vec_a0 = _mm256_load_ps(&a_data[i + 0]);
        vec_a1 = _mm256_load_ps(&a_data[i + 8]);
        vec_a2 = _mm256_load_ps(&a_data[i + 16]);
        vec_a3 = _mm256_load_ps(&a_data[i + 24]);
        
        // Load 4x8 elements from b (aligned loads: data is 32-byte aligned)
        vec_b0 = _mm256_load_ps(&b_data[i + 0]);
        vec_b1 = _mm256_load_ps(&b_data[i + 8]);
        vec_b2 = _mm256_load_ps(&b_data[i + 16]);
        vec_b3 = _mm256_load_ps(&b_data[i + 24]);
        
        // Vectorized multiply: output = a * b
        vec_out0 = _mm256_mul_ps(vec_a0, vec_b0);
        vec_out1 = _mm256_mul_ps(vec_a1, vec_b1);
        vec_out2 = _mm256_mul_ps(vec_a2, vec_b2);
        vec_out3 = _mm256_mul_ps(vec_a3, vec_b3);
        
        // Store results (may overwrite a or b if in-place)
        // Safe because we read from a and b BEFORE writing to output
        _mm256_store_ps(&out_data[i + 0], vec_out0);
        _mm256_store_ps(&out_data[i + 8], vec_out1);
        _mm256_store_ps(&out_data[i + 16], vec_out2);
        _mm256_store_ps(&out_data[i + 24], vec_out3);
    }
    
    // Tail Loop: Process remaining elements (scalar)
    for (uint32_t i = vec_end; i < N; i++) {
        out_data[i] = a_data[i] * b_data[i];
    }
    
    return Q_OK;
}

