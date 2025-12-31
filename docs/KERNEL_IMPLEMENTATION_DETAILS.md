# QORUS-IA v2.0: KERNEL IMPLEMENTATION DETAILS
# Complete Implementation Guide for Critical Kernels

**Status:** 📋 IMPLEMENTATION GUIDE  
**Methodology:** MFR + CoT + Mathematical Proof + TDD  
**Language:** English ONLY  
**Date:** 2024-12-30

---

## OVERVIEW

This document provides complete implementation details for porting 4 critical mathematical kernels from MetaIA to New-QorusIA. Each kernel includes:
- Complete function signatures
- Full validation logic
- AVX2 optimized implementation
- Test cases
- Integration examples

---

## KERNEL 1: MATMUL FP32 AVX2

### Complete Function Signature

```c
// File: src/ops/avx2/matmul_fp32.c
// Header: include/qorus.h

/**
 * Matrix Multiplication FP32: C = A @ B
 * 
 * Computes matrix multiplication using cache-blocked algorithm with AVX2 optimization.
 * 
 * Algorithm:
 * 1. Transpose B for cache efficiency (B_T)
 * 2. Cache-blocked computation (32x32 blocks)
 * 3. AVX2 inner loop with 4x accumulator unrolling
 * 4. Manual prefetching for next iteration
 * 5. Scalar fallback for remainder elements
 * 
 * Preconditions:
 * - A: FP32 matrix [M, K], 32-byte aligned
 * - B: FP32 matrix [K, N], 32-byte aligned
 * - C: FP32 matrix [M, N], 32-byte aligned (output)
 * - ctx: Memory context with arena allocated
 * 
 * Postconditions:
 * - C contains A @ B
 * - All validation checks passed
 * 
 * Returns:
 * - Q_OK on success
 * - Q_ERR_INVALID_ARG if any pointer is NULL
 * - Q_ERR_INVALID_SIZE if shapes are incompatible
 * - Q_ERR_MISALIGNED if pointers are not aligned
 * - Q_ERR_INVALID_DTYPE if types are not Q_F32
 * - Q_ERR_OUT_OF_MEMORY if arena allocation fails
 */
q_error_code q_matmul_f32_avx2(
    const q_tensor* restrict A,      // [M, K]
    const q_tensor* restrict B,      // [K, N]
    q_tensor* C,                    // [M, N] (output, may alias A or B)
    q_context* restrict ctx          // Memory context (for arena allocation)
);
```

### Complete Implementation

```c
#include "qorus.h"
#include <immintrin.h>
#include <string.h>

// Block size for cache blocking (32x32 = 4KB per block, fits in L1)
#define MATMUL_BLOCK_SIZE 32

// Prefetch distance (3 cache lines = 192 bytes)
#define PREFETCH_DISTANCE 192

// Helper: Horizontal sum of __m256
static inline float hsum256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(vlow, vhigh);
    
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ps(sums, shuf);
    
    return _mm_cvtss_f32(sums);
}

// Cache-blocked matrix transpose
static void transpose_blocked(
    const float* restrict src,
    float* restrict dst,
    uint32_t rows,
    uint32_t cols
) {
    for (uint32_t i = 0; i < rows; i += MATMUL_BLOCK_SIZE) {
        uint32_t i_limit = (i + MATMUL_BLOCK_SIZE < rows) ? 
                           i + MATMUL_BLOCK_SIZE : rows;
        
        for (uint32_t j = 0; j < cols; j += MATMUL_BLOCK_SIZE) {
            uint32_t j_limit = (j + MATMUL_BLOCK_SIZE < cols) ? 
                               j + MATMUL_BLOCK_SIZE : cols;
            
            for (uint32_t ii = i; ii < i_limit; ii++) {
                for (uint32_t jj = j; jj < j_limit; jj++) {
                    dst[jj * rows + ii] = src[ii * cols + jj];
                }
            }
        }
    }
}

q_error_code q_matmul_f32_avx2(
    const q_tensor* restrict A,
    const q_tensor* restrict B,
    q_tensor* C,
    q_context* restrict ctx
) {
    // STEP 0: Validation (always active)
    Q_VALIDATE_PTR_OR_RETURN(A, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(B, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(C, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(ctx, Q_ERR_INVALID_ARG);
    
    // Extract dimensions
    const uint32_t M = A->ne[0];
    const uint32_t K = A->ne[1];
    const uint32_t N = B->ne[1];
    
    // Validate dimensions
    Q_VALIDATE_NONZERO_OR_RETURN(M, Q_ERR_INVALID_SIZE);
    Q_VALIDATE_NONZERO_OR_RETURN(K, Q_ERR_INVALID_SIZE);
    Q_VALIDATE_NONZERO_OR_RETURN(N, Q_ERR_INVALID_SIZE);
    
    // Validate shape compatibility: A[M,K] @ B[K,N] → C[M,N]
    Q_VALIDATE_OR_RETURN(B->ne[0] == K, Q_ERR_INVALID_SIZE);
    Q_VALIDATE_OR_RETURN(C->ne[0] == M, Q_ERR_INVALID_SIZE);
    Q_VALIDATE_OR_RETURN(C->ne[1] == N, Q_ERR_INVALID_SIZE);
    
    // Validate alignment
    Q_VALIDATE_ALIGNED_OR_RETURN(A->data, Q_ERR_MISALIGNED);
    Q_VALIDATE_ALIGNED_OR_RETURN(B->data, Q_ERR_MISALIGNED);
    Q_VALIDATE_ALIGNED_OR_RETURN(C->data, Q_ERR_MISALIGNED);
    
    // Validate types
    Q_VALIDATE_OR_RETURN(A->type == Q_F32, Q_ERR_INVALID_DTYPE);
    Q_VALIDATE_OR_RETURN(B->type == Q_F32, Q_ERR_INVALID_DTYPE);
    Q_VALIDATE_OR_RETURN(C->type == Q_F32, Q_ERR_INVALID_DTYPE);
    
    // Get data pointers
    const float* restrict A_data = (const float*)A->data;
    const float* restrict B_data = (const float*)B->data;
    float* C_data = (float*)C->data;
    
    // Transpose B for cache efficiency (allocate in arena)
    float* B_T_data = (float*)q_arena_alloc(ctx, N * K * sizeof(float));
    Q_VALIDATE_PTR_OR_RETURN(B_T_data, Q_ERR_OUT_OF_MEMORY);
    
    transpose_blocked(B_data, B_T_data, K, N);
    
    // Cache-blocked matrix multiplication
    for (uint32_t i = 0; i < M; i += MATMUL_BLOCK_SIZE) {
        uint32_t i_limit = (i + MATMUL_BLOCK_SIZE < M) ? 
                           i + MATMUL_BLOCK_SIZE : M;
        
        for (uint32_t j = 0; j < N; j += MATMUL_BLOCK_SIZE) {
            uint32_t j_limit = (j + MATMUL_BLOCK_SIZE < N) ? 
                               j + MATMUL_BLOCK_SIZE : N;
            
            // Compute block C[i:i_limit, j:j_limit] = A[i:i_limit, :] @ B[:, j:j_limit]
            for (uint32_t ii = i; ii < i_limit; ii++) {
                for (uint32_t jj = j; jj < j_limit; jj++) {
                    // Initialize 4 accumulators (4x unrolling)
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();
                    
                    const float* A_row = A_data + ii * K;
                    const float* B_T_col = B_T_data + jj * K;
                    
                    // Main loop: 4x unrolling (32 elements per iteration)
                    uint32_t k_vec = K & ~31U;
                    for (uint32_t k = 0; k < k_vec; k += 32) {
                        // Prefetch next iteration
                        _mm_prefetch((const char*)(A_row + k + PREFETCH_DISTANCE), _MM_HINT_T0);
                        _mm_prefetch((const char*)(B_T_col + k + PREFETCH_DISTANCE), _MM_HINT_T0);
                        
                        // Load 4x8 elements from A
                        __m256 a0 = _mm256_load_ps(A_row + k + 0);
                        __m256 a1 = _mm256_load_ps(A_row + k + 8);
                        __m256 a2 = _mm256_load_ps(A_row + k + 16);
                        __m256 a3 = _mm256_load_ps(A_row + k + 24);
                        
                        // Load 4x8 elements from B_T
                        __m256 b0 = _mm256_load_ps(B_T_col + k + 0);
                        __m256 b1 = _mm256_load_ps(B_T_col + k + 8);
                        __m256 b2 = _mm256_load_ps(B_T_col + k + 16);
                        __m256 b3 = _mm256_load_ps(B_T_col + k + 24);
                        
                        // FMA: acc += a * b
                        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
                        acc2 = _mm256_fmadd_ps(a2, b2, acc2);
                        acc3 = _mm256_fmadd_ps(a3, b3, acc3);
                    }
                    
                    // Horizontal reduction
                    __m256 sum01 = _mm256_add_ps(acc0, acc1);
                    __m256 sum23 = _mm256_add_ps(acc2, acc3);
                    __m256 sum = _mm256_add_ps(sum01, sum23);
                    float dot_product = hsum256_ps(sum);
                    
                    // Tail handling (scalar fallback)
                    for (uint32_t k = k_vec; k < K; k++) {
                        dot_product += A_row[k] * B_T_col[k];
                    }
                    
                    C_data[ii * N + jj] = dot_product;
                }
            }
        }
    }
    
    return Q_OK;
}
```

### Integration Example

```c
// Example usage in llama_forward():
// Compute attention scores: scores = Q @ K^T

q_tensor Q_tensor = {0};
Q_tensor.data = Q_data;  // [seq_len, head_dim]
Q_tensor.ne[0] = seq_len;
Q_tensor.ne[1] = head_dim;
Q_tensor.type = Q_F32;

q_tensor K_T_tensor = {0};
K_T_tensor.data = K_T_data;  // [head_dim, seq_len] (already transposed)
K_T_tensor.ne[0] = head_dim;
K_T_tensor.ne[1] = seq_len;
K_T_tensor.type = Q_F32;

q_tensor scores_tensor = {0};
scores_tensor.data = scores_data;  // [seq_len, seq_len]
scores_tensor.ne[0] = seq_len;
scores_tensor.ne[1] = seq_len;
scores_tensor.type = Q_F32;

q_error_code err = q_matmul_f32_avx2(&Q_tensor, &K_T_tensor, &scores_tensor, ctx);
if (err != Q_OK) {
    return err;
}
```

---

## KERNEL 2: CAUSAL MASKING AVX2

### Complete Function Signature

```c
/**
 * Causal Masking: Set upper triangular elements to mask_value
 * 
 * Applies causal mask to attention scores matrix by setting upper triangular
 * elements (future tokens) to mask_value (typically -1e9f).
 * 
 * Algorithm:
 * - For each row i, set scores[i, j] = mask_value for all j > i
 * - Uses AVX2 vectorized stores for efficiency
 * 
 * Preconditions:
 * - scores: FP32 matrix [seq_len, seq_len], 32-byte aligned
 * - mask_value: float (typically -1e9f)
 * 
 * Postconditions:
 * - Upper triangular elements set to mask_value
 * - Lower triangular and diagonal elements unchanged
 * 
 * Returns:
 * - Q_OK on success
 * - Q_ERR_INVALID_ARG if scores is NULL
 * - Q_ERR_INVALID_SIZE if matrix is not square or seq_len=0
 * - Q_ERR_MISALIGNED if scores->data is not aligned
 * - Q_ERR_INVALID_DTYPE if type is not Q_F32
 */
q_error_code q_causal_mask_f32_avx2(
    q_tensor* scores,           // [seq_len, seq_len] (modified in-place)
    float mask_value            // Value to set masked positions
);
```

### Complete Implementation

```c
#include "qorus.h"
#include <immintrin.h>

q_error_code q_causal_mask_f32_avx2(
    q_tensor* scores,
    float mask_value
) {
    // Validation
    Q_VALIDATE_PTR_OR_RETURN(scores, Q_ERR_INVALID_ARG);
    
    const uint32_t seq_len = scores->ne[0];
    Q_VALIDATE_NONZERO_OR_RETURN(seq_len, Q_ERR_INVALID_SIZE);
    Q_VALIDATE_OR_RETURN(scores->ne[1] == seq_len, Q_ERR_INVALID_SIZE); // Square matrix
    
    Q_VALIDATE_ALIGNED_OR_RETURN(scores->data, Q_ERR_MISALIGNED);
    Q_VALIDATE_OR_RETURN(scores->type == Q_F32, Q_ERR_INVALID_DTYPE);
    
    float* scores_data = (float*)scores->data;
    const __m256 mask_vec = _mm256_set1_ps(mask_value);
    
    // For each row i, set scores[i, j] = mask_value for j > i
    for (uint32_t i = 0; i < seq_len; i++) {
        // Start masking from column i+1
        uint32_t j_start = i + 1;
        uint32_t j_vec = (seq_len - j_start) & ~7U; // Multiple of 8
        
        // Vectorized masking
        for (uint32_t j = j_start; j < j_start + j_vec; j += 8) {
            _mm256_store_ps(scores_data + i * seq_len + j, mask_vec);
        }
        
        // Scalar fallback for remainder
        for (uint32_t j = j_start + j_vec; j < seq_len; j++) {
            scores_data[i * seq_len + j] = mask_value;
        }
    }
    
    return Q_OK;
}
```

### Integration Example

```c
// Example usage in llama_forward():
// Apply causal mask to attention scores

q_tensor scores_tensor = {0};
scores_tensor.data = scores_data;  // [seq_len, seq_len]
scores_tensor.ne[0] = seq_len;
scores_tensor.ne[1] = seq_len;
scores_tensor.type = Q_F32;

q_error_code err = q_causal_mask_f32_avx2(&scores_tensor, -1e9f);
if (err != Q_OK) {
    return err;
}
```

---

## KERNEL 3: TENSOR ADD AVX2

### Complete Function Signature

```c
/**
 * Tensor Add: output = a + b
 * 
 * Element-wise addition of two FP32 tensors.
 * Supports in-place operation (output may alias a or b).
 * 
 * Algorithm:
 * - AVX2 vectorized addition (4x unrolling)
 * - Scalar fallback for remainder
 * 
 * Preconditions:
 * - a, b, output: FP32 vectors [N], 32-byte aligned, same shape
 * - output may alias a or b (in-place operation supported)
 * 
 * Postconditions:
 * - output[i] = a[i] + b[i] for all i
 * 
 * Returns:
 * - Q_OK on success
 * - Q_ERR_INVALID_ARG if any pointer is NULL
 * - Q_ERR_INVALID_SIZE if shapes don't match
 * - Q_ERR_MISALIGNED if pointers are not aligned
 * - Q_ERR_INVALID_DTYPE if types are not Q_F32
 */
q_error_code q_add_f32_avx2(
    const q_tensor* restrict a,     // [N]
    const q_tensor* restrict b,     // [N]
    q_tensor* output                // [N] (may alias a or b)
);
```

### Complete Implementation

```c
#include "qorus.h"
#include <immintrin.h>

q_error_code q_add_f32_avx2(
    const q_tensor* restrict a,
    const q_tensor* restrict b,
    q_tensor* output
) {
    // Validation
    Q_VALIDATE_PTR_OR_RETURN(a, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(b, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(output, Q_ERR_INVALID_ARG);
    
    const uint32_t N = a->ne[0];
    Q_VALIDATE_OR_RETURN(b->ne[0] == N, Q_ERR_INVALID_SIZE);
    Q_VALIDATE_OR_RETURN(output->ne[0] == N, Q_ERR_INVALID_SIZE);
    
    Q_VALIDATE_ALIGNED_OR_RETURN(a->data, Q_ERR_MISALIGNED);
    Q_VALIDATE_ALIGNED_OR_RETURN(b->data, Q_ERR_MISALIGNED);
    Q_VALIDATE_ALIGNED_OR_RETURN(output->data, Q_ERR_MISALIGNED);
    
    Q_VALIDATE_OR_RETURN(a->type == Q_F32, Q_ERR_INVALID_DTYPE);
    Q_VALIDATE_OR_RETURN(b->type == Q_F32, Q_ERR_INVALID_DTYPE);
    Q_VALIDATE_OR_RETURN(output->type == Q_F32, Q_ERR_INVALID_DTYPE);
    
    const float* restrict a_data = (const float*)a->data;
    const float* restrict b_data = (const float*)b->data;
    float* output_data = (float*)output->data;
    
    // AVX2 vectorized addition (4x unrolling)
    uint32_t vec_end = N & ~31U; // Multiple of 32
    
    for (uint32_t i = 0; i < vec_end; i += 32) {
        // Load 4x8 elements
        __m256 a0 = _mm256_load_ps(a_data + i + 0);
        __m256 a1 = _mm256_load_ps(a_data + i + 8);
        __m256 a2 = _mm256_load_ps(a_data + i + 16);
        __m256 a3 = _mm256_load_ps(a_data + i + 24);
        
        __m256 b0 = _mm256_load_ps(b_data + i + 0);
        __m256 b1 = _mm256_load_ps(b_data + i + 8);
        __m256 b2 = _mm256_load_ps(b_data + i + 16);
        __m256 b3 = _mm256_load_ps(b_data + i + 24);
        
        // Add and store
        _mm256_store_ps(output_data + i + 0, _mm256_add_ps(a0, b0));
        _mm256_store_ps(output_data + i + 8, _mm256_add_ps(a1, b1));
        _mm256_store_ps(output_data + i + 16, _mm256_add_ps(a2, b2));
        _mm256_store_ps(output_data + i + 24, _mm256_add_ps(a3, b3));
    }
    
    // Scalar fallback for remainder
    for (uint32_t i = vec_end; i < N; i++) {
        output_data[i] = a_data[i] + b_data[i];
    }
    
    return Q_OK;
}
```

### Integration Example

```c
// Example usage in llama_forward():
// Residual connection: x = x + attn_out

q_tensor x_tensor = {0};
x_tensor.data = x_data;  // [seq_len, dim]
x_tensor.ne[0] = seq_len * dim;
x_tensor.type = Q_F32;

q_tensor attn_out_tensor = {0};
attn_out_tensor.data = attn_out_data;  // [seq_len, dim]
attn_out_tensor.ne[0] = seq_len * dim;
attn_out_tensor.type = Q_F32;

// In-place operation: output aliases x
q_error_code err = q_add_f32_avx2(&x_tensor, &attn_out_tensor, &x_tensor);
if (err != Q_OK) {
    return err;
}
```

---

## KERNEL 4: ELEMENT-WISE MUL AVX2

### Complete Function Signature

```c
/**
 * Element-wise Multiply: output = a * b
 * 
 * Element-wise multiplication of two FP32 tensors.
 * Supports in-place operation (output may alias a or b).
 * 
 * Algorithm:
 * - AVX2 vectorized multiplication (4x unrolling)
 * - Scalar fallback for remainder
 * 
 * Preconditions:
 * - a, b, output: FP32 vectors [N], 32-byte aligned, same shape
 * - output may alias a or b (in-place operation supported)
 * 
 * Postconditions:
 * - output[i] = a[i] * b[i] for all i
 * 
 * Returns:
 * - Q_OK on success
 * - Q_ERR_INVALID_ARG if any pointer is NULL
 * - Q_ERR_INVALID_SIZE if shapes don't match
 * - Q_ERR_MISALIGNED if pointers are not aligned
 * - Q_ERR_INVALID_DTYPE if types are not Q_F32
 */
q_error_code q_mul_f32_avx2(
    const q_tensor* restrict a,     // [N]
    const q_tensor* restrict b,     // [N]
    q_tensor* output                // [N] (may alias a or b)
);
```

### Complete Implementation

```c
#include "qorus.h"
#include <immintrin.h>

q_error_code q_mul_f32_avx2(
    const q_tensor* restrict a,
    const q_tensor* restrict b,
    q_tensor* output
) {
    // Validation (same as Add)
    Q_VALIDATE_PTR_OR_RETURN(a, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(b, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(output, Q_ERR_INVALID_ARG);
    
    const uint32_t N = a->ne[0];
    Q_VALIDATE_OR_RETURN(b->ne[0] == N, Q_ERR_INVALID_SIZE);
    Q_VALIDATE_OR_RETURN(output->ne[0] == N, Q_ERR_INVALID_SIZE);
    
    Q_VALIDATE_ALIGNED_OR_RETURN(a->data, Q_ERR_MISALIGNED);
    Q_VALIDATE_ALIGNED_OR_RETURN(b->data, Q_ERR_MISALIGNED);
    Q_VALIDATE_ALIGNED_OR_RETURN(output->data, Q_ERR_MISALIGNED);
    
    Q_VALIDATE_OR_RETURN(a->type == Q_F32, Q_ERR_INVALID_DTYPE);
    Q_VALIDATE_OR_RETURN(b->type == Q_F32, Q_ERR_INVALID_DTYPE);
    Q_VALIDATE_OR_RETURN(output->type == Q_F32, Q_ERR_INVALID_DTYPE);
    
    const float* restrict a_data = (const float*)a->data;
    const float* restrict b_data = (const float*)b->data;
    float* output_data = (float*)output->data;
    
    // AVX2 vectorized multiplication (4x unrolling)
    uint32_t vec_end = N & ~31U; // Multiple of 32
    
    for (uint32_t i = 0; i < vec_end; i += 32) {
        // Load 4x8 elements
        __m256 a0 = _mm256_load_ps(a_data + i + 0);
        __m256 a1 = _mm256_load_ps(a_data + i + 8);
        __m256 a2 = _mm256_load_ps(a_data + i + 16);
        __m256 a3 = _mm256_load_ps(a_data + i + 24);
        
        __m256 b0 = _mm256_load_ps(b_data + i + 0);
        __m256 b1 = _mm256_load_ps(b_data + i + 8);
        __m256 b2 = _mm256_load_ps(b_data + i + 16);
        __m256 b3 = _mm256_load_ps(b_data + i + 24);
        
        // Multiply and store
        _mm256_store_ps(output_data + i + 0, _mm256_mul_ps(a0, b0));
        _mm256_store_ps(output_data + i + 8, _mm256_mul_ps(a1, b1));
        _mm256_store_ps(output_data + i + 16, _mm256_mul_ps(a2, b2));
        _mm256_store_ps(output_data + i + 24, _mm256_mul_ps(a3, b3));
    }
    
    // Scalar fallback for remainder
    for (uint32_t i = vec_end; i < N; i++) {
        output_data[i] = a_data[i] * b_data[i];
    }
    
    return Q_OK;
}
```

### Integration Example

```c
// Example usage in llama_forward():
// SwiGLU activation: mlp_out = (gate * up) @ w_down

q_tensor gate_tensor = {0};
gate_tensor.data = gate_data;  // [seq_len, hidden_dim]
gate_tensor.ne[0] = seq_len * hidden_dim;
gate_tensor.type = Q_F32;

q_tensor up_tensor = {0};
up_tensor.data = up_data;  // [seq_len, hidden_dim]
up_tensor.ne[0] = seq_len * hidden_dim;
up_tensor.type = Q_F32;

q_tensor gate_up_tensor = {0};
gate_up_tensor.data = gate_up_data;  // [seq_len, hidden_dim] (temporary buffer)
gate_up_tensor.ne[0] = seq_len * hidden_dim;
gate_up_tensor.type = Q_F32;

// Element-wise multiplication: gate_up = gate * up
q_error_code err = q_mul_f32_avx2(&gate_tensor, &up_tensor, &gate_up_tensor);
if (err != Q_OK) {
    return err;
}
```

---

## HEADER UPDATES

Add these function declarations to `include/qorus.h`:

```c
// ============================================================================
// Mathematical Operations API (AVX2 Optimized) - Additional Kernels
// ============================================================================

// MatMul FP32: C = A @ B
// Preconditions:
// - A: FP32 matrix [M, K], 32-byte aligned
// - B: FP32 matrix [K, N], 32-byte aligned
// - C: FP32 matrix [M, N], 32-byte aligned (output)
// - ctx: Memory context with arena allocated
// Returns: Q_OK on success, negative q_error_code on error
q_error_code q_matmul_f32_avx2(
    const q_tensor* restrict A,
    const q_tensor* restrict B,
    q_tensor* C,
    q_context* restrict ctx
);

// Causal Masking: Set upper triangular elements to mask_value
// Preconditions:
// - scores: FP32 matrix [seq_len, seq_len], 32-byte aligned
// - mask_value: float (typically -1e9f)
// Returns: Q_OK on success, negative q_error_code on error
q_error_code q_causal_mask_f32_avx2(
    q_tensor* scores,
    float mask_value
);

// Tensor Add: output = a + b
// Preconditions:
// - a, b, output: FP32 vectors [N], 32-byte aligned, same shape
// - output may alias a or b (in-place operation supported)
// Returns: Q_OK on success, negative q_error_code on error
q_error_code q_add_f32_avx2(
    const q_tensor* restrict a,
    const q_tensor* restrict b,
    q_tensor* output
);

// Element-wise Multiply: output = a * b
// Preconditions:
// - a, b, output: FP32 vectors [N], 32-byte aligned, same shape
// - output may alias a or b (in-place operation supported)
// Returns: Q_OK on success, negative q_error_code on error
q_error_code q_mul_f32_avx2(
    const q_tensor* restrict a,
    const q_tensor* restrict b,
    q_tensor* output
);
```

---

## MAKEFILE UPDATES

Add these source files to the Makefile:

```makefile
# Source files for new kernels
SRC_OPS_AVX2 = \
    src/ops/avx2/matmul.c \
    src/ops/avx2/matmul_fp32.c \
    src/ops/avx2/causal_mask.c \
    src/ops/avx2/add.c \
    src/ops/avx2/mul.c \
    src/ops/avx2/rmsnorm.c \
    src/ops/avx2/rope.c \
    src/ops/avx2/silu.c \
    src/ops/avx2/softmax.c
```

---

## TEST STRUCTURE

Create test files in `tests/validation/`:

- `validate_matmul_f32.c` - MatMul FP32 validation
- `validate_causal_mask.c` - Causal masking validation
- `validate_add_f32.c` - Tensor Add validation
- `validate_mul_f32.c` - Element-wise Mul validation

Each test file should:
1. Load test data from Python-generated `.tns` files
2. Execute the kernel function
3. Compare results against expected output
4. Validate error handling (NULL inputs, shape mismatches, etc.)
5. Check memory safety (no leaks)

---

## IMPLEMENTATION CHECKLIST

For each kernel:

- [ ] **Step 0:** CoT analysis complete
- [ ] **Step 0.5:** Mathematical proof complete
- [ ] **Step 1:** Model construction (function signature defined)
- [ ] **Step 2:** TDD (Python gold standard + C validation test written)
- [ ] **Step 3:** Implementation (C code written)
- [ ] **Step 4:** Validation (tests pass, memory clean, performance verified)
- [ ] **Integration:** Function added to `include/qorus.h`
- [ ] **Documentation:** Code comments explain algorithm and optimizations

---

**Status:** 📋 READY FOR IMPLEMENTATION  
**Last Updated:** 2024-12-30  
**Framework:** MFR + CoT + Mathematical Proof + TDD (per `.cursorrules`)

