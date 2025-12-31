#include "qorus.h"
#include "avx_math.h"
#include <immintrin.h>
#include <math.h>

// Softmax: stable computation with max-sub trick
// output[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
//
// Algorithm:
// 1. Find maximum value (vectorized)
// 2. Subtract maximum from all values (stability)
// 3. Compute exp(x - max) and sum (vectorized)
// 4. Normalize by dividing by sum
//
// Time Complexity: O(N) where N = vector length
// Space Complexity: O(1) - only AVX2 registers
q_error_code q_softmax_f32_avx2(
    const float* restrict x,
    float* restrict output,
    uint32_t N
) {
    // Security: Critical validations (always active)
    Q_VALIDATE_PTR_OR_RETURN(x, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(output, Q_ERR_INVALID_ARG);
    
    Q_VALIDATE_NONZERO_OR_RETURN(N, Q_ERR_INVALID_SIZE);
    
    // DEBUG: Print parameters for diagnosis
    #ifdef DEBUG
    fprintf(stderr, "DEBUG: q_softmax_f32_avx2: N=%u, x=%p (align=%zu), output=%p (align=%zu)\n",
            N, (void*)x, ((uintptr_t)x % 32), (void*)output, ((uintptr_t)output % 32));
    #endif
    
    // CRITICAL FIX: For small sizes (< 8), use scalar fallback (no alignment requirement)
    if (N < 8) {
        // Scalar implementation for small sizes
        float max_val = x[0];
        for (uint32_t i = 1; i < N; i++) {
            if (x[i] > max_val) {
                max_val = x[i];
            }
        }
        
        float sum_val = 0.0f;
        for (uint32_t i = 0; i < N; i++) {
            float exp_val = expf(x[i] - max_val);
            output[i] = exp_val;
            sum_val += exp_val;
        }
        
        for (uint32_t i = 0; i < N; i++) {
            output[i] /= sum_val;
        }
        
        return Q_OK;
    }
    
    // For larger sizes (>= 8), validate alignment and use vectorized code
    Q_VALIDATE_ALIGNED_OR_RETURN(x, Q_ERR_MISALIGNED);
    Q_VALIDATE_ALIGNED_OR_RETURN(output, Q_ERR_MISALIGNED);
    
    // CRITICAL FIX: Handle non-multiple-of-8 sizes
    // For small sizes (< 8), use scalar fallback
    // For larger sizes, use vectorized code with tail handling
    const uint32_t vec_count = N / 8;
    // tail_count not used (handled by remaining loop below)
    
    // Step 1: Find maximum (vectorized + scalar tail)
    __m256 max_vec = _mm256_set1_ps(-INFINITY);
    for (uint32_t i = 0; i < vec_count; i++) {
        __m256 x_vec = _mm256_load_ps(x + i * 8);
        max_vec = _mm256_max_ps(max_vec, x_vec);
    }
    float max_val = horizontal_max_avx(max_vec);
    
    // Handle tail for maximum
    for (uint32_t i = vec_count * 8; i < N; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    __m256 max_broadcast = _mm256_set1_ps(max_val);
    
    // Step 2: Compute exp(x - max) and sum (vectorized + scalar tail)
    __m256 sum_vec = _mm256_setzero_ps();
    for (uint32_t i = 0; i < vec_count; i++) {
        __m256 x_vec = _mm256_load_ps(x + i * 8);
        __m256 shifted = _mm256_sub_ps(x_vec, max_broadcast);
        __m256 exp_vec = exp_approx_avx(shifted);
        sum_vec = _mm256_add_ps(sum_vec, exp_vec);
        _mm256_store_ps(output + i * 8, exp_vec);
    }
    float sum_val = horizontal_sum_avx(sum_vec);
    
    // Handle tail for exp and sum
    for (uint32_t i = vec_count * 8; i < N; i++) {
        float exp_val = expf(x[i] - max_val);
        output[i] = exp_val;
        sum_val += exp_val;
    }
    
    __m256 sum_broadcast = _mm256_set1_ps(sum_val);
    
    // Step 3: Normalize (divide by sum) - vectorized + scalar tail
    for (uint32_t i = 0; i < vec_count; i++) {
        __m256 exp_vec = _mm256_load_ps(output + i * 8);
        __m256 normalized = _mm256_div_ps(exp_vec, sum_broadcast);
        _mm256_store_ps(output + i * 8, normalized);
    }
    
    // Handle tail for normalization
    for (uint32_t i = vec_count * 8; i < N; i++) {
        output[i] /= sum_val;
    }
    
    #ifdef DEBUG
    // Validate invariant: sum should be ≈ 1.0
    float check_sum = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        check_sum += output[i];
    }
    if (fabsf(check_sum - 1.0f) > 1e-4f) {
        fprintf(stderr, "WARNING: q_softmax_f32_avx2: Sum = %.6f (expected 1.0)\n", check_sum);
    }
    #endif
    
    return Q_OK;
}

