#include "qorus.h"
#include "avx_math.h"
#include <immintrin.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// SiLU: f(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Optimized with AVX2 exp approximation
//
// Algorithm:
// 1. Compute exp(-x) using polynomial approximation
// 2. Compute sigmoid(x) = 1 / (1 + exp(-x))
// 3. Multiply x * sigmoid(x)
//
// Time Complexity: O(N) where N = vector length
// Space Complexity: O(1) - only AVX2 registers
q_error_code q_silu_f32_avx2(
    const float* restrict x,
    float* restrict output,
    uint32_t N
) {
    // Security: Critical validations (always active)
    if (x == NULL) {
        return Q_ERR_INVALID_ARG;
    }
    
    if (output == NULL) {
        return Q_ERR_INVALID_ARG;
    }
    
    // Validate size
    if (N == 0) {
        return Q_ERR_INVALID_SIZE;
    }
    
    // DEBUG: Print parameters for diagnosis
    #ifdef DEBUG
    fprintf(stderr, "DEBUG: q_silu_f32_avx2: N=%u, x=%p (align=%zu), output=%p (align=%zu), N%%8=%u\n",
            N, (void*)x, ((uintptr_t)x % 32), (void*)output, ((uintptr_t)output % 32), N % 8);
    #endif
    
    // CRITICAL FIX: For small sizes (< 8), use scalar fallback (no alignment requirement)
    if (N < 8) {
        // Scalar implementation for small sizes
        for (uint32_t i = 0; i < N; i++) {
            float sigmoid = 1.0f / (1.0f + expf(-x[i]));
            output[i] = x[i] * sigmoid;
        }
        return Q_OK;
    }
    
    // For larger sizes (>= 8), validate alignment and use vectorized code
    if (((uintptr_t)x % 32) != 0) {
        return Q_ERR_MISALIGNED;
    }
    
    if (((uintptr_t)output % 32) != 0) {
        return Q_ERR_MISALIGNED;
    }
    
    if (N == 0) {
        return Q_ERR_INVALID_SIZE;
    }
    
    // CRITICAL FIX: Handle non-multiple-of-8 sizes
    // For sizes that are not multiple of 8, use vectorized code with tail handling
    const uint32_t vec_count = N / 8;
    // tail_count not used (handled by remaining loop below)
    const __m256 one = _mm256_set1_ps(1.0f);
    
    for (uint32_t i = 0; i < vec_count; i++) {
        __m256 x_vec = _mm256_load_ps(x + i * 8);
        
        // Compute sigmoid(x) = 1 / (1 + exp(-x))
        __m256 neg_x = _mm256_xor_ps(x_vec, _mm256_set1_ps(-0.0f)); // Negate
        __m256 exp_neg_x = exp_approx_avx(neg_x);
        __m256 one_plus_exp = _mm256_add_ps(one, exp_neg_x);
        __m256 sigmoid = _mm256_div_ps(one, one_plus_exp);
        
        // SiLU: x * sigmoid(x)
        __m256 result = _mm256_mul_ps(x_vec, sigmoid);
        
        _mm256_store_ps(output + i * 8, result);
    }
    
    // Tail handling: scalar fallback for remainder
    for (uint32_t i = vec_count * 8; i < N; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-x[i]));
        output[i] = x[i] * sigmoid;
    }
    
    return Q_OK;
}

