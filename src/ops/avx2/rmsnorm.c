#include "qorus.h"
#include <immintrin.h>

// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
// Optimized with rsqrt + Newton-Raphson refinement
//
// Algorithm:
// 1. Compute sum of squares (x^2) using AVX2
// 2. Compute mean: sum / N
// 3. Compute rsqrt(mean + eps) with Newton-Raphson refinement
// 4. Multiply x by rsqrt and weight
//
// Time Complexity: O(N) where N = vector length
// Space Complexity: O(1) - only AVX2 registers
q_error_code q_rmsnorm_f32_avx2(
    const float* restrict x,      // Input vector [N]
    const float* restrict weight, // Weight vector [N] (gamma)
    float* restrict output,      // Output vector [N]
    uint32_t N,                  // Vector length (must be multiple of 8)
    float eps                    // Epsilon for numerical stability
) {
    // Security: Critical validations (always active)
    Q_VALIDATE_PTR_OR_RETURN(x, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(weight, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(output, Q_ERR_INVALID_ARG);
    Q_VALIDATE_ALIGNED_OR_RETURN(x, Q_ERR_MISALIGNED);
    Q_VALIDATE_ALIGNED_OR_RETURN(weight, Q_ERR_MISALIGNED);
    Q_VALIDATE_ALIGNED_OR_RETURN(output, Q_ERR_MISALIGNED);
    
    Q_VALIDATE_NONZERO_OR_RETURN(N, Q_ERR_INVALID_SIZE);
    Q_VALIDATE_MULTIPLE_OR_RETURN(N, 8, Q_ERR_INVALID_SIZE);
    
    // Step 1: Compute sum of squares (x^2)
    __m256 sum_sq = _mm256_setzero_ps();
    
    const uint32_t vec_count = N / 8;
    for (uint32_t i = 0; i < vec_count; i++) {
        __m256 x_vec = _mm256_load_ps(x + i * 8);
        __m256 x_sq = _mm256_mul_ps(x_vec, x_vec);
        sum_sq = _mm256_add_ps(sum_sq, x_sq);
    }
    
    // Horizontal reduction: Sum all 8 elements
    // sum_sq = [s0, s1, s2, s3, s4, s5, s6, s7]
    __m128 low = _mm256_extractf128_ps(sum_sq, 0);
    __m128 high = _mm256_extractf128_ps(sum_sq, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    
    // Horizontal add
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    __m128 shuf2 = _mm_movehl_ps(shuf, sums);
    float sum_val = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
    
    // Step 2: Compute mean
    float mean_sq = sum_val / (float)N;
    
    // Step 3: Compute rsqrt(mean + eps) with Newton-Raphson refinement
    // rsqrt_ps gives approximate 1/sqrt(x) with ~12 bits precision
    // Newton-Raphson: r = r * (3 - x * r^2) / 2
    // This refines to ~22 bits precision (sufficient for FP32)
    __m256 mean_eps = _mm256_set1_ps(mean_sq + eps);
    __m256 rsqrt_approx = _mm256_rsqrt_ps(mean_eps);
    
    // Newton-Raphson refinement: r = r * (3 - x * r^2) / 2
    __m256 three = _mm256_set1_ps(3.0f);
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 r_sq = _mm256_mul_ps(rsqrt_approx, rsqrt_approx);
    __m256 x_r_sq = _mm256_mul_ps(mean_eps, r_sq);
    __m256 three_minus = _mm256_sub_ps(three, x_r_sq);
    __m256 rsqrt_refined = _mm256_mul_ps(_mm256_mul_ps(rsqrt_approx, three_minus), half);
    
    // Extract scalar (all elements are the same)
    float rsqrt_val = _mm_cvtss_f32(_mm256_extractf128_ps(rsqrt_refined, 0));
    
    // Broadcast rsqrt for vectorized multiplication
    __m256 rsqrt_vec = _mm256_set1_ps(rsqrt_val);
    
    // Step 4: Multiply x by rsqrt and weight
    for (uint32_t i = 0; i < vec_count; i++) {
        __m256 x_vec = _mm256_load_ps(x + i * 8);
        __m256 w_vec = _mm256_load_ps(weight + i * 8);
        
        // output = x * rsqrt * weight
        __m256 normalized = _mm256_mul_ps(x_vec, rsqrt_vec);
        __m256 result = _mm256_mul_ps(normalized, w_vec);
        
        _mm256_store_ps(output + i * 8, result);
    }
    
    return Q_OK;
}
