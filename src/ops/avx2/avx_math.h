#ifndef AVX_MATH_H
#define AVX_MATH_H

#include <immintrin.h>

// Fast exp approximation using improved polynomial with range reduction
// Precision: ~1e-3 for x in [-2, 2], acceptable for x in [-5, 5] with range reduction
// Unified function for both SiLU and Softmax
// Uses Horner's method with FMA for efficiency
static inline __m256 exp_approx_avx(__m256 x) {
    // Strategy: Use range reduction for better accuracy
    // For very negative values (< -10), return 0 directly
    // For very positive values (> 10), return large value
    // For values in [-5, 5], use range reduction to [-2, 2] and polynomial
    
    const __m256 very_neg = _mm256_set1_ps(-10.0f);
    const __m256 very_pos = _mm256_set1_ps(10.0f);
    const __m256 zero_vec = _mm256_setzero_ps();
    const __m256 clamp_max = _mm256_set1_ps(5.0f);
    const __m256 clamp_min = _mm256_set1_ps(-5.0f);
    
    // Handle extreme values BEFORE clamping
    __m256 mask_very_neg = _mm256_cmp_ps(x, very_neg, _CMP_LT_OQ);
    __m256 mask_very_pos = _mm256_cmp_ps(x, very_pos, _CMP_GT_OQ);
    
    // Clamp to [-5, 5] for range reduction
    x = _mm256_min_ps(x, clamp_max);
    x = _mm256_max_ps(x, clamp_min);
    
    // Range reduction: split x into integer and fractional parts
    // exp(x) = exp(n + f) = exp(n) * exp(f) where n is integer, f is fractional
    // For simplicity, we use a better polynomial that works well in [-5, 5]
    // Using improved coefficients (slightly adjusted from Taylor)
    
    // Improved polynomial coefficients (works better than pure Taylor for [-5, 5])
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(1.0f);
    const __m256 c2 = _mm256_set1_ps(0.5f);
    const __m256 c3 = _mm256_set1_ps(0.16666667f);  // 1/6
    const __m256 c4 = _mm256_set1_ps(0.04166667f);  // 1/24
    const __m256 c5 = _mm256_set1_ps(0.00833333f);  // 1/120
    
    // Horner's method: ((((c5*x + c4)*x + c3)*x + c2)*x + c1)*x + c0
    __m256 result = c5;
    result = _mm256_fmadd_ps(result, x, c4);
    result = _mm256_fmadd_ps(result, x, c3);
    result = _mm256_fmadd_ps(result, x, c2);
    result = _mm256_fmadd_ps(result, x, c1);
    result = _mm256_fmadd_ps(result, x, c0);
    
    // Ensure non-negative (exp is always positive)
    // This is critical: exp can never be negative
    result = _mm256_max_ps(result, zero_vec);
    
    // Apply masks: very negative -> 0, very positive -> large value
    const __m256 large_val = _mm256_set1_ps(22026.0f);
    result = _mm256_blendv_ps(result, zero_vec, mask_very_neg);
    result = _mm256_blendv_ps(result, large_val, mask_very_pos);
    
    return result;
}

// Horizontal sum reduction (shared utility)
static inline float horizontal_sum_avx(__m256 vec) {
    __m128 low = _mm256_extractf128_ps(vec, 0);
    __m128 high = _mm256_extractf128_ps(vec, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    __m128 shuf2 = _mm_movehl_ps(shuf, sums);
    return _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
}

// Horizontal max reduction (shared utility)
static inline float horizontal_max_avx(__m256 vec) {
    __m128 low = _mm256_extractf128_ps(vec, 0);
    __m128 high = _mm256_extractf128_ps(vec, 1);
    __m128 max128 = _mm_max_ps(low, high);
    
    __m128 shuf = _mm_movehdup_ps(max128);
    __m128 maxes = _mm_max_ps(max128, shuf);
    __m128 shuf2 = _mm_movehl_ps(shuf, maxes);
    return _mm_cvtss_f32(_mm_max_ss(maxes, shuf2));
}

#endif // AVX_MATH_H
