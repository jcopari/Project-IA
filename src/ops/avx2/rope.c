#include "qorus.h"
#include <immintrin.h>

// RoPE: Rotate pairs (x, y) by angle theta
// Optimized using AVX2 addsub instruction for complex multiplication
//
// Formula:
//   x_out = x * cos - y * sin
//   y_out = y * cos + x * sin
//
// AVX2 Strategy:
//   1. Load vec = [x0, y0, x1, y1, x2, y2, x3, y3]
//   2. Load cos/sin and duplicate: [c0, c0, c1, c1, c2, c2, c3, c3]
//   3. Create vec_swap = [y0, x0, y1, x1, y2, x2, y3, x3] (using permute)
//   4. Compute term1 = vec * cos = [x0*c0, y0*c0, x1*c1, y1*c1, ...]
//   5. Compute term2 = vec_swap * sin = [y0*s0, x0*s0, y1*s1, x1*s1, ...]
//   6. Result = addsub(term1, term2)
//      Even lanes: term1 - term2 = x0*c0 - y0*s0 (Correct x_out)
//      Odd lanes:  term1 + term2 = y0*c0 + x0*s0 (Correct y_out)
//
// Time Complexity: O(N) where N = vector length
// Space Complexity: O(1) - only AVX2 registers
q_error_code q_rope_f32_avx2(
    const float* restrict x,     // Input vector [N] (must be even)
    const float* restrict cos,   // Cosine table [N/2]
    const float* restrict sin,   // Sine table [N/2]
    float* restrict output,      // Output vector [N]
    uint32_t N                   // Vector length (must be multiple of 8)
) {
    // Security: Critical validations (always active)
    Q_VALIDATE_PTR_OR_RETURN(x, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(cos, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(sin, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(output, Q_ERR_INVALID_ARG);
    Q_VALIDATE_ALIGNED_OR_RETURN(x, Q_ERR_MISALIGNED);
    Q_VALIDATE_ALIGNED_OR_RETURN(cos, Q_ERR_MISALIGNED);
    Q_VALIDATE_ALIGNED_OR_RETURN(sin, Q_ERR_MISALIGNED);
    Q_VALIDATE_ALIGNED_OR_RETURN(output, Q_ERR_MISALIGNED);
    
    Q_VALIDATE_NONZERO_OR_RETURN(N, Q_ERR_INVALID_SIZE);
    Q_VALIDATE_MULTIPLE_OR_RETURN(N, 8, Q_ERR_INVALID_SIZE);
    Q_VALIDATE_MULTIPLE_OR_RETURN(N, 2, Q_ERR_INVALID_SIZE); // Must be even
    
    // Process 4 pairs (8 floats) per iteration
    const uint32_t vec_count = N / 8;
    
    for (uint32_t i = 0; i < vec_count; i++) {
        // 1. Load Data: [x0, y0, x1, y1, x2, y2, x3, y3]
        __m256 src = _mm256_load_ps(x + i * 8);
        
        // 2. Load Cos/Sin: [c0, c1, c2, c3] -> Duplicate to [c0, c0, c1, c1, c2, c2, c3, c3]
        // Load 128-bit (4 floats)
        __m128 c_small = _mm_load_ps(cos + i * 4);
        __m128 s_small = _mm_load_ps(sin + i * 4);
        
        // Expand to 256-bit (duplicate 128-bit lane)
        // [c0, c1, c2, c3, c0, c1, c2, c3]
        __m256 c_dup = _mm256_castps128_ps256(c_small);
        c_dup = _mm256_insertf128_ps(c_dup, c_small, 1);
        
        __m256 s_dup = _mm256_castps128_ps256(s_small);
        s_dup = _mm256_insertf128_ps(s_dup, s_small, 1);
        
        // Shuffle to get [c0, c0, c1, c1, c2, c2, c3, c3]
        // CORRECTED: Use setr (reverse order) to get indices 0,0,1,1,2,2,3,3
        __m256i shuf_mask = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
        __m256 cos_vec = _mm256_permutevar8x32_ps(c_dup, shuf_mask);
        __m256 sin_vec = _mm256_permutevar8x32_ps(s_dup, shuf_mask);
        
        // 3. Create Swapped Vector: [y0, x0, y1, x1, y2, x2, y3, x3]
        // Permute mask: 0xB1 = 10 11 00 01 (Swap adjacent pairs)
        __m256 src_swap = _mm256_permute_ps(src, 0xB1);
        
        // 4. Compute Terms
        // term1 = [x0*c0, y0*c0, x1*c1, y1*c1, x2*c2, y2*c2, x3*c3, y3*c3]
        __m256 term1 = _mm256_mul_ps(src, cos_vec);
        
        // term2 = [y0*s0, x0*s0, y1*s1, x1*s1, y2*s2, x2*s2, y3*s3, x3*s3]
        __m256 term2 = _mm256_mul_ps(src_swap, sin_vec);
        
        // 5. AddSub: Even lanes subtract, Odd lanes add
        // result = [x0*c0 - y0*s0, y0*c0 + x0*s0, x1*c1 - y1*s1, y1*c1 + x1*s1, ...]
        // This is exactly [x0', y0', x1', y1', x2', y2', x3', y3']
        __m256 result = _mm256_addsub_ps(term1, term2);
        
        // 6. Store
        _mm256_store_ps(output + i * 8, result);
    }
    
    // Handle remaining pairs (if N % 8 != 0)
    const uint32_t remaining_start = vec_count * 8;
    for (uint32_t i = remaining_start; i < N; i += 2) {
        float x_val = x[i];
        float y_val = x[i + 1];
        uint32_t pair_idx = i / 2;
        float c = cos[pair_idx];
        float s = sin[pair_idx];
        
        output[i] = x_val * c - y_val * s;
        output[i + 1] = y_val * c + x_val * s;
    }
    
    return Q_OK;
}
