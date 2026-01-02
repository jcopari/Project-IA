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
    
    // Validação de Contrato de Layout (DEBUG apenas)
    // Se model.c mudar o layout (remover duplicação), isso detecta o bug imediatamente
    // Custo zero em RELEASE, mas previne corrupção silenciosa de inferência
    #ifdef DEBUG
    {
        const uint32_t num_pairs = N / 2;
        for (uint32_t i = 0; i < num_pairs; i++) {
            if (cos[i*2] != cos[i*2+1] || sin[i*2] != sin[i*2+1]) {
                fprintf(stderr, "FATAL: RoPE table corrupted/invalid layout at pair %u\n", i);
                fprintf(stderr, "  cos[%u]=%f, cos[%u]=%f\n", i*2, cos[i*2], i*2+1, cos[i*2+1]);
                fprintf(stderr, "  sin[%u]=%f, sin[%u]=%f\n", i*2, sin[i*2], i*2+1, sin[i*2+1]);
                fprintf(stderr, "  Expected layout: [c0, c0, c1, c1, ...]\n");
                abort();
            }
        }
    }
    #endif
    
    // Process 4 pairs (8 floats) per iteration
    const uint32_t vec_count = N / 8;
    
    for (uint32_t i = 0; i < vec_count; i++) {
        // 1. Load Data: [x0, y0, x1, y1, x2, y2, x3, y3]
        __m256 src = _mm256_load_ps(x + i * 8);
        
        // CORREÇÃO 3: Load Cos/Sin diretamente (layout já está correto: [c0, c0, c1, c1, c2, c2, c3, c3])
        // REMOVIDO: Toda lógica de load_ps(128-bit) + cast + insert + permute
        // O produtor (model.c) já garante o layout duplicado
        __m256 cos_vec = _mm256_load_ps(cos + i * 8);
        __m256 sin_vec = _mm256_load_ps(sin + i * 8);
        
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
    // CORREÇÃO 3: Layout duplicado - usar índice * 2
    const uint32_t remaining_start = vec_count * 8;
    for (uint32_t i = remaining_start; i < N; i += 2) {
        float x_val = x[i];
        float y_val = x[i + 1];
        uint32_t pair_idx = i / 2;
        float c = cos[pair_idx * 2]; // Layout duplicado: usar índice * 2
        float s = sin[pair_idx * 2];
        
        output[i] = x_val * c - y_val * s;
        output[i + 1] = y_val * c + x_val * s;
    }
    
    return Q_OK;
}
