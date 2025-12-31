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
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"silu.c:%d\",\"message\":\"q_silu_f32_avx2 ENTRY\",\"data\":{\"x\":\"%p\",\"output\":\"%p\",\"N\":%u,\"x_is_null\":%d,\"output_is_null\":%d,\"N_is_zero\":%d},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"E\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                    (void*)x, (void*)output, N, (x == NULL ? 1 : 0), (output == NULL ? 1 : 0), (N == 0 ? 1 : 0));
            fclose(log_file);
        }
    }
    // #endregion
    
    // Security: Critical validations (always active)
    if (x == NULL) {
        // #region agent log
        {
            FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
            if (log_file) {
                fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"silu.c:%d\",\"message\":\"VALIDATION FAILED: x is NULL\",\"data\":{\"error\":\"Q_ERR_INVALID_ARG\"},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"E\"}\n",
                        (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__);
                fclose(log_file);
            }
        }
        // #endregion
        return Q_ERR_INVALID_ARG;
    }
    
    if (output == NULL) {
        // #region agent log
        {
            FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
            if (log_file) {
                fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"silu.c:%d\",\"message\":\"VALIDATION FAILED: output is NULL\",\"data\":{\"error\":\"Q_ERR_INVALID_ARG\"},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"E\"}\n",
                        (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__);
                fclose(log_file);
            }
        }
        // #endregion
        return Q_ERR_INVALID_ARG;
    }
    
    // DEBUG: Print parameters for diagnosis
    #ifdef DEBUG
    fprintf(stderr, "DEBUG: q_silu_f32_avx2: N=%u, x=%p (align=%zu), output=%p (align=%zu), N%%8=%u\n",
            N, (void*)x, ((uintptr_t)x % 32), (void*)output, ((uintptr_t)output % 32), N % 8);
    #endif
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"silu.c:%d\",\"message\":\"AFTER pointer validation\",\"data\":{\"x_align\":%zu,\"output_align\":%zu,\"N\":%u},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"A,B\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                    ((uintptr_t)x % 32), ((uintptr_t)output % 32), N);
            fclose(log_file);
        }
    }
    // #endregion
    
    // CRITICAL FIX: For small sizes (< 8), use scalar fallback (no alignment requirement)
    if (N < 8) {
        // #region agent log
        {
            FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
            if (log_file) {
                fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"silu.c:%d\",\"message\":\"Using scalar fallback (N < 8)\",\"data\":{\"N\":%u},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"C\"}\n",
                        (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__, N);
                fclose(log_file);
            }
        }
        // #endregion
        // Scalar implementation for small sizes
        for (uint32_t i = 0; i < N; i++) {
            float sigmoid = 1.0f / (1.0f + expf(-x[i]));
            output[i] = x[i] * sigmoid;
        }
        // #region agent log
        {
            FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
            if (log_file) {
                fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"silu.c:%d\",\"message\":\"Scalar fallback completed\",\"data\":{\"ret\":\"Q_OK\"},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"C\"}\n",
                        (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__);
                fclose(log_file);
            }
        }
        // #endregion
        return Q_OK;
    }
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"silu.c:%d\",\"message\":\"BEFORE alignment validation\",\"data\":{\"x_align\":%zu,\"output_align\":%zu,\"N\":%u},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"A,B\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                    ((uintptr_t)x % 32), ((uintptr_t)output % 32), N);
            fclose(log_file);
        }
    }
    // #endregion
    
    // For larger sizes (>= 8), validate alignment and use vectorized code
    if (((uintptr_t)x % 32) != 0) {
        // #region agent log
        {
            FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
            if (log_file) {
                fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"silu.c:%d\",\"message\":\"VALIDATION FAILED: x not aligned\",\"data\":{\"x_align\":%zu,\"error\":\"Q_ERR_MISALIGNED\"},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"A\"}\n",
                        (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__, ((uintptr_t)x % 32));
                fclose(log_file);
            }
        }
        // #endregion
        return Q_ERR_MISALIGNED;
    }
    
    if (((uintptr_t)output % 32) != 0) {
        // #region agent log
        {
            FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
            if (log_file) {
                fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"silu.c:%d\",\"message\":\"VALIDATION FAILED: output not aligned\",\"data\":{\"output_align\":%zu,\"error\":\"Q_ERR_MISALIGNED\"},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"A\"}\n",
                        (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__, ((uintptr_t)output % 32));
                fclose(log_file);
            }
        }
        // #endregion
        return Q_ERR_MISALIGNED;
    }
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"silu.c:%d\",\"message\":\"BEFORE N validation\",\"data\":{\"N\":%u,\"N_is_zero\":%d},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"C\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__, N, (N == 0 ? 1 : 0));
            fclose(log_file);
        }
    }
    // #endregion
    
    if (N == 0) {
        // #region agent log
        {
            FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
            if (log_file) {
                fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"silu.c:%d\",\"message\":\"VALIDATION FAILED: N is zero\",\"data\":{\"error\":\"Q_ERR_INVALID_SIZE\"},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"C\"}\n",
                        (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__);
                fclose(log_file);
            }
        }
        // #endregion
        return Q_ERR_INVALID_SIZE;
    }
    
    // CRITICAL FIX: Handle non-multiple-of-8 sizes
    // For sizes that are not multiple of 8, use vectorized code with tail handling
    const uint32_t vec_count = N / 8;
    const uint32_t tail_count = N % 8;
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
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"silu.c:%d\",\"message\":\"q_silu_f32_avx2 EXIT SUCCESS\",\"data\":{\"ret\":\"Q_OK\",\"vec_count\":%u,\"tail_count\":%u},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"A,B,C\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__, vec_count, tail_count);
            fclose(log_file);
        }
    }
    // #endregion
    
    return Q_OK;
}

