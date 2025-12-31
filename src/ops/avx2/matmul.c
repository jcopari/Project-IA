#include "qorus.h"
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

// GEMV Q4_F32: Matrix Q4_0 * Vector F32 -> Vector F32
// Critical operation for Llama-3 inference (called millions of times per token)
//
// Algorithm (Fused Dequantization + Dot Product):
// 1. For each output row i:
//    a. Process Q4_0 blocks in row (32 values per block)
//    b. Dequantize block on-the-fly (in registers only, no memory)
//    c. Compute dot product with input vector using AVX2/FMA
//    d. Accumulate result in YMM registers
//    e. Horizontal reduction to scalar
//
// Time Complexity: O(M * N) where M=rows, N=cols
// Space Complexity: O(1) - only AVX2 registers (no stack buffers)
// Latency: ~10-15 cycles per block (dequantize) + ~8 cycles (FMA dot product)
//
// Preconditions:
// - weights: Q4_0 matrix [M, N], block-aligned (N must be multiple of 32)
// - input: F32 vector [N], 32-byte aligned
// - output: F32 vector [M], 32-byte aligned
// - M, N > 0, N % 32 == 0
// - input and output must NOT alias (checked in DEBUG mode)

// Constantes fundamentais
#define Q4_0_ZERO_POINT 8.0f

// Helper Function: Processa um bloco Q4_0
// Otimizado para minimizar pressão em registradores via reutilização explícita de variáveis.
// O atributo always_inline permite que o compilador otimize o contexto completo (SROA).
static inline __attribute__((always_inline)) __m256 q_process_block_avx2(
    const q_block_q4_0* restrict block,
    const float* restrict input_ptr,
    __m256 acc,
    const __m128i low_mask
) {
    // 1. Load Scale & Setup Offset
    // Broadcast scale once per block
    const float scale = block->scale;
    const __m256 scale_vec = _mm256_broadcast_ss(&scale);
    const __m256 offset_vec = _mm256_mul_ps(_mm256_set1_ps(-Q4_0_ZERO_POINT), scale_vec);

    // 2. Load Quantized Data (16 bytes)
    // Unaligned load is safe on modern AVX2 CPUs
    const __m128i raw = _mm_loadu_si128((const __m128i*)block->qs);

    // 3. Separate Nibbles
    const __m128i low  = _mm_and_si128(raw, low_mask);
    const __m128i high = _mm_and_si128(_mm_srli_epi16(raw, 4), low_mask);
    
    // Interleave to restore order
    const __m128i v0_15  = _mm_unpacklo_epi8(low, high);
    const __m128i v16_31 = _mm_unpackhi_epi8(low, high);

    // 4. FMA Batches
    // Explicitly reusing 'w' to hint register reuse to the compiler
    __m256 w;

    // Batch 1 (0-7)
    w = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v0_15));
    w = _mm256_fmadd_ps(w, scale_vec, offset_vec);
    acc = _mm256_fmadd_ps(w, _mm256_load_ps(input_ptr + 0), acc);

    // Batch 2 (8-15)
    // Shift v0_15 to access upper half. v0_15 can be retired after this.
    __m128i v8_15 = _mm_bsrli_si128(v0_15, 8);
    w = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v8_15));
    w = _mm256_fmadd_ps(w, scale_vec, offset_vec);
    acc = _mm256_fmadd_ps(w, _mm256_load_ps(input_ptr + 8), acc);

    // Batch 3 (16-23)
    w = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v16_31));
    w = _mm256_fmadd_ps(w, scale_vec, offset_vec);
    acc = _mm256_fmadd_ps(w, _mm256_load_ps(input_ptr + 16), acc);

    // Batch 4 (24-31)
    // Shift v16_31. v16_31 can be retired after this.
    __m128i v24_31 = _mm_bsrli_si128(v16_31, 8);
    w = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v24_31));
    w = _mm256_fmadd_ps(w, scale_vec, offset_vec);
    acc = _mm256_fmadd_ps(w, _mm256_load_ps(input_ptr + 24), acc);

    return acc;
}

q_error_code q_gemv_q4_f32_avx2(
    const q_tensor* restrict weights,  // Q4_0 matrix [M, N]
    const float* restrict input,         // F32 vector [N]
    float* restrict output              // F32 vector [M] (output)
) {
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"matmul.c:%d\",\"message\":\"q_gemv_q4_f32_avx2 ENTRY\",\"data\":{\"weights\":\"%p\",\"input\":\"%p\",\"output\":\"%p\",\"weights_ne\":[%u,%u,%u,%u],\"input_is_null\":%d,\"output_is_null\":%d},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"G\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                    (void*)weights, (void*)input, (void*)output,
                    weights ? weights->ne[0] : 0, weights ? weights->ne[1] : 0, weights ? weights->ne[2] : 0, weights ? weights->ne[3] : 0,
                    (input == NULL ? 1 : 0), (output == NULL ? 1 : 0));
            fclose(log_file);
        }
    }
    // #endregion
    
    // Security: Critical validations (always active, optimized for Release)
    Q_VALIDATE_PTR_OR_RETURN(weights, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(input, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(output, Q_ERR_INVALID_ARG);
    
    Q_VALIDATE_ALIGNED_OR_RETURN(input, Q_ERR_MISALIGNED);
    Q_VALIDATE_ALIGNED_OR_RETURN(output, Q_ERR_MISALIGNED);
    
    // CRITICAL: Aliasing check (always active)
    Q_VALIDATE_OR_RETURN(input != output, Q_ERR_ALIASING);
    
    const uint32_t M = weights->ne[0];  // Number of rows (output size)
    const uint32_t N = weights->ne[1];  // Number of cols (input size)
    
    // DEBUG: Print dimensions for diagnosis
    char debug_msg[256];
    int debug_len = snprintf(debug_msg, sizeof(debug_msg),
        "DEBUG: q_gemv_q4_f32_avx2: M=%u, N=%u, N%%32=%u\n", M, N, N % 32);
    write(2, debug_msg, (size_t)debug_len);
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"matmul.c:%d\",\"message\":\"BEFORE dimension validation\",\"data\":{\"M\":%u,\"N\":%u,\"M_is_zero\":%d,\"N_is_zero\":%d,\"N_mod32\":%u},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"G\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                    M, N, (M == 0 ? 1 : 0), (N == 0 ? 1 : 0), N % 32);
            fclose(log_file);
        }
    }
    // #endregion
    
    // Security: Dimension validations (always active)
    if (M == 0) {
        // #region agent log
        {
            FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
            if (log_file) {
                fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"matmul.c:%d\",\"message\":\"VALIDATION FAILED: M is zero\",\"data\":{\"error\":\"Q_ERR_INVALID_SIZE\"},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"G\"}\n",
                        (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__);
                fclose(log_file);
            }
        }
        // #endregion
        char err_msg[128];
        int err_len = snprintf(err_msg, sizeof(err_msg),
            "ERROR: q_gemv_q4_f32_avx2: M (weights->ne[0]) is zero\n");
        write(2, err_msg, (size_t)err_len);
        #ifdef DEBUG
        abort();
        #endif
        return Q_ERR_INVALID_SIZE;
    }
    if (N == 0) {
        char err_msg[128];
        int err_len = snprintf(err_msg, sizeof(err_msg),
            "ERROR: q_gemv_q4_f32_avx2: N (weights->ne[1]) is zero\n");
        write(2, err_msg, (size_t)err_len);
        #ifdef DEBUG
        abort();
        #endif
        return Q_ERR_INVALID_SIZE;
    }
    if (N % 32 != 0) {
        char err_msg[128];
        int err_len = snprintf(err_msg, sizeof(err_msg),
            "ERROR: q_gemv_q4_f32_avx2: N (weights->ne[1]=%u) is not multiple of 32\n", N);
        write(2, err_msg, (size_t)err_len);
        #ifdef DEBUG
        abort();
        #endif
        return Q_ERR_INVALID_SIZE;
    }
    
    // Security: Type validation (always active)
    if (weights->type != Q_Q4_0) {
        char err_msg[128];
        int err_len = snprintf(err_msg, sizeof(err_msg),
            "ERROR: q_gemv_q4_f32_avx2: weights->type=%d (expected Q_Q4_0=%d)\n",
            weights->type, Q_Q4_0);
        write(2, err_msg, (size_t)err_len);
        return Q_ERR_INVALID_DTYPE;
    }
    
    // Security: Overflow validation (always active)
    const uint32_t blocks_per_row = N / 32;
    
    // DEBUG: Print overflow validation info
    #ifdef DEBUG
    char overflow_debug[256];
    int overflow_debug_len = snprintf(overflow_debug, sizeof(overflow_debug),
        "DEBUG: q_gemv_q4_f32_avx2: Overflow check: M=%u, blocks_per_row=%u, M*blocks_per_row=%llu\n",
        M, blocks_per_row, (unsigned long long)M * (unsigned long long)blocks_per_row);
    write(2, overflow_debug, (size_t)overflow_debug_len);
    #endif
    
    // CRITICAL: Check for overflow in M * blocks_per_row
    // This is used to calculate total number of blocks = M * blocks_per_row
    if (M > 0 && blocks_per_row > UINT32_MAX / M) {
        char err_msg[128];
        int err_len = snprintf(err_msg, sizeof(err_msg),
            "ERROR: q_gemv_q4_f32_avx2: Overflow in M*blocks_per_row: M=%u, blocks_per_row=%u\n",
            M, blocks_per_row);
        write(2, err_msg, (size_t)err_len);
        return Q_ERR_OVERFLOW;
    }
    
    const q_block_q4_0* restrict weight_blocks = (const q_block_q4_0* restrict)weights->data;
    const __m128i low_mask = _mm_set1_epi8(0x0F);
    
    // Process each output row
    for (uint32_t i = 0; i < M; i++) {
        // Initialize 4 accumulators to hide FMA latency (Instruction Level Parallelism)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        
        // CRITICAL FIX: Use size_t for pointer arithmetic to avoid overflow
        const size_t row_offset = (size_t)i * (size_t)blocks_per_row;
        const q_block_q4_0* restrict row_blocks = weight_blocks + row_offset;
        const float* restrict x_ptr = input;
        
        // Main Loop: Unrolling 4x
        // This is the hot path. Compiler should interleave instructions from the 4 calls.
        const uint32_t num_block_groups = blocks_per_row / 4;
        
        for (uint32_t bg = 0; bg < num_block_groups; bg++) {
            // ROBUSTNESS: Use size_t for offset calculations to prevent uint32_t wraparound
            // in pointer arithmetic, even though validation guarantees safety
            const size_t block_base = (size_t)(bg * 4);
            
            // Process 4 blocks in parallel (instruction level parallelism)
            // Cast to size_t ensures multiplication happens in wider type before pointer arithmetic
            acc0 = q_process_block_avx2(&row_blocks[block_base + 0], x_ptr + (block_base + 0) * 32, acc0, low_mask);
            acc1 = q_process_block_avx2(&row_blocks[block_base + 1], x_ptr + (block_base + 1) * 32, acc1, low_mask);
            acc2 = q_process_block_avx2(&row_blocks[block_base + 2], x_ptr + (block_base + 2) * 32, acc2, low_mask);
            acc3 = q_process_block_avx2(&row_blocks[block_base + 3], x_ptr + (block_base + 3) * 32, acc3, low_mask);
        }
        
        // Tail Loop: Handle remaining blocks (0 to 3 blocks)
        // Optimized for K=0 case (common in LLMs) where cost is near zero due to branch prediction.
        // We use sequential IFs to avoid computing unused blocks (which branchless masking would do).
        const uint32_t remaining = blocks_per_row % 4;
        // ROBUSTNESS: Use size_t for tail_start to match block_base type consistency
        const size_t tail_start = (size_t)(num_block_groups * 4);

        if (remaining > 0) {
            acc0 = q_process_block_avx2(&row_blocks[tail_start + 0], x_ptr + (tail_start + 0) * 32, acc0, low_mask);
        }
        if (remaining > 1) {
            acc1 = q_process_block_avx2(&row_blocks[tail_start + 1], x_ptr + (tail_start + 1) * 32, acc1, low_mask);
        }
        if (remaining > 2) {
            acc2 = q_process_block_avx2(&row_blocks[tail_start + 2], x_ptr + (tail_start + 2) * 32, acc2, low_mask);
        }
        
        // Horizontal reduction
        // Note: Order of addition may affect precision, but is acceptable for LLM inference
        // 1. Vector reduction
        __m256 sum01 = _mm256_add_ps(acc0, acc1);
        __m256 sum23 = _mm256_add_ps(acc2, acc3);
        __m256 sum = _mm256_add_ps(sum01, sum23);
        
        // 2. Lane reduction (256 -> 128)
        __m128 low = _mm256_extractf128_ps(sum, 0);
        __m128 high = _mm256_extractf128_ps(sum, 1);
        __m128 sum128 = _mm_add_ps(low, high);
        
        // 3. Horizontal add within 128-bit lane
        __m128 shuf = _mm_movehdup_ps(sum128);
        __m128 sums = _mm_add_ps(sum128, shuf);
        __m128 shuf2 = _mm_movehl_ps(shuf, sums);
        float final_sum = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
        
        output[i] = final_sum;
    }
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"matmul.c:%d\",\"message\":\"q_gemv_q4_f32_avx2 EXIT SUCCESS\",\"data\":{\"ret\":\"Q_OK\",\"M\":%u,\"N\":%u},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"G\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__, M, N);
            fclose(log_file);
        }
    }
    // #endregion
    
    return Q_OK;
}

