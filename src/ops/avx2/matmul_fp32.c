#include "qorus.h"
#include <immintrin.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

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
// Transposes src[rows, cols] to dst[cols, rows]
// NOTE: Currently unused (we use simple transpose), but kept for future optimization
static void __attribute__((unused)) transpose_blocked(
    const float* restrict src,
    float* restrict dst,
    uint32_t rows,
    uint32_t cols,
    size_t src_stride,  // Stride of src in elements
    size_t dst_stride   // Stride of dst in elements
) {
    for (uint32_t i = 0; i < rows; i += MATMUL_BLOCK_SIZE) {
        uint32_t i_limit = (i + MATMUL_BLOCK_SIZE < rows) ? 
                           i + MATMUL_BLOCK_SIZE : rows;
        
        for (uint32_t j = 0; j < cols; j += MATMUL_BLOCK_SIZE) {
            uint32_t j_limit = (j + MATMUL_BLOCK_SIZE < cols) ? 
                               j + MATMUL_BLOCK_SIZE : cols;
            
            for (uint32_t ii = i; ii < i_limit; ii++) {
                for (uint32_t jj = j; jj < j_limit; jj++) {
                    dst[jj * dst_stride + ii] = src[ii * src_stride + jj];
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
    
    // DEBUG: Print dimensions for diagnosis (only in DEBUG mode)
    #ifdef DEBUG
    char debug_msg[256];
    int debug_len = snprintf(debug_msg, sizeof(debug_msg),
        "DEBUG: q_matmul_f32_avx2: A[%u,%u] @ B[%u,%u] -> C[%u,%u]\n",
        M, K, B->ne[0], N, C->ne[0], C->ne[1]);
    write(2, debug_msg, (size_t)debug_len);
    #endif
    
    // Validate dimensions
    if (M == 0) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: q_matmul_f32_avx2: M (A->ne[0]) is zero\n");
        abort();
        #endif
        return Q_ERR_INVALID_SIZE;
    }
    if (K == 0) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: q_matmul_f32_avx2: K (A->ne[1]) is zero\n");
        abort();
        #endif
        return Q_ERR_INVALID_SIZE;
    }
    if (N == 0) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: q_matmul_f32_avx2: N (B->ne[1]) is zero\n");
        abort();
        #endif
        return Q_ERR_INVALID_SIZE;
    }
    
    // Validate shape compatibility: A[M,K] @ B[K,N] → C[M,N]
    if (B->ne[0] != K) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: q_matmul_f32_avx2: B->ne[0] (%u) != K (%u)\n", B->ne[0], K);
        abort();
        #endif
        return Q_ERR_INVALID_SIZE;
    }
    if (C->ne[0] != M) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: q_matmul_f32_avx2: C->ne[0] (%u) != M (%u)\n", C->ne[0], M);
        abort();
        #endif
        return Q_ERR_INVALID_SIZE;
    }
    if (C->ne[1] != N) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: q_matmul_f32_avx2: C->ne[1] (%u) != N (%u)\n", C->ne[1], N);
        abort();
        #endif
        return Q_ERR_INVALID_SIZE;
    }
    
    // CRITICAL FIX: For transposed tensors (B->nb[0] == sizeof(float)),
    // we cannot guarantee alignment of all elements.
    // We need to use unaligned loads (_mm256_loadu_ps) instead of aligned loads (_mm256_load_ps).
    
    // Determine if we need unaligned loads based on stride
    // If stride is not multiple of 32 bytes, elements may not be aligned
    bool A_needs_unaligned = (A->nb[0] % 32 != 0);
    bool B_needs_unaligned = (B->nb[0] % 32 != 0);
    // C_needs_unaligned not used (C is always written, alignment checked separately)
    
    // CRITICAL FIX: Only validate base pointer alignment for tensors that need aligned loads
    // For tensors with stride not multiple of 32 bytes, we'll use unaligned loads anyway
    // Use explicit 32-byte alignment check (not Q_ALIGN which is 64)
    if (!A_needs_unaligned) {
        // Only validate if we're going to use aligned loads (32-byte alignment required)
        uintptr_t A_addr = (uintptr_t)A->data;
        if (A_addr % 32 != 0) {
            // Always print debug info (not just in DEBUG mode) to diagnose alignment issues
            fprintf(stderr, "ERROR: q_matmul_f32_avx2: A->data not 32-byte aligned: %p (offset: %zu)\n",
                    A->data, A_addr % 32);
            fprintf(stderr, "  A->nb[0]=%zu, A_needs_unaligned=%d\n", A->nb[0], A_needs_unaligned);
            fprintf(stderr, "  A->ne[0]=%u, A->ne[1]=%u\n", A->ne[0], A->ne[1]);
            fprintf(stderr, "  B->ne[0]=%u, B->ne[1]=%u, B->nb[0]=%zu\n", B->ne[0], B->ne[1], B->nb[0]);
            fprintf(stderr, "  C->ne[0]=%u, C->ne[1]=%u, C->nb[0]=%zu\n", C->ne[0], C->ne[1], C->nb[0]);
            #ifdef DEBUG
            abort();
            #endif
            return Q_ERR_MISALIGNED;
        }
    }
    // B can be unaligned (transposed tensor with stride=4), skip validation
    // C can be unaligned (transposed output), skip validation
    
    // Validate types
    Q_VALIDATE_OR_RETURN(A->type == Q_F32, Q_ERR_INVALID_DTYPE);
    Q_VALIDATE_OR_RETURN(B->type == Q_F32, Q_ERR_INVALID_DTYPE);
    Q_VALIDATE_OR_RETURN(C->type == Q_F32, Q_ERR_INVALID_DTYPE);
    
    // Get data pointers
    const float* restrict A_data = (const float*)A->data;
    const float* restrict B_data = (const float*)B->data;
    float* C_data = (float*)C->data;
    
    // Calculate strides in elements (not bytes)
    // nb[0] is stride in bytes for dimension 0 (row stride)
    const size_t A_stride = A->nb[0] / sizeof(float);  // Row stride of A in elements
    const size_t B_stride = B->nb[0] / sizeof(float);  // Row stride of B in elements
    const size_t C_stride = C->nb[0] / sizeof(float);  // Row stride of C in elements
    
    // DEBUG: Print strides for diagnosis
    #ifdef DEBUG
    char stride_debug[256];
    int stride_debug_len = snprintf(stride_debug, sizeof(stride_debug),
        "DEBUG: q_matmul_f32_avx2: strides A=%zu (K=%u), B=%zu (N=%u), C=%zu (N=%u)\n",
        A_stride, K, B_stride, N, C_stride, N);
    write(2, stride_debug, (size_t)stride_debug_len);
    #endif
    
    // CRITICAL FIX: Validate strides are reasonable (prevent overflow)
    // For transposed tensors (B->nb[0] == sizeof(float)), B_stride = 1, which is valid
    // We need to check B->nb[1] (column stride) instead for transposed tensors
    if (A_stride < K) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: q_matmul_f32_avx2: A_stride (%zu) < K (%u)\n", A_stride, K);
        abort();
        #endif
        return Q_ERR_INVALID_SIZE;
    }
    // CRITICAL FIX: For transposed B, check column stride instead
    // A transposed tensor has:
    // - B->nb[0] = sizeof(float) (row stride = 1 element)
    // - B->nb[1] = K * sizeof(float) (column stride = K elements)
    // - B->ne[0] = K (number of rows in transposed = columns in original)
    // - B->ne[1] = N (number of cols in transposed = rows in original)
    // But we need to distinguish from a normal tensor where ne[0] = 1 (which also has nb[0] = sizeof(float))
    bool is_transposed = (B->nb[0] == sizeof(float)) && (B->ne[0] > 1) && (B->nb[1] > sizeof(float));
    
    if (is_transposed) {
        // Transposed tensor: B_stride = 1 (row stride), but column stride should be >= K
        size_t B_col_stride = B->nb[1] / sizeof(float);
        #ifdef DEBUG
        char transposed_debug[256];
        int transposed_debug_len = snprintf(transposed_debug, sizeof(transposed_debug),
            "DEBUG: q_matmul_f32_avx2: Transposed B detected: B->nb[0]=%zu, B->nb[1]=%zu, B_col_stride=%zu, K=%u\n",
            B->nb[0], B->nb[1], B_col_stride, K);
        write(2, transposed_debug, (size_t)transposed_debug_len);
        #endif
        
        if (B_col_stride < K) {
            #ifdef DEBUG
            fprintf(stderr, "ERROR: q_matmul_f32_avx2: B_col_stride (%zu) < K (%u) for transposed tensor\n", B_col_stride, K);
            abort();
            #endif
            return Q_ERR_INVALID_SIZE;
        }
    } else {
        // Normal tensor: row stride should be >= N
        if (B_stride < N) {
            #ifdef DEBUG
            fprintf(stderr, "ERROR: q_matmul_f32_avx2: B_stride (%zu) < N (%u)\n", B_stride, N);
            abort();
            #endif
            return Q_ERR_INVALID_SIZE;
        }
    }
    if (C_stride < N) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: q_matmul_f32_avx2: C_stride (%zu) < N (%u)\n", C_stride, N);
        abort();
        #endif
        return Q_ERR_INVALID_SIZE;
    }
    
    // Transpose B for cache efficiency (allocate in arena) - ONLY if not already transposed
    // B_T will be [N, K] (transposed from B[K, N])
    // B_T is stored row-major: B_T[j, k] = B[k, j]
    float* B_T_data;
    
    if (is_transposed) {
        // B is already transposed, use it directly
        B_T_data = (float*)B_data;
        #ifdef DEBUG
        char transposed_skip_debug[128];
        int transposed_skip_len = snprintf(transposed_skip_debug, sizeof(transposed_skip_debug),
            "DEBUG: q_matmul_f32_avx2: B already transposed, skipping transpose\n");
        write(2, transposed_skip_debug, (size_t)transposed_skip_len);
        #endif
    } else {
        // B is not transposed, allocate and transpose
        #ifdef DEBUG
        char pre_transpose_debug[128];
        int pre_transpose_len = snprintf(pre_transpose_debug, sizeof(pre_transpose_debug),
            "DEBUG: q_matmul_f32_avx2: Before transpose, N=%u, K=%u\n", N, K);
        write(2, pre_transpose_debug, (size_t)pre_transpose_len);
        #endif
        
        size_t B_T_size = (size_t)N * (size_t)K * sizeof(float);
        B_T_data = (float*)q_arena_alloc(ctx, B_T_size);
        if (B_T_data == NULL) {
            #ifdef DEBUG
            fprintf(stderr, "ERROR: q_matmul_f32_avx2: Failed to allocate B_T buffer (%zu bytes)\n", B_T_size);
            abort();
            #endif
            return Q_ERR_ARENA_OOM;
        }
        
        #ifdef DEBUG
        char post_transpose_debug[128];
        int post_transpose_len = snprintf(post_transpose_debug, sizeof(post_transpose_debug),
            "DEBUG: q_matmul_f32_avx2: After transpose allocation, B_T_data=%p\n", (void*)B_T_data);
        write(2, post_transpose_debug, (size_t)post_transpose_len);
        #endif
        
        // Transpose B[K, N] to B_T[N, K]
        // B[k, j] -> B_T[j, k]
        // B_data[k * B_stride + j] -> B_T_data[j * K + k]
        for (uint32_t k = 0; k < K; k++) {
            for (uint32_t j = 0; j < N; j++) {
                B_T_data[j * K + k] = B_data[k * B_stride + j];
            }
        }
    }
    
    #ifdef DEBUG
    char pre_loop_debug[128];
    int pre_loop_len = snprintf(pre_loop_debug, sizeof(pre_loop_debug),
        "DEBUG: q_matmul_f32_avx2: Before MatMul loop, M=%u, N=%u, K=%u, MATMUL_BLOCK_SIZE=%u\n",
        M, N, K, MATMUL_BLOCK_SIZE);
    write(2, pre_loop_debug, (size_t)pre_loop_len);
    #endif
    
    // Cache-blocked matrix multiplication: C = A @ B
    // Using B_T for cache-friendly access: C[i,j] = sum_k(A[i,k] * B_T[j,k])
    for (uint32_t i = 0; i < M; i += MATMUL_BLOCK_SIZE) {
        uint32_t i_limit = (i + MATMUL_BLOCK_SIZE < M) ? 
                           i + MATMUL_BLOCK_SIZE : M;
        
        for (uint32_t j = 0; j < N; j += MATMUL_BLOCK_SIZE) {
            uint32_t j_limit = (j + MATMUL_BLOCK_SIZE < N) ? 
                               j + MATMUL_BLOCK_SIZE : N;
            
            // Compute block C[i:i_limit, j:j_limit] = A[i:i_limit, :] @ B[:, j:j_limit]
            for (uint32_t ii = i; ii < i_limit; ii++) {
                const float* A_row = A_data + (size_t)ii * A_stride;
                
                #ifdef DEBUG
                // DEBUG: Check alignment of A_row
                if (ii == i && j == 0) {
                    char align_debug[256];
                    int align_debug_len = snprintf(align_debug, sizeof(align_debug),
                        "DEBUG: q_matmul_f32_avx2: A_row alignment: %p %% 32 = %zu\n",
                        (void*)A_row, (uintptr_t)A_row % 32);
                    write(2, align_debug, (size_t)align_debug_len);
                }
                #endif
                
                for (uint32_t jj = j; jj < j_limit; jj++) {
                    const float* B_T_col = B_T_data + (size_t)jj * (size_t)K;
                    
                    #ifdef DEBUG
                    // DEBUG: Check alignment of B_T_col
                    if (ii == i && jj == j) {
                        char align_debug[256];
                        int align_debug_len = snprintf(align_debug, sizeof(align_debug),
                            "DEBUG: q_matmul_f32_avx2: B_T_col alignment: %p %% 32 = %zu, K=%u\n",
                            (void*)B_T_col, (uintptr_t)B_T_col % 32, K);
                        write(2, align_debug, (size_t)align_debug_len);
                    }
                    #endif
                    
                    // Initialize accumulator
                    float dot_product = 0.0f;
                    
                    // Main loop: 4x unrolling (32 elements per iteration)
                    uint32_t k_vec = K & ~31U;
                    
                    #ifdef DEBUG
                    // DEBUG: Print k_vec calculation
                    if (ii == i && jj == j) {
                        char kvec_debug[128];
                        int kvec_debug_len = snprintf(kvec_debug, sizeof(kvec_debug),
                            "DEBUG: q_matmul_f32_avx2: k_vec = %u (K=%u, K&~31U=%u)\n",
                            k_vec, K, K & ~31U);
                        write(2, kvec_debug, (size_t)kvec_debug_len);
                    }
                    #endif
                    
                    if (k_vec > 0) {
                        // Initialize 4 accumulators (4x unrolling)
                        __m256 acc0 = _mm256_setzero_ps();
                        __m256 acc1 = _mm256_setzero_ps();
                        __m256 acc2 = _mm256_setzero_ps();
                        __m256 acc3 = _mm256_setzero_ps();
                        
                        for (uint32_t k = 0; k < k_vec; k += 32) {
                            #ifdef DEBUG
                            // DEBUG: Check alignment before loads (first iteration only)
                            if (k == 0 && ii == i && jj == j) {
                                char load_debug[256];
                                int load_debug_len = snprintf(load_debug, sizeof(load_debug),
                                    "DEBUG: q_matmul_f32_avx2: Before loads, k=%u, A_row+k=%p (align=%zu), B_T_col+k=%p (align=%zu)\n",
                                    k, (void*)(A_row + k), ((uintptr_t)(A_row + k) % 32),
                                    (void*)(B_T_col + k), ((uintptr_t)(B_T_col + k) % 32));
                                write(2, load_debug, (size_t)load_debug_len);
                            }
                            #endif
                            
                            // Prefetch next iteration
                            if (k + PREFETCH_DISTANCE < K) {
                                _mm_prefetch((const char*)(A_row + k + PREFETCH_DISTANCE), _MM_HINT_T0);
                                _mm_prefetch((const char*)(B_T_col + k + PREFETCH_DISTANCE), _MM_HINT_T0);
                            }
                            
                            // CRITICAL FIX: Use aligned or unaligned loads based on stride
                            // If stride is multiple of 32 bytes, use aligned loads (faster)
                            // Otherwise, use unaligned loads (works but slightly slower)
                            __m256 a0, a1, a2, a3;
                            __m256 b0, b1, b2, b3;
                            
                            if (!A_needs_unaligned) {
                                // Aligned loads for A (faster)
                                a0 = _mm256_load_ps(A_row + k + 0);
                                a1 = _mm256_load_ps(A_row + k + 8);
                                a2 = _mm256_load_ps(A_row + k + 16);
                                a3 = _mm256_load_ps(A_row + k + 24);
                            } else {
                                // Unaligned loads for A (works for transposed tensors)
                                a0 = _mm256_loadu_ps(A_row + k + 0);
                                a1 = _mm256_loadu_ps(A_row + k + 8);
                                a2 = _mm256_loadu_ps(A_row + k + 16);
                                a3 = _mm256_loadu_ps(A_row + k + 24);
                            }
                            
                            if (!B_needs_unaligned) {
                                // Aligned loads for B (faster)
                                b0 = _mm256_load_ps(B_T_col + k + 0);
                                b1 = _mm256_load_ps(B_T_col + k + 8);
                                b2 = _mm256_load_ps(B_T_col + k + 16);
                                b3 = _mm256_load_ps(B_T_col + k + 24);
                            } else {
                                // Unaligned loads for B (works for transposed tensors)
                                b0 = _mm256_loadu_ps(B_T_col + k + 0);
                                b1 = _mm256_loadu_ps(B_T_col + k + 8);
                                b2 = _mm256_loadu_ps(B_T_col + k + 16);
                                b3 = _mm256_loadu_ps(B_T_col + k + 24);
                            }
                            
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
                        dot_product = hsum256_ps(sum);
                        
                        #ifdef DEBUG
                        // DEBUG: Print after AVX2 loop
                        if (ii == i && jj == j) {
                            char post_avx_debug[128];
                            int post_avx_len = snprintf(post_avx_debug, sizeof(post_avx_debug),
                                "DEBUG: q_matmul_f32_avx2: After AVX2 loop, dot_product=%f, k_vec=%u, K=%u\n",
                                dot_product, k_vec, K);
                            write(2, post_avx_debug, (size_t)post_avx_len);
                        }
                        #endif
                    }
                    
                    // Tail handling (scalar fallback for remainder)
                    #ifdef DEBUG
                    // DEBUG: Print tail loop info
                    if (ii == i && jj == j && k_vec < K) {
                        char tail_debug[128];
                        int tail_len = snprintf(tail_debug, sizeof(tail_debug),
                            "DEBUG: q_matmul_f32_avx2: Tail loop, k_vec=%u, K=%u, remaining=%u\n",
                            k_vec, K, K - k_vec);
                        write(2, tail_debug, (size_t)tail_len);
                    }
                    #endif
                    
                    for (uint32_t k = k_vec; k < K; k++) {
                        dot_product += A_row[k] * B_T_col[k];
                    }
                    
                    #ifdef DEBUG
                    // DEBUG: Print before store
                    if (ii == i && jj == j) {
                        char store_debug[128];
                        int store_len = snprintf(store_debug, sizeof(store_debug),
                            "DEBUG: q_matmul_f32_avx2: Before store, dot_product=%f, C_stride=%zu, jj=%u\n",
                            dot_product, C_stride, jj);
                        write(2, store_debug, (size_t)store_len);
                    }
                    #endif
                    
                    // Store result
                    C_data[(size_t)ii * C_stride + jj] = dot_product;
                }
            }
        }
    }
    
    return Q_OK;
}

