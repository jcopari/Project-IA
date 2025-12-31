#include "qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// Reference implementation (scalar, slow but correct)
// Used to validate the AVX2 implementation
static void gemv_q4_f32_ref(
    const q_tensor* restrict weights,
    const float* restrict input,
    float* restrict output
) {
    const uint32_t M = weights->ne[0];
    const uint32_t N = weights->ne[1];
    const q_block_q4_0* restrict blocks = (const q_block_q4_0* restrict)weights->data;
    const uint32_t blocks_per_row = N / 32;
    
    for (uint32_t i = 0; i < M; i++) {
        float sum = 0.0f;
        const q_block_q4_0* restrict row_blocks = blocks + (i * blocks_per_row);
        
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            const q_block_q4_0* restrict blk = &row_blocks[b];
            const float scale = blk->scale;
            
            // Dequantize and compute dot product
            for (uint32_t j = 0; j < 32; j++) {
                // Extract nibble
                uint8_t byte_idx = j / 2;
                uint8_t nibble_idx = j % 2;
                uint8_t byte = blk->qs[byte_idx];
                uint8_t nibble = (nibble_idx == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                
                // Dequantize: value = (quantized - 8) * scale
                float weight = ((float)nibble - 8.0f) * scale;
                
                // Accumulate dot product
                sum += weight * input[b * 32 + j];
            }
        }
        
        output[i] = sum;
    }
}

// Generate random Q4_0 matrix
static void generate_q4_matrix(
    q_tensor* tensor,
    void* data_buffer,
    uint32_t M,
    uint32_t N
) {
    // Validate N is multiple of 32
    if (N % 32 != 0) {
        fprintf(stderr, "ERROR: N must be multiple of 32\n");
        abort();
    }
    
    const uint32_t blocks_per_row = N / 32;
    q_block_q4_0* blocks = (q_block_q4_0*)data_buffer;
    
    // Initialize tensor
    tensor->data = blocks;
    tensor->scales = NULL;
    tensor->ne[0] = M;
    tensor->ne[1] = N;
    tensor->ne[2] = 1;
    tensor->ne[3] = 1;
    tensor->nb[0] = blocks_per_row * sizeof(q_block_q4_0);
    tensor->nb[1] = sizeof(q_block_q4_0);
    tensor->nb[2] = 0;
    tensor->nb[3] = 0;
    tensor->type = Q_Q4_0;
    strncpy(tensor->name, "test_weights", sizeof(tensor->name) - 1);
    
    // Generate random blocks
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            q_block_q4_0* blk = &blocks[i * blocks_per_row + b];
            
            // Random scale (0.01 to 1.0)
            blk->scale = 0.01f + ((float)rand() / RAND_MAX) * 0.99f;
            
            // Generate random quantized values (0-15)
            for (uint32_t j = 0; j < 16; j++) {
                uint8_t low_nibble = rand() % 16;
                uint8_t high_nibble = rand() % 16;
                blk->qs[j] = (high_nibble << 4) | low_nibble;
            }
        }
    }
}

// Generate random input vector
static void generate_input_vector(float* input, uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        // Random values in [-1.0, 1.0]
        input[i] = -1.0f + ((float)rand() / RAND_MAX) * 2.0f;
    }
}

// Compare results with tolerance
static int compare_results(
    const float* ref,
    const float* test,
    uint32_t M,
    float abs_tol,
    float rel_tol
) {
    int errors = 0;
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    
    for (uint32_t i = 0; i < M; i++) {
        float abs_err = fabsf(ref[i] - test[i]);
        float rel_err = (fabsf(ref[i]) > 1e-8f) ? abs_err / fabsf(ref[i]) : abs_err;
        
        if (abs_err > max_abs_error) max_abs_error = abs_err;
        if (rel_err > max_rel_error) max_rel_error = rel_err;
        
        // Hybrid tolerance: pass if EITHER absolute OR relative error is within tolerance
        // This handles cases where values are very small (high rel_err but low abs_err)
        // or very large (high abs_err but low rel_err)
        if (abs_err > abs_tol && rel_err > rel_tol) {
            if (errors < 10) {  // Print first 10 errors
                printf("  Error at [%u]: ref=%.6f, test=%.6f, abs_err=%.6e, rel_err=%.6e\n",
                       i, ref[i], test[i], abs_err, rel_err);
            }
            errors++;
        }
    }
    
    printf("  Max absolute error: %.6e (tolerance: %.6e)\n", max_abs_error, abs_tol);
    printf("  Max relative error: %.6e (tolerance: %.6e)\n", max_rel_error, rel_tol);
    
    return errors;
}

// Test case runner
static int run_test_case(uint32_t M, uint32_t N, int test_num) {
    printf("\n=== Test Case %d ===\n", test_num);
    printf("  Matrix: [%u, %u]\n", M, N);
    
    // Validate N is multiple of 32
    if (N % 32 != 0) {
        printf("  ✗ SKIPPED: N (%u) must be multiple of 32\n", N);
        return 0;
    }
    
    // Allocate buffers
    const uint32_t blocks_per_row = N / 32;
    const size_t blocks_size = M * blocks_per_row * sizeof(q_block_q4_0);
    void* blocks_buffer = aligned_alloc(Q_ALIGN, blocks_size);
    
    const size_t input_size = N * sizeof(float);
    float* input = (float*)aligned_alloc(Q_ALIGN, input_size);
    
    const size_t output_size = M * sizeof(float);
    float* output_ref = (float*)aligned_alloc(Q_ALIGN, output_size);
    float* output_test = (float*)aligned_alloc(Q_ALIGN, output_size);
    
    if (!blocks_buffer || !input || !output_ref || !output_test) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        abort();
    }
    
    // Initialize tensor
    q_tensor weights;
    generate_q4_matrix(&weights, blocks_buffer, M, N);
    
    // Generate random input
    generate_input_vector(input, N);
    
    // Run reference implementation
    memset(output_ref, 0, output_size);
    gemv_q4_f32_ref(&weights, input, output_ref);
    
    // Run AVX2 implementation
    memset(output_test, 0, output_size);
    q_error_code err = q_gemv_q4_f32_avx2(&weights, input, output_test);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: q_gemv_q4_f32_avx2 failed: %s\n", q_strerror(err));
        return 1;
    }
    
    // Compare results
    // Use Q4_0 tolerances (more relaxed) since we're comparing quantized operations
    // Small numerical differences are expected due to FMA vs scalar precision
    // FMA operations can accumulate small rounding differences over many operations
    const float abs_tol = 1.5e-4f;  // Relaxed for quantized MatMul (FMA vs scalar)
    const float rel_tol = Q_EPSILON_REL_F32;
    int errors = compare_results(output_ref, output_test, M, abs_tol, rel_tol);
    
    if (errors == 0) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED: %d / %u errors\n", errors, M);
    }
    
    // Cleanup
    free(blocks_buffer);
    free(input);
    free(output_ref);
    free(output_test);
    
    return errors;
}

int main(void) {
    printf("=== Q4_F32 MatMul Test Suite ===\n");
    printf("Validating AVX2 implementation against scalar reference\n");
    
    // Seed RNG for reproducibility
    srand(42);
    
    int total_errors = 0;
    
    // Test Case 1: Small matrix (1 row, 32 cols)
    total_errors += run_test_case(1, 32, 1);
    
    // Test Case 2: Small matrix (4 rows, 64 cols)
    total_errors += run_test_case(4, 64, 2);
    
    // Test Case 3: Medium matrix (16 rows, 128 cols)
    total_errors += run_test_case(16, 128, 3);
    
    // Test Case 4: Larger matrix (64 rows, 256 cols)
    total_errors += run_test_case(64, 256, 4);
    
    // Test Case 5: Typical attention size (32 rows, 512 cols)
    total_errors += run_test_case(32, 512, 5);
    
    // Test Case 6: Large matrix (128 rows, 1024 cols)
    total_errors += run_test_case(128, 1024, 6);
    
    printf("\n=== Test Summary ===\n");
    if (total_errors == 0) {
        printf("✓ All tests PASSED\n");
        return 0;
    } else {
        printf("✗ FAILED: %d total errors\n", total_errors);
        return 1;
    }
}

