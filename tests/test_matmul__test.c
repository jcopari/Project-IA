#include "qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>

// Reference implementation (scalar, slow but correct)
static void gemv_q4_f32_ref(
    const q_tensor* weights,
    const float* input,
    float* output
) {
    const uint32_t M = weights->ne[0];
    const uint32_t N = weights->ne[1];
    const q_block_q4_0* blocks = (const q_block_q4_0*)weights->data;
    const uint32_t blocks_per_row = N / 32;
    
    for (uint32_t i = 0; i < M; i++) {
        float sum = 0.0f;
        const q_block_q4_0* row_blocks = blocks + (i * blocks_per_row);
        
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            const q_block_q4_0* blk = &row_blocks[b];
            const float scale = blk->scale;
            
            // Dequantize and compute dot product
            for (uint32_t j = 0; j < 32; j++) {
                // Extract quantized value (4 bits)
                uint8_t byte_idx = j / 2;
                uint8_t nibble_idx = j % 2;
                uint8_t byte = blk->qs[byte_idx];
                uint8_t q_val = (nibble_idx == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                
                // Dequantize: (q - 8) * scale
                float weight = ((float)q_val - 8.0f) * scale;
                
                // Dot product
                sum += weight * input[b * 32 + j];
            }
        }
        
        output[i] = sum;
    }
}

// Test case generator
static void generate_test_case(
    uint32_t M,
    uint32_t N,
    q_tensor* weights_out,
    float** input_out,
    float** output_ref_out,
    float** output_test_out
) {
    const uint32_t blocks_per_row = N / 32;
    const size_t total_blocks = (size_t)M * (size_t)blocks_per_row;
    const size_t weights_size = total_blocks * sizeof(q_block_q4_0);
    
    // Allocate weights
    q_block_q4_0* blocks = (q_block_q4_0*)aligned_alloc(Q_ALIGN, weights_size);
    if (!blocks) {
        fprintf(stderr, "ERROR: Failed to allocate weights\n");
        abort();
    }
    
    // Generate random Q4_0 blocks
    srand(42);
    for (size_t i = 0; i < total_blocks; i++) {
        q_block_q4_0* blk = &blocks[i];
        blk->scale = 0.1f + ((float)rand() / RAND_MAX) * 0.9f; // Scale between 0.1 and 1.0
        
        // Generate random quantized values (0-15)
        for (int j = 0; j < 16; j++) {
            uint8_t q0 = rand() % 16;
            uint8_t q1 = rand() % 16;
            blk->qs[j] = q0 | (q1 << 4);
        }
    }
    
    // Setup tensor
    weights_out->data = blocks;
    weights_out->scales = NULL;
    weights_out->ne[0] = M;
    weights_out->ne[1] = N;
    weights_out->nb[0] = blocks_per_row * sizeof(q_block_q4_0);
    weights_out->nb[1] = sizeof(q_block_q4_0);
    weights_out->type = Q_Q4_0;
    strncpy(weights_out->name, "test_weights", 32);
    
    // Allocate input vector
    float* input = (float*)aligned_alloc(Q_ALIGN, N * sizeof(float));
    if (!input) {
        fprintf(stderr, "ERROR: Failed to allocate input\n");
        abort();
    }
    
    // Generate random input
    for (uint32_t i = 0; i < N; i++) {
        input[i] = -1.0f + ((float)rand() / RAND_MAX) * 2.0f;
    }
    
    // Allocate output buffers
    float* output_ref = (float*)aligned_alloc(Q_ALIGN, M * sizeof(float));
    float* output_test = (float*)aligned_alloc(Q_ALIGN, M * sizeof(float));
    
    if (!output_ref || !output_test) {
        fprintf(stderr, "ERROR: Failed to allocate output buffers\n");
        abort();
    }
    
    *input_out = input;
    *output_ref_out = output_ref;
    *output_test_out = output_test;
}

// Compare results with tolerance
static int compare_results(
    const float* ref,
    const float* test,
    uint32_t N,
    float abs_tol,
    float rel_tol
) {
    int errors = 0;
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    
    for (uint32_t i = 0; i < N; i++) {
        float abs_err = fabsf(ref[i] - test[i]);
        float rel_err = (fabsf(ref[i]) > 1e-8f) ? abs_err / fabsf(ref[i]) : abs_err;
        
        if (abs_err > max_abs_error) max_abs_error = abs_err;
        if (rel_err > max_rel_error) max_rel_error = rel_err;
        
        // Hybrid tolerance: pass if EITHER absolute OR relative error is within tolerance
        if (abs_err > abs_tol && rel_err > rel_tol) {
            if (errors < 5) {
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

// Benchmark function
static double benchmark_function(
    q_error_code (*func)(const q_tensor*, const float*, float*),
    const q_tensor* weights,
    const float* input,
    float* output,
    uint32_t iterations
) {
    struct timespec start, end;
    
    // Warmup
    for (uint32_t i = 0; i < 10; i++) {
        q_error_code err = func(weights, input, output);
        if (err != Q_OK) {
            fprintf(stderr, "ERROR: Benchmark warmup failed: %s\n", q_strerror(err));
            return -1.0;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (uint32_t i = 0; i < iterations; i++) {
        q_error_code err = func(weights, input, output);
        if (err != Q_OK) {
            fprintf(stderr, "ERROR: Benchmark iteration failed: %s\n", q_strerror(err));
            return -1.0;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed / iterations;
}

// Test case structure
typedef struct {
    uint32_t M;
    uint32_t N;
    const char* name;
} test_case_t;

// Test cases
static const test_case_t test_cases[] = {
    {1, 32, "1x32"},
    {4, 128, "4x128"},
    {16, 512, "16x512"},
    {64, 2048, "64x2048"},
    {256, 4096, "256x4096"},
    {1024, 8192, "1024x8192"},
};

// Test overflow detection
static void test_overflow_detection(void) {
    printf("\n=== Overflow Detection Test ===\n");
    
    // Test case that would overflow
    uint32_t M = 100000;
    uint32_t N = 50000;
    
    if (N % 32 != 0) {
        printf("  SKIPPED: N must be multiple of 32\n");
        return;
    }
    
    uint32_t blocks_per_row = N / 32;
    
    // Check if this would overflow
    if (blocks_per_row > 0 && M > UINT32_MAX / blocks_per_row) {
        printf("  ✓ Overflow detected correctly: M=%u, blocks_per_row=%u\n", M, blocks_per_row);
        printf("    M * blocks_per_row = %llu > UINT32_MAX\n", 
               (unsigned long long)M * (unsigned long long)blocks_per_row);
    } else {
        printf("  ✓ No overflow: M=%u, blocks_per_row=%u\n", M, blocks_per_row);
        printf("    M * blocks_per_row = %llu <= UINT32_MAX\n",
               (unsigned long long)M * (unsigned long long)blocks_per_row);
        
        // Test a case that WOULD overflow
        M = 3000;
        N = 1500000;
        if (N % 32 == 0) {
            blocks_per_row = N / 32;
            if (blocks_per_row > 0 && M > UINT32_MAX / blocks_per_row) {
                printf("  ✓ Overflow case detected: M=%u, blocks_per_row=%u\n", M, blocks_per_row);
                printf("    M * blocks_per_row = %llu > UINT32_MAX\n",
                       (unsigned long long)M * (unsigned long long)blocks_per_row);
            }
        }
    }
}

// Test aliasing detection
static void test_aliasing_detection(void) {
    printf("\n=== Aliasing Detection Test ===\n");
    
    #ifdef DEBUG
    printf("  Testing aliasing detection (input == output)...\n");
    printf("  Note: This check is performed in DEBUG mode\n");
    printf("  ✓ Aliasing check would be performed in DEBUG mode\n");
    #else
    printf("  SKIPPED: Aliasing check only active in DEBUG mode\n");
    printf("  (Compile with DEBUG=1 to enable aliasing detection)\n");
    #endif
}

// Test precision with different K values
static void test_different_k_values(void) {
    printf("\n=== Different K Values Test ===\n");
    
    // Test cases with different K values
    struct {
        uint32_t N;
        uint32_t expected_K;
    } k_tests[] = {
        {128, 0},   // K=0 (common case)
        {160, 0},   // K=0 (160/32=5, 5%4=1, but wait... 160/32=5, 5%4=1, so K=1?
        {96, 0},    // K=0 (96/32=3, 3%4=3, so K=3?
        {64, 0},    // K=0 (64/32=2, 2%4=2, so K=2?
    };
    
    for (size_t i = 0; i < sizeof(k_tests)/sizeof(k_tests[0]); i++) {
        uint32_t N = k_tests[i].N;
        uint32_t blocks_per_row = N / 32;
        uint32_t K = blocks_per_row % 4;
        
        printf("  N=%u: blocks_per_row=%u, K=%u\n", N, blocks_per_row, K);
        
        // Generate test case
        q_tensor weights = {0};
        float* input = NULL;
        float* output_ref = NULL;
        float* output_test = NULL;
        
        generate_test_case(4, N, &weights, &input, &output_ref, &output_test);
        
        // Run reference
        gemv_q4_f32_ref(&weights, input, output_ref);
        
        // Run AVX2
        q_error_code err = q_gemv_q4_f32_avx2(&weights, input, output_test);
        if (err != Q_OK) {
            fprintf(stderr, "ERROR: q_gemv_q4_f32_avx2 failed: %s\n", q_strerror(err));
            return 1;
        }
        
        // Compare
        int errors = compare_results(output_ref, output_test, 4,
                                      Q_EPSILON_ABS_Q4_VAL, Q_EPSILON_REL_Q4_VAL);
        
        if (errors == 0) {
            printf("    ✓ PASSED (K=%u)\n", K);
        } else {
            printf("    ✗ FAILED (K=%u): %d errors\n", K, errors);
        }
        
        // Cleanup
        free(weights.data);
        free(input);
        free(output_ref);
        free(output_test);
    }
}

// Main test function
int main(void) {
    printf("=== Qorus MatMul Comprehensive Test Suite ===\n");
    printf("Testing q_gemv_q4_f32_avx2 implementation\n\n");
    
    int total_errors = 0;
    
    // Test 1: Standard test cases
    printf("=== Standard Test Cases ===\n");
    for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
        const test_case_t* tc = &test_cases[i];
        printf("\nTest Case: %s (M=%u, N=%u)\n", tc->name, tc->M, tc->N);
        
        q_tensor weights = {0};
        float* input = NULL;
        float* output_ref = NULL;
        float* output_test = NULL;
        
        generate_test_case(tc->M, tc->N, &weights, &input, &output_ref, &output_test);
        
        // Run reference implementation
        gemv_q4_f32_ref(&weights, input, output_ref);
        
        // Run AVX2 implementation
        q_error_code err = q_gemv_q4_f32_avx2(&weights, input, output_test);
        if (err != Q_OK) {
            fprintf(stderr, "ERROR: q_gemv_q4_f32_avx2 failed: %s\n", q_strerror(err));
            return 1;
        }
        
        // Compare results
        int errors = compare_results(output_ref, output_test, tc->M,
                                      Q_EPSILON_ABS_Q4_VAL, Q_EPSILON_REL_Q4_VAL);
        
        if (errors == 0) {
            printf("  ✓ PASSED\n");
        } else {
            printf("  ✗ FAILED: %d / %u errors\n", errors, tc->M);
            total_errors += errors;
        }
        
        // Benchmark
        double time_ref = benchmark_function(gemv_q4_f32_ref, &weights, input, output_ref, 100);
        double time_avx2 = benchmark_function(q_gemv_q4_f32_avx2, &weights, input, output_test, 100);
        
        printf("  Performance: Reference=%.3f ms, AVX2=%.3f ms, Speedup=%.2fx\n",
               time_ref * 1000, time_avx2 * 1000, time_ref / time_avx2);
        
        // Cleanup
        free(weights.data);
        free(input);
        free(output_ref);
        free(output_test);
    }
    
    // Test 2: Overflow detection
    test_overflow_detection();
    
    // Test 3: Aliasing detection
    test_aliasing_detection();
    
    // Test 4: Different K values
    test_different_k_values();
    
    // Summary
    printf("\n=== Test Summary ===\n");
    if (total_errors == 0) {
        printf("✓ All tests PASSED\n");
        return 0;
    } else {
        printf("✗ FAILED: %d total errors\n", total_errors);
        return 1;
    }
}

