#include "qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <time.h>

#define MAX_DIFF 1e-5f  // Maximum allowed difference (FP32 precision)
#define MASK_VALUE -1e9f  // Typical mask value for attention

// Reference implementation (scalar, slow but correct)
static void causal_mask_f32_ref(
    float* restrict scores,
    float mask_value,
    uint32_t seq_len
) {
    for (uint32_t i = 0; i < seq_len; i++) {
        for (uint32_t j = i + 1; j < seq_len; j++) {
            scores[i * seq_len + j] = mask_value;
        }
    }
}

// Generate random FP32 matrix
static void generate_f32_matrix(float* matrix, uint32_t rows, uint32_t cols) {
    for (uint32_t i = 0; i < rows * cols; i++) {
        matrix[i] = -1.0f + ((float)rand() / RAND_MAX) * 2.0f; // Values in [-1.0, 1.0]
    }
}

// Compare results with tolerance
static int compare_results(
    const float* ref,
    const float* test,
    uint32_t seq_len,
    float abs_tol,
    float rel_tol
) {
    int errors = 0;
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    
    for (uint32_t i = 0; i < seq_len; i++) {
        for (uint32_t j = 0; j < seq_len; j++) {
            size_t idx = (size_t)i * seq_len + j;
            float abs_err = fabsf(ref[idx] - test[idx]);
            float rel_err = (fabsf(ref[idx]) > 1e-8f) ? abs_err / fabsf(ref[idx]) : abs_err;
            
            if (abs_err > max_abs_error) max_abs_error = abs_err;
            if (rel_err > max_rel_error) max_rel_error = rel_err;
            
            if (abs_err > abs_tol && rel_err > rel_tol) {
                if (errors < 10) { // Print first 10 errors
                    printf("  Error at [%u, %u]: ref=%.6f, test=%.6f, abs_err=%.6e, rel_err=%.6e\n",
                           i, j, ref[idx], test[idx], abs_err, rel_err);
                }
                errors++;
            }
        }
    }
    
    printf("  Max absolute error: %.6e (tolerance: %.6e)\n", max_abs_error, abs_tol);
    printf("  Max relative error: %.6e (tolerance: %.6e)\n", max_rel_error, rel_tol);
    
    return errors;
}

// Test case runner
static int run_test_case(uint32_t seq_len, int test_num, float mask_value) {
    printf("\n=== Test Case %d ===\n", test_num);
    printf("  Matrix: [%u, %u], mask_value = %.1e\n", seq_len, seq_len, mask_value);
    
    size_t matrix_size = (size_t)seq_len * seq_len * sizeof(float);
    size_t aligned_size = (matrix_size + Q_ALIGN - 1) & ~(Q_ALIGN - 1);  // Align to Q_ALIGN
    
    float* scores_data = (float*)aligned_alloc(Q_ALIGN, aligned_size);
    float* ref_data = (float*)aligned_alloc(Q_ALIGN, aligned_size);
    float* test_data = (float*)aligned_alloc(Q_ALIGN, aligned_size);
    
    assert(scores_data && ref_data && test_data);
    
    // Generate random matrix
    generate_f32_matrix(scores_data, seq_len, seq_len);
    
    // Copy to reference and test
    memcpy(ref_data, scores_data, matrix_size);
    memcpy(test_data, scores_data, matrix_size);
    
    q_tensor scores_tensor = {0};
    q_error_code err;

    scores_tensor.data = test_data;
    scores_tensor.ne[0] = seq_len;
    scores_tensor.ne[1] = seq_len;
    scores_tensor.ne[2] = 1;
    scores_tensor.ne[3] = 1;
    scores_tensor.nb[0] = seq_len * sizeof(float);
    scores_tensor.nb[1] = sizeof(float);
    scores_tensor.nb[2] = 0;
    scores_tensor.nb[3] = 0;
    scores_tensor.type = Q_F32;
    strncpy(scores_tensor.name, "scores", sizeof(scores_tensor.name) - 1);
    
    // Compute reference (scalar)
    causal_mask_f32_ref(ref_data, mask_value, seq_len);
    
    // Execute AVX2 Causal Mask
    err = q_causal_mask_f32_avx2(&scores_tensor, mask_value);
    if (err != Q_OK) {
        printf("  ✗ FAILED: q_causal_mask_f32_avx2 returned error: %s\n", q_strerror(err));
        free(scores_data); free(ref_data); free(test_data);
        return 1;
    }
    
    // Compare results
    int error_count = compare_results(ref_data, test_data, seq_len, 
                                      Q_EPSILON_ABS_F32, Q_EPSILON_REL_F32);
    
    free(scores_data); free(ref_data); free(test_data);

    if (error_count == 0) {
        printf("  ✓ PASSED\n");
        return 0;
    } else {
        printf("  ✗ FAILED: %d elements out of tolerance\n", error_count);
        return 1;
    }
}

int main(void) {
    srand(time(NULL)); // Seed random for reproducibility
    printf("========================================\n");
    printf("Causal Mask FP32 AVX2 Test Suite\n");
    printf("========================================\n");

    int total_failures = 0;

    // Test Cases (seq_len, test_num, mask_value)
    total_failures += run_test_case(1, 1, MASK_VALUE);   // Single element (no masking)
    total_failures += run_test_case(4, 2, MASK_VALUE);  // Small matrix
    total_failures += run_test_case(8, 3, MASK_VALUE); // Medium matrix (aligned)
    total_failures += run_test_case(15, 4, MASK_VALUE); // Non-multiple of 8
    total_failures += run_test_case(32, 5, MASK_VALUE); // Large matrix (aligned)
    total_failures += run_test_case(64, 6, MASK_VALUE); // Very large matrix
    total_failures += run_test_case(7, 7, MASK_VALUE);  // Edge case: 7 (just below 8)
    total_failures += run_test_case(9, 8, MASK_VALUE);  // Edge case: 9 (just above 8)
    
    // Test with different mask values
    total_failures += run_test_case(8, 9, 0.0f);       // Mask value = 0.0
    total_failures += run_test_case(8, 10, -FLT_MAX);   // Mask value = -inf

    // Error Handling Tests
    printf("\n=== Error Handling Tests ===\n");
    q_tensor scores_err = {0};
    scores_err.type = Q_F32;
    scores_err.ne[0] = 4;
    scores_err.ne[1] = 4;
    size_t err_size = (4*4*sizeof(float) + Q_ALIGN - 1) & ~(Q_ALIGN - 1);
    scores_err.data = (float*)aligned_alloc(Q_ALIGN, err_size);
    assert(scores_err.data);

    // NULL scores
    q_error_code err = q_causal_mask_f32_avx2(NULL, MASK_VALUE);
    assert(err == Q_ERR_INVALID_ARG);
    printf("  ✓ PASSED: NULL scores validation\n");

    // Non-square matrix
    q_tensor non_square = {0};
    non_square.type = Q_F32;
    non_square.ne[0] = 4;
    non_square.ne[1] = 8;  // Not square!
    size_t non_square_size = (4*8*sizeof(float) + Q_ALIGN - 1) & ~(Q_ALIGN - 1);
    non_square.data = (float*)aligned_alloc(Q_ALIGN, non_square_size);
    assert(non_square.data);
    err = q_causal_mask_f32_avx2(&non_square, MASK_VALUE);
    assert(err == Q_ERR_INVALID_SIZE);
    printf("  ✓ PASSED: Non-square matrix validation\n");
    free(non_square.data);

    // Zero seq_len
    q_tensor zero_size = {0};
    zero_size.type = Q_F32;
    zero_size.ne[0] = 0;
    zero_size.ne[1] = 0;
    zero_size.data = (float*)aligned_alloc(Q_ALIGN, Q_ALIGN); // Still allocate some memory (aligned)
    assert(zero_size.data);
    err = q_causal_mask_f32_avx2(&zero_size, MASK_VALUE);
    assert(err == Q_ERR_INVALID_SIZE);
    printf("  ✓ PASSED: Zero seq_len validation\n");
    free(zero_size.data);

    free(scores_err.data);

    if (total_failures == 0) {
        printf("\n✓ All Causal Mask FP32 tests PASSED\n");
        return 0;
    } else {
        printf("\n✗ %d Causal Mask FP32 test case(s) FAILED\n", total_failures);
        return 1;
    }
}

