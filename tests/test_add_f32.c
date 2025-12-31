#include "qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#define MAX_DIFF 1e-5f  // Maximum allowed difference (FP32 precision)

// Reference implementation (scalar, slow but correct)
static void add_f32_ref(
    const float* restrict a,
    const float* restrict b,
    float* restrict output,
    uint32_t N
) {
    for (uint32_t i = 0; i < N; i++) {
        output[i] = a[i] + b[i];
    }
}

// Generate random FP32 vector
static void generate_vector(float* vector, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        vector[i] = -1.0f + ((float)rand() / RAND_MAX) * 2.0f; // Values in [-1.0, 1.0]
    }
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
        
        if (abs_err > abs_tol && rel_err > rel_tol) {
            if (errors < 10) { // Print first 10 errors
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
static int run_test_case(uint32_t N, int test_num) {
    printf("\n=== Test Case %d ===\n", test_num);
    printf("  Vector size: %u\n", N);
    
    size_t vector_size = (size_t)N * sizeof(float);
    size_t aligned_size = (vector_size + Q_ALIGN - 1) & ~(Q_ALIGN - 1);
    
    float* a_data = (float*)aligned_alloc(Q_ALIGN, aligned_size);
    float* b_data = (float*)aligned_alloc(Q_ALIGN, aligned_size);
    float* ref_data = (float*)aligned_alloc(Q_ALIGN, aligned_size);
    float* test_data = (float*)aligned_alloc(Q_ALIGN, aligned_size);
    
    assert(a_data && b_data && ref_data && test_data);
    
    // Generate random vectors
    generate_vector(a_data, N);
    generate_vector(b_data, N);
    
    // Initialize output to zero
    memset(ref_data, 0, aligned_size);
    memset(test_data, 0, aligned_size);
    
    q_tensor a_tensor = {0};
    q_tensor b_tensor = {0};
    q_tensor output_tensor = {0};
    q_error_code err;

    a_tensor.data = a_data;
    a_tensor.ne[0] = N;
    a_tensor.ne[1] = 1;
    a_tensor.ne[2] = 1;
    a_tensor.ne[3] = 1;
    a_tensor.nb[0] = N * sizeof(float);
    a_tensor.nb[1] = sizeof(float);
    a_tensor.nb[2] = 0;
    a_tensor.nb[3] = 0;
    a_tensor.type = Q_F32;
    strncpy(a_tensor.name, "a", sizeof(a_tensor.name) - 1);
    
    b_tensor.data = b_data;
    b_tensor.ne[0] = N;
    b_tensor.ne[1] = 1;
    b_tensor.ne[2] = 1;
    b_tensor.ne[3] = 1;
    b_tensor.nb[0] = N * sizeof(float);
    b_tensor.nb[1] = sizeof(float);
    b_tensor.nb[2] = 0;
    b_tensor.nb[3] = 0;
    b_tensor.type = Q_F32;
    strncpy(b_tensor.name, "b", sizeof(b_tensor.name) - 1);
    
    output_tensor.data = test_data;
    output_tensor.ne[0] = N;
    output_tensor.ne[1] = 1;
    output_tensor.ne[2] = 1;
    output_tensor.ne[3] = 1;
    output_tensor.nb[0] = N * sizeof(float);
    output_tensor.nb[1] = sizeof(float);
    output_tensor.nb[2] = 0;
    output_tensor.nb[3] = 0;
    output_tensor.type = Q_F32;
    strncpy(output_tensor.name, "output", sizeof(output_tensor.name) - 1);
    
    // Compute reference (scalar)
    add_f32_ref(a_data, b_data, ref_data, N);
    
    // Execute AVX2 Add
    err = q_add_f32_avx2(&a_tensor, &b_tensor, &output_tensor);
    if (err != Q_OK) {
        printf("  ✗ FAILED: q_add_f32_avx2 returned error: %s\n", q_strerror(err));
        free(a_data); free(b_data); free(ref_data); free(test_data);
        return 1;
    }
    
    // Compare results
    int error_count = compare_results(ref_data, test_data, N, 
                                      Q_EPSILON_ABS_F32, Q_EPSILON_REL_F32);
    
    free(a_data); free(b_data); free(ref_data); free(test_data);

    if (error_count == 0) {
        printf("  ✓ PASSED\n");
        return 0;
    } else {
        printf("  ✗ FAILED: %d elements out of tolerance\n", error_count);
        return 1;
    }
}

// Test in-place operation (output aliases input)
static int run_inplace_test(uint32_t N, int test_num) {
    printf("\n=== In-Place Test Case %d ===\n", test_num);
    printf("  Vector size: %u (output == a)\n", N);
    
    size_t vector_size = (size_t)N * sizeof(float);
    size_t aligned_size = (vector_size + Q_ALIGN - 1) & ~(Q_ALIGN - 1);
    
    float* a_data = (float*)aligned_alloc(Q_ALIGN, aligned_size);
    float* b_data = (float*)aligned_alloc(Q_ALIGN, aligned_size);
    float* ref_data = (float*)aligned_alloc(Q_ALIGN, aligned_size);
    
    assert(a_data && b_data && ref_data);
    
    // Generate random vectors
    generate_vector(a_data, N);
    generate_vector(b_data, N);
    
    // Copy a_data to ref_data for reference computation
    memcpy(ref_data, a_data, vector_size);
    
    q_tensor a_tensor = {0};
    q_tensor b_tensor = {0};
    q_error_code err;

    a_tensor.data = a_data;
    a_tensor.ne[0] = N;
    a_tensor.ne[1] = 1;
    a_tensor.ne[2] = 1;
    a_tensor.ne[3] = 1;
    a_tensor.nb[0] = N * sizeof(float);
    a_tensor.nb[1] = sizeof(float);
    a_tensor.nb[2] = 0;
    a_tensor.nb[3] = 0;
    a_tensor.type = Q_F32;
    strncpy(a_tensor.name, "a", sizeof(a_tensor.name) - 1);
    
    b_tensor.data = b_data;
    b_tensor.ne[0] = N;
    b_tensor.ne[1] = 1;
    b_tensor.ne[2] = 1;
    b_tensor.ne[3] = 1;
    b_tensor.nb[0] = N * sizeof(float);
    b_tensor.nb[1] = sizeof(float);
    b_tensor.nb[2] = 0;
    b_tensor.nb[3] = 0;
    b_tensor.type = Q_F32;
    strncpy(b_tensor.name, "b", sizeof(b_tensor.name) - 1);
    
    // Compute reference (scalar, in-place)
    add_f32_ref(ref_data, b_data, ref_data, N);
    
    // Execute AVX2 Add (in-place: output == a)
    err = q_add_f32_avx2(&a_tensor, &b_tensor, &a_tensor);
    if (err != Q_OK) {
        printf("  ✗ FAILED: q_add_f32_avx2 returned error: %s\n", q_strerror(err));
        free(a_data); free(b_data); free(ref_data);
        return 1;
    }
    
    // Compare results
    int error_count = compare_results(ref_data, a_data, N, 
                                      Q_EPSILON_ABS_F32, Q_EPSILON_REL_F32);
    
    free(a_data); free(b_data); free(ref_data);

    if (error_count == 0) {
        printf("  ✓ PASSED\n");
        return 0;
    } else {
        printf("  ✗ FAILED: %d elements out of tolerance\n", error_count);
        return 1;
    }
}

int main(void) {
    srand(time(NULL));
    printf("========================================\n");
    printf("Tensor Add FP32 AVX2 Test Suite\n");
    printf("========================================\n");

    int total_failures = 0;

    // Test Cases (N, test_num)
    total_failures += run_test_case(0, 1);    // Zero size (no-op)
    total_failures += run_test_case(1, 2);    // Single element
    total_failures += run_test_case(7, 3);    // Small vector (below 32)
    total_failures += run_test_case(8, 4);    // Exactly 8 elements
    total_failures += run_test_case(31, 5);   // Just below 32
    total_failures += run_test_case(32, 6);   // Exactly 32 elements (aligned)
    total_failures += run_test_case(33, 7);   // Just above 32
    total_failures += run_test_case(64, 8);    // Medium vector
    total_failures += run_test_case(256, 9);  // Large vector
    total_failures += run_test_case(1024, 10); // Very large vector
    
    // In-place tests
    total_failures += run_inplace_test(32, 11);   // In-place: output == a
    total_failures += run_inplace_test(64, 12);   // In-place: output == a

    // Error Handling Tests
    printf("\n=== Error Handling Tests ===\n");
    q_tensor a_err = {0}, b_err = {0}, output_err = {0};
    a_err.type = Q_F32;
    b_err.type = Q_F32;
    output_err.type = Q_F32;
    a_err.ne[0] = 4;
    b_err.ne[0] = 4;
    output_err.ne[0] = 4;
    size_t err_size = (4*sizeof(float) + Q_ALIGN - 1) & ~(Q_ALIGN - 1);
    a_err.data = (float*)aligned_alloc(Q_ALIGN, err_size);
    b_err.data = (float*)aligned_alloc(Q_ALIGN, err_size);
    output_err.data = (float*)aligned_alloc(Q_ALIGN, err_size);
    assert(a_err.data && b_err.data && output_err.data);

    // NULL a
    q_error_code err = q_add_f32_avx2(NULL, &b_err, &output_err);
    assert(err == Q_ERR_INVALID_ARG);
    printf("  ✓ PASSED: NULL a validation\n");

    // NULL b
    err = q_add_f32_avx2(&a_err, NULL, &output_err);
    assert(err == Q_ERR_INVALID_ARG);
    printf("  ✓ PASSED: NULL b validation\n");

    // NULL output
    err = q_add_f32_avx2(&a_err, &b_err, NULL);
    assert(err == Q_ERR_INVALID_ARG);
    printf("  ✓ PASSED: NULL output validation\n");

    // Shape mismatch (N != b->ne[0])
    q_tensor shape_mismatch = {0};
    shape_mismatch.type = Q_F32;
    shape_mismatch.ne[0] = 8;  // Different size!
    shape_mismatch.data = (float*)aligned_alloc(Q_ALIGN, Q_ALIGN);
    assert(shape_mismatch.data);
    err = q_add_f32_avx2(&a_err, &shape_mismatch, &output_err);
    assert(err == Q_ERR_INVALID_SIZE);
    printf("  ✓ PASSED: Shape mismatch validation\n");
    free(shape_mismatch.data);

    free(a_err.data); free(b_err.data); free(output_err.data);

    if (total_failures == 0) {
        printf("\n✓ All Tensor Add FP32 tests PASSED\n");
        return 0;
    } else {
        printf("\n✗ %d Tensor Add FP32 test case(s) FAILED\n", total_failures);
        return 1;
    }
}

