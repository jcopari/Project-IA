#include "qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// Reference implementation (scalar, slow but correct)
// Used to validate the AVX2 implementation
static void matmul_f32_ref(
    const float* restrict A,  // [M, K]
    const float* restrict B,  // [K, N]
    float* restrict C,        // [M, N] (output)
    uint32_t M,
    uint32_t K,
    uint32_t N
) {
    // Standard matrix multiplication: C = A @ B
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Generate random FP32 matrix
static void generate_matrix(float* matrix, uint32_t rows, uint32_t cols) {
    for (uint32_t i = 0; i < rows * cols; i++) {
        // Random values in [-1.0, 1.0]
        matrix[i] = -1.0f + ((float)rand() / RAND_MAX) * 2.0f;
    }
}

// Compare results with tolerance (hybrid: absolute + relative)
static int compare_results(
    const float* ref,
    const float* test,
    uint32_t M,
    uint32_t N,
    float abs_tol,
    float rel_tol
) {
    int errors = 0;
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    
    for (uint32_t i = 0; i < M * N; i++) {
        float abs_err = fabsf(ref[i] - test[i]);
        float rel_err = (fabsf(ref[i]) > 1e-8f) ? abs_err / fabsf(ref[i]) : abs_err;
        
        if (abs_err > max_abs_error) max_abs_error = abs_err;
        if (rel_err > max_rel_error) max_rel_error = rel_err;
        
        // Hybrid tolerance: pass if EITHER absolute OR relative error is within tolerance
        if (abs_err > abs_tol && rel_err > rel_tol) {
            if (errors < 10) {  // Print first 10 errors
                uint32_t row = i / N;
                uint32_t col = i % N;
                printf("  Error at [%u, %u]: ref=%.6f, test=%.6f, abs_err=%.6e, rel_err=%.6e\n",
                       row, col, ref[i], test[i], abs_err, rel_err);
            }
            errors++;
        }
    }
    
    printf("  Max absolute error: %.6e (tolerance: %.6e)\n", max_abs_error, abs_tol);
    printf("  Max relative error: %.6e (tolerance: %.6e)\n", max_rel_error, rel_tol);
    
    return errors;
}

// Test case runner
static int run_test_case(uint32_t M, uint32_t K, uint32_t N, int test_num) {
    printf("\n=== Test Case %d ===\n", test_num);
    printf("  Matrix: A[%u, %u] @ B[%u, %u] = C[%u, %u]\n", M, K, K, N, M, N);
    
    // Initialize context
    q_context ctx = {0};
    q_error_code err = q_alloc_arena(&ctx, 1024 * 1024 * 1024); // 1GB arena
    if (err != Q_OK) {
        printf("  ✗ FAILED: Arena allocation failed\n");
        return 1;
    }
    
    // Allocate aligned buffers
    const size_t A_size = M * K * sizeof(float);
    const size_t B_size = K * N * sizeof(float);
    const size_t C_size = M * N * sizeof(float);
    const size_t ref_size = M * N * sizeof(float);
    
    // Allocate in arena (64-byte aligned)
    float* A_data = (float*)q_arena_alloc(&ctx, A_size);
    float* B_data = (float*)q_arena_alloc(&ctx, B_size);
    float* C_data = (float*)q_arena_alloc(&ctx, C_size);
    float* ref_data = (float*)q_arena_alloc(&ctx, ref_size);
    
    if (!A_data || !B_data || !C_data || !ref_data) {
        printf("  ✗ FAILED: Memory allocation failed\n");
        q_free_memory(&ctx);
        return 1;
    }
    
    // Generate random matrices
    generate_matrix(A_data, M, K);
    generate_matrix(B_data, K, N);
    
    // Initialize output to zero
    memset(C_data, 0, C_size);
    memset(ref_data, 0, ref_size);
    
    // Create tensor structures
    q_tensor A_tensor = {0};
    q_tensor B_tensor = {0};
    q_tensor C_tensor = {0};
    
    A_tensor.data = A_data;
    A_tensor.ne[0] = M;
    A_tensor.ne[1] = K;
    A_tensor.ne[2] = 1;
    A_tensor.ne[3] = 1;
    A_tensor.nb[0] = K * sizeof(float);
    A_tensor.nb[1] = sizeof(float);
    A_tensor.nb[2] = 0;
    A_tensor.nb[3] = 0;
    A_tensor.type = Q_F32;
    strncpy(A_tensor.name, "A", sizeof(A_tensor.name) - 1);
    
    B_tensor.data = B_data;
    B_tensor.ne[0] = K;
    B_tensor.ne[1] = N;
    B_tensor.ne[2] = 1;
    B_tensor.ne[3] = 1;
    B_tensor.nb[0] = N * sizeof(float);
    B_tensor.nb[1] = sizeof(float);
    B_tensor.nb[2] = 0;
    B_tensor.nb[3] = 0;
    B_tensor.type = Q_F32;
    strncpy(B_tensor.name, "B", sizeof(B_tensor.name) - 1);
    
    C_tensor.data = C_data;
    C_tensor.ne[0] = M;
    C_tensor.ne[1] = N;
    C_tensor.ne[2] = 1;
    C_tensor.ne[3] = 1;
    C_tensor.nb[0] = N * sizeof(float);
    C_tensor.nb[1] = sizeof(float);
    C_tensor.nb[2] = 0;
    C_tensor.nb[3] = 0;
    C_tensor.type = Q_F32;
    strncpy(C_tensor.name, "C", sizeof(C_tensor.name) - 1);
    
    // Compute reference (scalar)
    matmul_f32_ref(A_data, B_data, ref_data, M, K, N);
    
    // Allocate output in arena (don't reset - A and B data are still in arena)
    C_data = (float*)q_arena_alloc(&ctx, C_size);
    if (!C_data) {
        printf("  ✗ FAILED: Output allocation failed\n");
        q_free_memory(&ctx);
        return 1;
    }
    C_tensor.data = C_data;
    memset(C_data, 0, C_size);
    
    // Execute AVX2 MatMul
    err = q_matmul_f32_avx2(&A_tensor, &B_tensor, &C_tensor, &ctx);
    if (err != Q_OK) {
        printf("  ✗ FAILED: q_matmul_f32_avx2 returned error: %s\n", q_strerror(err));
        q_free_memory(&ctx);
        return 1;
    }
    
    // Compare results
    int error_count = compare_results(ref_data, C_data, M, N, 
                                      Q_EPSILON_ABS_F32, Q_EPSILON_REL_F32);
    
    if (error_count == 0) {
        printf("  ✓ PASSED\n");
        q_free_memory(&ctx);
        return 0;
    } else {
        printf("  ✗ FAILED: %d elements out of tolerance\n", error_count);
        q_free_memory(&ctx);
        return 1;
    }
}

// Error handling tests
static int test_error_handling(void) {
    printf("\n=== Error Handling Tests ===\n");
    int failures = 0;
    
    q_context ctx = {0};
    q_error_code err = q_alloc_arena(&ctx, 1024 * 1024);
    if (err != Q_OK) {
        printf("  ✗ FAILED: Arena allocation failed\n");
        return 1;
    }
    
    q_tensor A = {0}, B = {0}, C = {0};
    
    // Test 1: NULL A
    err = q_matmul_f32_avx2(NULL, &B, &C, &ctx);
    if (err != Q_ERR_INVALID_ARG) {
        printf("  ✗ FAILED: NULL A should return Q_ERR_INVALID_ARG, got %s\n", q_strerror(err));
        failures++;
    } else {
        printf("  ✓ PASSED: NULL A validation\n");
    }
    
    // Test 2: NULL B
    err = q_matmul_f32_avx2(&A, NULL, &C, &ctx);
    if (err != Q_ERR_INVALID_ARG) {
        printf("  ✗ FAILED: NULL B should return Q_ERR_INVALID_ARG, got %s\n", q_strerror(err));
        failures++;
    } else {
        printf("  ✓ PASSED: NULL B validation\n");
    }
    
    // Test 3: NULL C
    err = q_matmul_f32_avx2(&A, &B, NULL, &ctx);
    if (err != Q_ERR_INVALID_ARG) {
        printf("  ✗ FAILED: NULL C should return Q_ERR_INVALID_ARG, got %s\n", q_strerror(err));
        failures++;
    } else {
        printf("  ✓ PASSED: NULL C validation\n");
    }
    
    // Test 4: Shape mismatch (A[4,8] @ B[4,4] should fail)
    A.ne[0] = 4; A.ne[1] = 8;
    B.ne[0] = 4; B.ne[1] = 4;  // Wrong! Should be [8, 4]
    C.ne[0] = 4; C.ne[1] = 4;
    err = q_matmul_f32_avx2(&A, &B, &C, &ctx);
    if (err != Q_ERR_INVALID_SIZE) {
        printf("  ✗ FAILED: Shape mismatch should return Q_ERR_INVALID_SIZE, got %s\n", q_strerror(err));
        failures++;
    } else {
        printf("  ✓ PASSED: Shape mismatch validation\n");
    }
    
    // Test 5: Zero dimensions
    A.ne[0] = 0; A.ne[1] = 8;
    B.ne[0] = 8; B.ne[1] = 4;
    C.ne[0] = 0; C.ne[1] = 4;
    err = q_matmul_f32_avx2(&A, &B, &C, &ctx);
    if (err != Q_ERR_INVALID_SIZE) {
        printf("  ✗ FAILED: Zero M should return Q_ERR_INVALID_SIZE, got %s\n", q_strerror(err));
        failures++;
    } else {
        printf("  ✓ PASSED: Zero M validation\n");
    }
    
    q_free_memory(&ctx);
    return failures;
}

int main(void) {
    printf("========================================\n");
    printf("MatMul FP32 AVX2 Test Suite\n");
    printf("========================================\n");
    
    int failures = 0;
    
    // Test cases: M, K, N
    struct {
        uint32_t M, K, N;
    } test_cases[] = {
        {4, 8, 4},      // Small matrices
        {32, 64, 32},   // Medium matrices
        {128, 256, 128}, // Large matrices
        {1, 8, 4},      // Edge case: M=1 (single row)
        {4, 8, 1},      // Edge case: N=1 (single column)
        {4, 1, 4},      // Edge case: K=1 (single dimension)
        {16, 32, 16},   // Power-of-2 dimensions
        {15, 31, 17},   // Non-power-of-2 dimensions
    };
    
    // Run test cases
    for (size_t i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++) {
        failures += run_test_case(test_cases[i].M, test_cases[i].K, test_cases[i].N, (int)(i + 1));
    }
    
    // Error handling tests
    failures += test_error_handling();
    
    printf("\n========================================\n");
    if (failures == 0) {
        printf("All tests PASSED\n");
        return 0;
    } else {
        printf("FAILED: %d test case(s) failed\n", failures);
        return 1;
    }
}

