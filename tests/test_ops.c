#include "qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// Reference RMSNorm (scalar)
static void rmsnorm_ref(
    const float* x,
    const float* weight,
    float* output,
    uint32_t N,
    float eps
) {
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        sum_sq += x[i] * x[i];
    }
    float mean_sq = sum_sq / (float)N;
    float rsqrt_val = 1.0f / sqrtf(mean_sq + eps);
    
    for (uint32_t i = 0; i < N; i++) {
        output[i] = x[i] * rsqrt_val * weight[i];
    }
}

// Reference RoPE (scalar)
static void rope_ref(
    const float* x,
    const float* cos,
    const float* sin,
    float* output,
    uint32_t N
) {
    for (uint32_t i = 0; i < N / 2; i++) {
        float x_val = x[2*i];
        float y_val = x[2*i + 1];
        float c = cos[i];
        float s = sin[i];
        
        output[2*i] = x_val * c - y_val * s;
        output[2*i + 1] = x_val * s + y_val * c;
    }
}

// Reference SiLU (scalar)
static void silu_ref(const float* x, float* output, uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-x[i]));
        output[i] = x[i] * sigmoid;
    }
}

// Reference Softmax (scalar)
static void softmax_ref(const float* x, float* output, uint32_t N) {
    // Find max
    float max_val = x[0];
    for (uint32_t i = 1; i < N; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        output[i] = expf(x[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    for (uint32_t i = 0; i < N; i++) {
        output[i] /= sum;
    }
}

// Compare results with tolerance
static int compare_results(
    const float* ref,
    const float* test,
    uint32_t N,
    float abs_tol,
    float rel_tol,
    const char* op_name __attribute__((unused))
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

// Test RMSNorm
static int test_rmsnorm(void) {
    printf("\n=== RMSNorm Test ===\n");
    
    const uint32_t N = 128;
    float* x = (float*)aligned_alloc(Q_ALIGN, N * sizeof(float));
    float* weight = (float*)aligned_alloc(Q_ALIGN, N * sizeof(float));
    float* output_ref = (float*)aligned_alloc(Q_ALIGN, N * sizeof(float));
    float* output_test = (float*)aligned_alloc(Q_ALIGN, N * sizeof(float));
    
    if (!x || !weight || !output_ref || !output_test) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        abort();
    }
    
    // Generate random data
    srand(42);
    for (uint32_t i = 0; i < N; i++) {
        x[i] = -2.0f + ((float)rand() / RAND_MAX) * 4.0f;
        weight[i] = 0.5f + ((float)rand() / RAND_MAX) * 1.5f;
    }
    
    const float eps = 1e-5f;
    
    rmsnorm_ref(x, weight, output_ref, N, eps);
    q_error_code err = q_rmsnorm_f32_avx2(x, weight, output_test, N, eps);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: q_rmsnorm_f32_avx2 failed: %s\n", q_strerror(err));
        return 1;
    }
    
    // Compare (use approximation tolerances for rsqrt)
    int errors = compare_results(output_ref, output_test, N,
                                  Q_EPSILON_ABS_APPROX, Q_EPSILON_REL_APPROX,
                                  "RMSNorm");
    
    if (errors == 0) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED: %d / %u errors\n", errors, N);
    }
    
    free(x);
    free(weight);
    free(output_ref);
    free(output_test);
    
    return errors;
}

// Test RoPE
static int test_rope(void) {
    printf("\n=== RoPE Test ===\n");
    
    const uint32_t N = 128;
    float* x = (float*)aligned_alloc(Q_ALIGN, N * sizeof(float));
    float* cos = (float*)aligned_alloc(Q_ALIGN, (N/2) * sizeof(float));
    float* sin = (float*)aligned_alloc(Q_ALIGN, (N/2) * sizeof(float));
    float* output_ref = (float*)aligned_alloc(Q_ALIGN, N * sizeof(float));
    float* output_test = (float*)aligned_alloc(Q_ALIGN, N * sizeof(float));
    
    if (!x || !cos || !sin || !output_ref || !output_test) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        abort();
    }
    
    // Generate random data
    srand(42);
    for (uint32_t i = 0; i < N; i++) {
        x[i] = -1.0f + ((float)rand() / RAND_MAX) * 2.0f;
    }
    
    // CRITICAL FIX: AVX2 implementation expects duplicated layout [c0, c0, c1, c1, ...]
    // Create arrays with duplicated layout for AVX2
    float* cos_avx2 = (float*)aligned_alloc(Q_ALIGN, N * sizeof(float));
    float* sin_avx2 = (float*)aligned_alloc(Q_ALIGN, N * sizeof(float));
    if (!cos_avx2 || !sin_avx2) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        abort();
    }
    
    for (uint32_t i = 0; i < N/2; i++) {
        float angle = (float)i * 0.1f;
        float c = cosf(angle);
        float s = sinf(angle);
        // Duplicate for AVX2 layout: [c0, c0, c1, c1, ...]
        cos_avx2[i * 2] = c;
        cos_avx2[i * 2 + 1] = c;
        sin_avx2[i * 2] = s;
        sin_avx2[i * 2 + 1] = s;
        // Also store in original arrays for reference implementation
        cos[i] = c;
        sin[i] = s;
    }
    
    rope_ref(x, cos, sin, output_ref, N);
    q_rope_f32_avx2(x, cos_avx2, sin_avx2, output_test, N);
    
    // Compare (use exact tolerances for RoPE - should be very precise)
    int errors = compare_results(output_ref, output_test, N,
                                  Q_EPSILON_ABS_F32, Q_EPSILON_REL_F32,
                                  "RoPE");
    
    if (errors == 0) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED: %d / %u errors\n", errors, N);
    }
    
    free(x);
    free(cos);
    free(sin);
    free(cos_avx2);
    free(sin_avx2);
    free(output_ref);
    free(output_test);
    
    return errors;
}

// Test SiLU
static int test_silu(void) {
    printf("\n=== SiLU Test ===\n");
    
    const uint32_t N = 128;
    float* x = (float*)aligned_alloc(Q_ALIGN, N * sizeof(float));
    float* output_ref = (float*)aligned_alloc(Q_ALIGN, N * sizeof(float));
    float* output_test = (float*)aligned_alloc(Q_ALIGN, N * sizeof(float));
    
    if (!x || !output_ref || !output_test) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        abort();
    }
    
    // Generate random data
    srand(42);
    for (uint32_t i = 0; i < N; i++) {
        x[i] = -5.0f + ((float)rand() / RAND_MAX) * 10.0f;
    }
    
    silu_ref(x, output_ref, N);
    q_error_code err = q_silu_f32_avx2(x, output_test, N);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: q_silu_f32_avx2 failed: %s\n", q_strerror(err));
        return 1;
    }
    
    // Compare (use approximation tolerances for SiLU)
    int errors = compare_results(output_ref, output_test, N,
                                  Q_EPSILON_ABS_APPROX, Q_EPSILON_REL_APPROX,
                                  "SiLU");
    
    if (errors == 0) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED: %d / %u errors\n", errors, N);
    }
    
    free(x);
    free(output_ref);
    free(output_test);
    
    return errors;
}

// Test Softmax
static int test_softmax(void) {
    printf("\n=== Softmax Test ===\n");
    
    const uint32_t N = 128;
    float* x = (float*)aligned_alloc(Q_ALIGN, N * sizeof(float));
    float* output_ref = (float*)aligned_alloc(Q_ALIGN, N * sizeof(float));
    float* output_test = (float*)aligned_alloc(Q_ALIGN, N * sizeof(float));
    
    if (!x || !output_ref || !output_test) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        abort();
    }
    
    // Generate random data
    srand(42);
    for (uint32_t i = 0; i < N; i++) {
        x[i] = -2.0f + ((float)rand() / RAND_MAX) * 4.0f;
    }
    
    softmax_ref(x, output_ref, N);
    q_error_code err = q_softmax_f32_avx2(x, output_test, N);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: q_softmax_f32_avx2 failed: %s\n", q_strerror(err));
        return 1;
    }
    
    // Verify sum = 1.0
    float sum_test = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        sum_test += output_test[i];
    }
    printf("  Sum check: %.6f (expected: 1.0)\n", sum_test);
    
    if (fabsf(sum_test - 1.0f) > 1e-4f) {
        printf("  ✗ FAILED: Sum invariant violated\n");
        free(x);
        free(output_ref);
        free(output_test);
        return N; // Return error count
    }
    
    // Compare (use approximation tolerances for Softmax)
    int errors = compare_results(output_ref, output_test, N,
                                  Q_EPSILON_ABS_APPROX, Q_EPSILON_REL_APPROX,
                                  "Softmax");
    
    if (errors == 0) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED: %d / %u errors\n", errors, N);
    }
    
    free(x);
    free(output_ref);
    free(output_test);
    
    return errors;
}

int main(void) {
    printf("=== Qorus Operations Test Suite ===\n");
    printf("Validating RMSNorm, RoPE, SiLU, and Softmax implementations\n");
    
    int total_errors = 0;
    total_errors += test_rmsnorm();
    total_errors += test_rope();
    total_errors += test_silu();
    total_errors += test_softmax();
    
    printf("\n=== Test Summary ===\n");
    if (total_errors == 0) {
        printf("✓ All tests PASSED\n");
        return 0;
    } else {
        printf("✗ FAILED: %d total errors\n", total_errors);
        return 1;
    }
}

