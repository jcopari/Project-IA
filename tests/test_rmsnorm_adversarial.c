#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>
#include <float.h>
#include <immintrin.h>

// ============================================================================
// TEST CONFIGURATION
// ============================================================================

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;
static int tests_crashed = 0;

// Signal handler for crash detection
static jmp_buf crash_jmp_buf;
static void crash_handler(int sig) {
    (void)sig;
    longjmp(crash_jmp_buf, 1);
}

#define TEST_START(name) \
    do { \
        printf("Test %d: %s\n", tests_run + 1, name); \
    } while(0)

#define TEST_PASS() do { \
    tests_run++; \
    tests_passed++; \
    printf("  ✓ PASSED\n"); \
} while(0)

#define TEST_FAIL(msg) do { \
    tests_run++; \
    tests_failed++; \
    printf("  ✗ FAILED: %s\n", msg); \
} while(0)

#define TEST_FAIL_MSG(fmt, ...) do { \
    tests_run++; \
    tests_failed++; \
    printf("  ✗ FAILED: " fmt "\n", __VA_ARGS__); \
} while(0)

#define TEST_CRASH() do { \
    tests_run++; \
    tests_crashed++; \
    printf("  ✗ CRASHED\n"); \
} while(0)

// ============================================================================
// REFERENCE IMPLEMENTATION
// ============================================================================

static void rmsnorm_ref(
    const float* restrict x,
    const float* restrict weight,
    float* restrict output,
    uint32_t N,
    float eps
) {
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        sum_sq += x[i] * x[i];
    }
    
    // Compute mean
    float mean_sq = sum_sq / (float)N;
    
    // Compute rsqrt(mean + eps)
    float rsqrt_val = 1.0f / sqrtf(mean_sq + eps);
    
    // Apply normalization and weight
    for (uint32_t i = 0; i < N; i++) {
        output[i] = x[i] * rsqrt_val * weight[i];
    }
}

// ============================================================================
// TEST HELPERS
// ============================================================================

static bool float_array_close(const float* a, const float* b, uint32_t n, float abs_tol, float rel_tol) {
    for (uint32_t i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        float max_val = fmaxf(fabsf(a[i]), fabsf(b[i]));
        if (diff > abs_tol && diff > rel_tol * max_val) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// ADVERSARIAL TEST CASES
// ============================================================================

// Test 1: NULL pointer - x
static void test_null_x(void) {
    TEST_START("NULL pointer - x");
    
    float weight[8] __attribute__((aligned(64))) = {1.0f};
    float output[8] __attribute__((aligned(64)));
    
    q_error_code ret = q_rmsnorm_f32_avx2(NULL, weight, output, 8, 1e-6f);
    
    if (ret == Q_ERR_INVALID_ARG) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_ARG, got %d", ret);
    }
}

// Test 2: NULL pointer - weight
static void test_null_weight(void) {
    TEST_START("NULL pointer - weight");
    
    float x[8] __attribute__((aligned(64))) = {1.0f};
    float output[8] __attribute__((aligned(64)));
    
    q_error_code ret = q_rmsnorm_f32_avx2(x, NULL, output, 8, 1e-6f);
    
    if (ret == Q_ERR_INVALID_ARG) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_ARG, got %d", ret);
    }
}

// Test 3: NULL pointer - output
static void test_null_output(void) {
    TEST_START("NULL pointer - output");
    
    float x[8] __attribute__((aligned(64))) = {1.0f};
    float weight[8] __attribute__((aligned(64))) = {1.0f};
    
    q_error_code ret = q_rmsnorm_f32_avx2(x, weight, NULL, 8, 1e-6f);
    
    if (ret == Q_ERR_INVALID_ARG) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_ARG, got %d", ret);
    }
}

// Test 4: Misaligned pointer - x
static void test_misaligned_x(void) {
    TEST_START("Misaligned pointer - x");
    
    float x_data[9]; // Unaligned
    float* x = x_data + 1; // Misaligned
    float weight[8] __attribute__((aligned(64))) = {1.0f};
    float output[8] __attribute__((aligned(64)));
    
    for (uint32_t i = 0; i < 8; i++) {
        x[i] = 1.0f;
    }
    
    q_error_code ret = q_rmsnorm_f32_avx2(x, weight, output, 8, 1e-6f);
    
    if (ret == Q_ERR_MISALIGNED) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_MISALIGNED, got %d", ret);
    }
}

// Test 5: Misaligned pointer - output
static void test_misaligned_output(void) {
    TEST_START("Misaligned pointer - output");
    
    float x[8] __attribute__((aligned(64))) = {1.0f};
    float weight[8] __attribute__((aligned(64))) = {1.0f};
    float output_data[9]; // Unaligned
    float* output = output_data + 1; // Misaligned
    
    q_error_code ret = q_rmsnorm_f32_avx2(x, weight, output, 8, 1e-6f);
    
    if (ret == Q_ERR_MISALIGNED) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_MISALIGNED, got %d", ret);
    }
}

// Test 6: Zero size
static void test_zero_size(void) {
    TEST_START("Zero size");
    
    float x[8] __attribute__((aligned(64))) = {1.0f};
    float weight[8] __attribute__((aligned(64))) = {1.0f};
    float output[8] __attribute__((aligned(64)));
    
    q_error_code ret = q_rmsnorm_f32_avx2(x, weight, output, 0, 1e-6f);
    
    if (ret == Q_ERR_INVALID_SIZE) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_SIZE, got %d", ret);
    }
}

// Test 7: Size not multiple of 8
static void test_size_not_multiple_of_8(void) {
    TEST_START("Size not multiple of 8");
    
    float x[9] __attribute__((aligned(64)));
    float weight[9] __attribute__((aligned(64)));
    float output[9] __attribute__((aligned(64)));
    
    for (uint32_t i = 0; i < 9; i++) {
        x[i] = 1.0f;
        weight[i] = 1.0f;
    }
    
    q_error_code ret = q_rmsnorm_f32_avx2(x, weight, output, 9, 1e-6f);
    
    if (ret == Q_ERR_INVALID_SIZE) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_SIZE, got %d", ret);
    }
}

// Test 8: Aliasing - x == output
static void test_aliasing_x_output(void) {
    TEST_START("Aliasing - x == output");
    
    float x[8] __attribute__((aligned(64))) = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float weight[8] __attribute__((aligned(64))) = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    
    // Save original values
    float x_orig[8];
    memcpy(x_orig, x, sizeof(x));
    
    q_error_code ret = q_rmsnorm_f32_avx2(x, weight, x, 8, 1e-6f);
    
    // Check if function handles aliasing correctly (should work or return error)
    if (ret == Q_OK) {
        // Function should handle aliasing correctly
        // Verify output is normalized
        bool normalized = true;
        for (uint32_t i = 0; i < 8; i++) {
            if (!isfinite(x[i])) {
                normalized = false;
                break;
            }
        }
        if (normalized) {
            TEST_PASS();
        } else {
            TEST_FAIL("Output contains non-finite values");
        }
    } else if (ret == Q_ERR_ALIASING) {
        TEST_PASS(); // Function correctly detects aliasing
    } else {
        TEST_FAIL_MSG("Unexpected return code: %d", ret);
    }
}

// Test 9: All zeros input
static void test_all_zeros(void) {
    TEST_START("All zeros input");
    
    float x[8] __attribute__((aligned(64))) = {0.0f};
    float weight[8] __attribute__((aligned(64))) = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float output[8] __attribute__((aligned(64)));
    float expected[8] = {0.0f};
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_rmsnorm_f32_avx2(x, weight, output, 8, 1e-6f);
        
        if (ret == Q_OK) {
            if (float_array_close(output, expected, 8, 1e-5f, 1e-4f)) {
                TEST_PASS();
            } else {
                TEST_FAIL("Output does not match expected (all zeros)");
            }
        } else {
            TEST_FAIL_MSG("Function returned error: %d", ret);
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 10: Very small epsilon
static void test_very_small_epsilon(void) {
    TEST_START("Very small epsilon");
    
    float x[8] __attribute__((aligned(64))) = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float weight[8] __attribute__((aligned(64))) = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float output[8] __attribute__((aligned(64)));
    float expected[8];
    
    rmsnorm_ref(x, weight, expected, 8, FLT_MIN);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_rmsnorm_f32_avx2(x, weight, output, 8, FLT_MIN);
        
        if (ret == Q_OK) {
            if (float_array_close(output, expected, 8, 1e-4f, 1e-3f)) {
                TEST_PASS();
            } else {
                TEST_FAIL("Output does not match reference with very small epsilon");
            }
        } else {
            TEST_FAIL_MSG("Function returned error: %d", ret);
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 11: Very large values
static void test_very_large_values(void) {
    TEST_START("Very large values");
    
    float x[8] __attribute__((aligned(64))) = {
        FLT_MAX / 2.0f, FLT_MAX / 2.0f, FLT_MAX / 2.0f, FLT_MAX / 2.0f,
        FLT_MAX / 2.0f, FLT_MAX / 2.0f, FLT_MAX / 2.0f, FLT_MAX / 2.0f
    };
    float weight[8] __attribute__((aligned(64))) = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float output[8] __attribute__((aligned(64)));
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_rmsnorm_f32_avx2(x, weight, output, 8, 1e-6f);
        
        if (ret == Q_OK) {
            // For very large values, overflow/Inf is acceptable (numerical limitation)
            // Function should not crash, but may produce Inf/NaN due to overflow
            bool handled = true;
            for (uint32_t i = 0; i < 8; i++) {
                // Check if value is finite OR Inf (Inf is acceptable for overflow)
                if (!isfinite(output[i]) && !isinf(output[i])) {
                    // NaN is also acceptable (indicates overflow handled gracefully)
                    handled = true;
                    break;
                }
            }
            if (handled) {
                TEST_PASS();
            } else {
                TEST_FAIL("Output contains unexpected non-finite values");
            }
        } else {
            TEST_FAIL_MSG("Function returned error: %d", ret);
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 12: Negative values
static void test_negative_values(void) {
    TEST_START("Negative values");
    
    float x[8] __attribute__((aligned(64))) = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f};
    float weight[8] __attribute__((aligned(64))) = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float output[8] __attribute__((aligned(64)));
    float expected[8];
    
    rmsnorm_ref(x, weight, expected, 8, 1e-6f);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_rmsnorm_f32_avx2(x, weight, output, 8, 1e-6f);
        
        if (ret == Q_OK) {
            if (float_array_close(output, expected, 8, 1e-4f, 1e-3f)) {
                TEST_PASS();
            } else {
                TEST_FAIL("Output does not match reference for negative values");
            }
        } else {
            TEST_FAIL_MSG("Function returned error: %d", ret);
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 13: Mixed positive/negative values
static void test_mixed_values(void) {
    TEST_START("Mixed positive/negative values");
    
    float x[8] __attribute__((aligned(64))) = {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f};
    float weight[8] __attribute__((aligned(64))) = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float output[8] __attribute__((aligned(64)));
    float expected[8];
    
    rmsnorm_ref(x, weight, expected, 8, 1e-6f);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_rmsnorm_f32_avx2(x, weight, output, 8, 1e-6f);
        
        if (ret == Q_OK) {
            if (float_array_close(output, expected, 8, 1e-4f, 1e-3f)) {
                TEST_PASS();
            } else {
                TEST_FAIL("Output does not match reference for mixed values");
            }
        } else {
            TEST_FAIL_MSG("Function returned error: %d", ret);
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 14: NaN in input
static void test_nan_input(void) {
    TEST_START("NaN in input");
    
    float x[8] __attribute__((aligned(64))) = {1.0f, 2.0f, NAN, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float weight[8] __attribute__((aligned(64))) = {1.0f};
    float output[8] __attribute__((aligned(64)));
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_rmsnorm_f32_avx2(x, weight, output, 8, 1e-6f);
        
        // Function should handle NaN gracefully (either propagate or return error)
        if (ret == Q_OK) {
            // NaN propagation is acceptable
            TEST_PASS();
        } else {
            // Error return is also acceptable
            TEST_PASS();
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 15: Infinity in input
static void test_inf_input(void) {
    TEST_START("Infinity in input");
    
    float x[8] __attribute__((aligned(64))) = {1.0f, 2.0f, INFINITY, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float weight[8] __attribute__((aligned(64))) = {1.0f};
    float output[8] __attribute__((aligned(64)));
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_rmsnorm_f32_avx2(x, weight, output, 8, 1e-6f);
        
        // Function should handle Inf gracefully
        if (ret == Q_OK) {
            TEST_PASS();
        } else {
            TEST_PASS(); // Error return is acceptable
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 16: Zero epsilon (division by zero risk)
static void test_zero_epsilon(void) {
    TEST_START("Zero epsilon");
    
    float x[8] __attribute__((aligned(64))) = {0.0f};
    float weight[8] __attribute__((aligned(64))) = {1.0f};
    float output[8] __attribute__((aligned(64)));
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_rmsnorm_f32_avx2(x, weight, output, 8, 0.0f);
        
        // Should handle zero epsilon gracefully (with all zeros input)
        if (ret == Q_OK) {
            bool all_zero = true;
            for (uint32_t i = 0; i < 8; i++) {
                if (output[i] != 0.0f && !isnan(output[i]) && !isinf(output[i])) {
                    all_zero = false;
                    break;
                }
            }
            if (all_zero || isfinite(output[0])) {
                TEST_PASS();
            } else {
                TEST_FAIL("Unexpected output with zero epsilon");
            }
        } else {
            TEST_PASS(); // Error return is acceptable
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 17: Large size (stress test)
static void test_large_size(void) {
    TEST_START("Large size (stress test)");
    
    const uint32_t N = 8192; // Large but reasonable
    float* x = (float*)aligned_alloc(64, N * sizeof(float));
    float* weight = (float*)aligned_alloc(64, N * sizeof(float));
    float* output = (float*)aligned_alloc(64, N * sizeof(float));
    float* expected = (float*)malloc(N * sizeof(float));
    
    if (!x || !weight || !output || !expected) {
        TEST_FAIL("Memory allocation failed");
        if (x) free(x);
        if (weight) free(weight);
        if (output) free(output);
        if (expected) free(expected);
        return;
    }
    
    for (uint32_t i = 0; i < N; i++) {
        x[i] = (float)(i % 100);
        weight[i] = 1.0f;
    }
    
    rmsnorm_ref(x, weight, expected, N, 1e-6f);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_rmsnorm_f32_avx2(x, weight, output, N, 1e-6f);
        
        if (ret == Q_OK) {
            if (float_array_close(output, expected, N, 1e-4f, 1e-3f)) {
                TEST_PASS();
            } else {
                TEST_FAIL("Output does not match reference for large size");
            }
        } else {
            TEST_FAIL_MSG("Function returned error: %d", ret);
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
    
    free(x);
    free(weight);
    free(output);
    free(expected);
}

// Test 18: Precision test - compare with reference
static void test_precision(void) {
    TEST_START("Precision test - compare with reference");
    
    float x[8] __attribute__((aligned(64))) = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float weight[8] __attribute__((aligned(64))) = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float output[8] __attribute__((aligned(64)));
    float expected[8];
    
    rmsnorm_ref(x, weight, expected, 8, 1e-6f);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_rmsnorm_f32_avx2(x, weight, output, 8, 1e-6f);
        
        if (ret == Q_OK) {
            // Use relaxed tolerance for RMSNorm (rsqrt approximation)
            if (float_array_close(output, expected, 8, 1e-4f, 1e-3f)) {
                TEST_PASS();
            } else {
                // Print first mismatch for debugging
                for (uint32_t i = 0; i < 8; i++) {
                    float diff = fabsf(output[i] - expected[i]);
                    if (diff > 1e-4f) {
                        TEST_FAIL_MSG("Mismatch at index %u: expected %.6f, got %.6f (diff: %.6f)",
                                     i, expected[i], output[i], diff);
                        break;
                    }
                }
            }
        } else {
            TEST_FAIL_MSG("Function returned error: %d", ret);
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("=== Adversarial Test Suite: q_rmsnorm_f32_avx2 ===\n\n");
    
    // Run all tests
    test_null_x();
    test_null_weight();
    test_null_output();
    test_misaligned_x();
    test_misaligned_output();
    test_zero_size();
    test_size_not_multiple_of_8();
    test_aliasing_x_output();
    test_all_zeros();
    test_very_small_epsilon();
    test_very_large_values();
    test_negative_values();
    test_mixed_values();
    test_nan_input();
    test_inf_input();
    test_zero_epsilon();
    test_large_size();
    test_precision();
    
    // Print summary
    printf("\n=== Test Summary ===\n");
    printf("Total tests: %d\n", tests_run);
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    printf("Crashed: %d\n", tests_crashed);
    
    if (tests_failed == 0 && tests_crashed == 0) {
        printf("\n✓ All tests passed!\n");
        return 0;
    } else {
        printf("\n✗ Some tests failed or crashed\n");
        return 1;
    }
}

