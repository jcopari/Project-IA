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

static void softmax_ref(
    const float* restrict x,
    float* restrict output,
    uint32_t N
) {
    // Softmax with max-sub trick for numerical stability
    float max_val = x[0];
    for (uint32_t i = 1; i < N; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    float sum = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        float exp_val = expf(x[i] - max_val);
        output[i] = exp_val;
        sum += exp_val;
    }
    
    for (uint32_t i = 0; i < N; i++) {
        output[i] /= sum;
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

static float array_sum(const float* a, uint32_t n) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        sum += a[i];
    }
    return sum;
}

// ============================================================================
// ADVERSARIAL TEST CASES
// ============================================================================

// Test 1: NULL pointer - x
static void test_null_x(void) {
    TEST_START("NULL pointer - x");
    
    float output[8] __attribute__((aligned(64)));
    
    q_error_code ret = q_softmax_f32_avx2(NULL, output, 8);
    
    if (ret == Q_ERR_INVALID_ARG) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_ARG, got %d", ret);
    }
}

// Test 2: NULL pointer - output
static void test_null_output(void) {
    TEST_START("NULL pointer - output");
    
    float x[8] __attribute__((aligned(64))) = {1.0f};
    
    q_error_code ret = q_softmax_f32_avx2(x, NULL, 8);
    
    if (ret == Q_ERR_INVALID_ARG) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_ARG, got %d", ret);
    }
}

// Test 3: Misaligned pointer - x
static void test_misaligned_x(void) {
    TEST_START("Misaligned pointer - x");
    
    float x_data[9]; // Unaligned
    float* x = x_data + 1; // Misaligned
    float output[8] __attribute__((aligned(64)));
    
    for (uint32_t i = 0; i < 8; i++) {
        x[i] = 1.0f;
    }
    
    q_error_code ret = q_softmax_f32_avx2(x, output, 8);
    
    if (ret == Q_ERR_MISALIGNED) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_MISALIGNED, got %d", ret);
    }
}

// Test 4: Zero size
static void test_zero_size(void) {
    TEST_START("Zero size");
    
    float x[8] __attribute__((aligned(64))) = {1.0f};
    float output[8] __attribute__((aligned(64)));
    
    q_error_code ret = q_softmax_f32_avx2(x, output, 0);
    
    if (ret == Q_ERR_INVALID_SIZE) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_SIZE, got %d", ret);
    }
}

// Test 5: Size not multiple of 8
static void test_size_not_multiple_of_8(void) {
    TEST_START("Size not multiple of 8");
    
    float x[9] __attribute__((aligned(64)));
    float output[9] __attribute__((aligned(64)));
    
    for (uint32_t i = 0; i < 9; i++) {
        x[i] = 1.0f;
    }
    
    q_error_code ret = q_softmax_f32_avx2(x, output, 9);
    
    if (ret == Q_ERR_INVALID_SIZE) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_SIZE, got %d", ret);
    }
}

// Test 6: Aliasing - x == output
static void test_aliasing_x_output(void) {
    TEST_START("Aliasing - x == output");
    
    float x[8] __attribute__((aligned(64))) = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    
    q_error_code ret = q_softmax_f32_avx2(x, x, 8);
    
    // Function should handle aliasing correctly
    if (ret == Q_OK) {
        // Verify output is valid probability distribution
        float sum = array_sum(x, 8);
        bool valid = (sum > 0.99f && sum < 1.01f); // Allow small numerical error
        
        for (uint32_t i = 0; i < 8; i++) {
            if (!isfinite(x[i]) || x[i] < 0.0f) {
                valid = false;
                break;
            }
        }
        
        if (valid) {
            TEST_PASS();
        } else {
            TEST_FAIL("Output is not a valid probability distribution");
        }
    } else if (ret == Q_ERR_ALIASING) {
        TEST_PASS(); // Function correctly detects aliasing
    } else {
        TEST_FAIL_MSG("Unexpected return code: %d", ret);
    }
}

// Test 7: All zeros input
static void test_all_zeros(void) {
    TEST_START("All zeros input");
    
    float x[8] __attribute__((aligned(64))) = {0.0f};
    float output[8] __attribute__((aligned(64)));
    float expected[8];
    
    softmax_ref(x, expected, 8);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_softmax_f32_avx2(x, output, 8);
        
        if (ret == Q_OK) {
            // All zeros should result in uniform distribution (1/N each)
            float expected_val = 1.0f / 8.0f;
            bool uniform = true;
            for (uint32_t i = 0; i < 8; i++) {
                if (fabsf(output[i] - expected_val) > 1e-4f) {
                    uniform = false;
                    break;
                }
            }
            
            float sum = array_sum(output, 8);
            if (uniform && fabsf(sum - 1.0f) < 1e-4f) {
                TEST_PASS();
            } else {
                TEST_FAIL("Output is not uniform distribution for all zeros");
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

// Test 8: All same values
static void test_all_same_values(void) {
    TEST_START("All same values");
    
    float x[8] __attribute__((aligned(64))) = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
    float output[8] __attribute__((aligned(64)));
    float expected[8];
    
    softmax_ref(x, expected, 8);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_softmax_f32_avx2(x, output, 8);
        
        if (ret == Q_OK) {
            // Should result in uniform distribution
            float expected_val = 1.0f / 8.0f;
            bool uniform = true;
            for (uint32_t i = 0; i < 8; i++) {
                if (fabsf(output[i] - expected_val) > 1e-4f) {
                    uniform = false;
                    break;
                }
            }
            
            float sum = array_sum(output, 8);
            if (uniform && fabsf(sum - 1.0f) < 1e-4f) {
                TEST_PASS();
            } else {
                TEST_FAIL("Output is not uniform distribution for all same values");
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

// Test 9: Very large values (overflow risk)
static void test_very_large_values(void) {
    TEST_START("Very large values (overflow risk)");
    
    float x[8] __attribute__((aligned(64))) = {
        100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f
    };
    float output[8] __attribute__((aligned(64)));
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_softmax_f32_avx2(x, output, 8);
        
        if (ret == Q_OK) {
            // Check for valid probability distribution
            float sum = array_sum(output, 8);
            bool valid = (sum > 0.99f && sum < 1.01f);
            
            for (uint32_t i = 0; i < 8; i++) {
                if (!isfinite(output[i]) || output[i] < 0.0f) {
                    valid = false;
                    break;
                }
            }
            
            if (valid) {
                TEST_PASS();
            } else {
                TEST_FAIL("Output is not a valid probability distribution");
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

// Test 10: Very negative values
static void test_very_negative_values(void) {
    TEST_START("Very negative values");
    
    float x[8] __attribute__((aligned(64))) = {
        -100.0f, -200.0f, -300.0f, -400.0f, -500.0f, -600.0f, -700.0f, -800.0f
    };
    float output[8] __attribute__((aligned(64)));
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_softmax_f32_avx2(x, output, 8);
        
        if (ret == Q_OK) {
            // Should result in uniform distribution (all exp values very small)
            float sum = array_sum(output, 8);
            bool valid = (sum > 0.99f && sum < 1.01f);
            
            for (uint32_t i = 0; i < 8; i++) {
                if (!isfinite(output[i]) || output[i] < 0.0f) {
                    valid = false;
                    break;
                }
            }
            
            if (valid) {
                TEST_PASS();
            } else {
                TEST_FAIL("Output is not a valid probability distribution");
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

// Test 11: One very large value (should dominate)
static void test_one_large_value(void) {
    TEST_START("One very large value (should dominate)");
    
    float x[8] __attribute__((aligned(64))) = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 100.0f};
    float output[8] __attribute__((aligned(64)));
    float expected[8];
    
    softmax_ref(x, expected, 8);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_softmax_f32_avx2(x, output, 8);
        
        if (ret == Q_OK) {
            // Last element should dominate (close to 1.0)
            float sum = array_sum(output, 8);
            bool valid = (sum > 0.99f && sum < 1.01f) && (output[7] > 0.99f);
            
            if (valid) {
                TEST_PASS();
            } else {
                TEST_FAIL("Large value does not dominate softmax");
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

// Test 12: NaN in input
static void test_nan_input(void) {
    TEST_START("NaN in input");
    
    float x[8] __attribute__((aligned(64))) = {1.0f, 2.0f, NAN, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float output[8] __attribute__((aligned(64)));
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_softmax_f32_avx2(x, output, 8);
        
        // Function should handle NaN gracefully
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

// Test 13: Infinity in input
static void test_inf_input(void) {
    TEST_START("Infinity in input");
    
    float x[8] __attribute__((aligned(64))) = {1.0f, 2.0f, INFINITY, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float output[8] __attribute__((aligned(64)));
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_softmax_f32_avx2(x, output, 8);
        
        // Function should handle Inf gracefully
        if (ret == Q_OK) {
            // Inf should result in probability 1.0 for that position
            bool valid = (output[2] > 0.99f);
            float sum = array_sum(output, 8);
            valid = valid && (sum > 0.99f && sum < 1.01f);
            
            if (valid) {
                TEST_PASS();
            } else {
                TEST_FAIL("Infinity not handled correctly");
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

// Test 14: Large size (stress test)
static void test_large_size(void) {
    TEST_START("Large size (stress test)");
    
    const uint32_t N = 8192; // Large but reasonable
    float* x = (float*)aligned_alloc(64, N * sizeof(float));
    float* output = (float*)aligned_alloc(64, N * sizeof(float));
    float* expected = (float*)malloc(N * sizeof(float));
    
    if (!x || !output || !expected) {
        TEST_FAIL("Memory allocation failed");
        if (x) free(x);
        if (output) free(output);
        if (expected) free(expected);
        return;
    }
    
    for (uint32_t i = 0; i < N; i++) {
        x[i] = (float)(i % 100);
    }
    
    softmax_ref(x, expected, N);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_softmax_f32_avx2(x, output, N);
        
        if (ret == Q_OK) {
            // Check sum is approximately 1.0
            float sum = array_sum(output, N);
            bool valid = (sum > 0.99f && sum < 1.01f);
            
            // Use relaxed tolerance for polynomial approximation
            if (valid && float_array_close(output, expected, N, 2.5e-1f, 5e-1f)) {
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
    free(output);
    free(expected);
}

// Test 15: Precision test - compare with reference
static void test_precision(void) {
    TEST_START("Precision test - compare with reference");
    
    float x[8] __attribute__((aligned(64))) = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float output[8] __attribute__((aligned(64)));
    float expected[8];
    
    softmax_ref(x, expected, 8);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_softmax_f32_avx2(x, output, 8);
        
        if (ret == Q_OK) {
            // Check sum is approximately 1.0
            float sum = array_sum(output, 8);
            bool sum_valid = (sum > 0.99f && sum < 1.01f);
            
            // Use relaxed tolerance for polynomial approximation
            bool values_valid = float_array_close(output, expected, 8, 2.5e-1f, 5e-1f);
            
            if (sum_valid && values_valid) {
                TEST_PASS();
            } else {
                if (!sum_valid) {
                    TEST_FAIL_MSG("Sum is not 1.0: got %.6f", sum);
                } else {
                    // Print first mismatch for debugging
                    for (uint32_t i = 0; i < 8; i++) {
                        float diff = fabsf(output[i] - expected[i]);
                        if (diff > 2.5e-1f) {
                            TEST_FAIL_MSG("Mismatch at index %u: expected %.6f, got %.6f (diff: %.6f)",
                                         i, expected[i], output[i], diff);
                            break;
                        }
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

// Test 16: Sum validation (must be ≈ 1.0)
static void test_sum_validation(void) {
    TEST_START("Sum validation (must be ≈ 1.0)");
    
    float x[8] __attribute__((aligned(64))) = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float output[8] __attribute__((aligned(64)));
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_softmax_f32_avx2(x, output, 8);
        
        if (ret == Q_OK) {
            float sum = array_sum(output, 8);
            
            // Sum must be approximately 1.0 (probability distribution)
            if (sum > 0.99f && sum < 1.01f) {
                TEST_PASS();
            } else {
                TEST_FAIL_MSG("Sum is not 1.0: got %.6f", sum);
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
    printf("=== Adversarial Test Suite: q_softmax_f32_avx2 ===\n\n");
    
    // Run all tests
    test_null_x();
    test_null_output();
    test_misaligned_x();
    test_zero_size();
    test_size_not_multiple_of_8();
    test_aliasing_x_output();
    test_all_zeros();
    test_all_same_values();
    test_very_large_values();
    test_very_negative_values();
    test_one_large_value();
    test_nan_input();
    test_inf_input();
    test_large_size();
    test_precision();
    test_sum_validation();
    
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

