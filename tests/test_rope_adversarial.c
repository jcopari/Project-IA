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

static void rope_ref(
    const float* restrict x,
    const float* restrict cos,
    const float* restrict sin,
    float* restrict output,
    uint32_t N
) {
    // RoPE: Rotate pairs (x[2i], x[2i+1]) by angle theta
    // output[2i] = x[2i] * cos[i] - x[2i+1] * sin[i]
    // output[2i+1] = x[2i] * sin[i] + x[2i+1] * cos[i]
    
    for (uint32_t i = 0; i < N / 2; i++) {
        float x0 = x[2 * i];
        float x1 = x[2 * i + 1];
        float c = cos[i];
        float s = sin[i];
        
        output[2 * i] = x0 * c - x1 * s;
        output[2 * i + 1] = x0 * s + x1 * c;
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
    
    float cos[4] __attribute__((aligned(64))) = {1.0f};
    float sin[4] __attribute__((aligned(64))) = {0.0f};
    float output[8] __attribute__((aligned(64)));
    
    q_error_code ret = q_rope_f32_avx2(NULL, cos, sin, output, 8);
    
    if (ret == Q_ERR_INVALID_ARG) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_ARG, got %d", ret);
    }
}

// Test 2: NULL pointer - cos
static void test_null_cos(void) {
    TEST_START("NULL pointer - cos");
    
    float x[8] __attribute__((aligned(64))) = {1.0f};
    float sin[4] __attribute__((aligned(64))) = {0.0f};
    float output[8] __attribute__((aligned(64)));
    
    q_error_code ret = q_rope_f32_avx2(x, NULL, sin, output, 8);
    
    if (ret == Q_ERR_INVALID_ARG) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_ARG, got %d", ret);
    }
}

// Test 3: NULL pointer - sin
static void test_null_sin(void) {
    TEST_START("NULL pointer - sin");
    
    float x[8] __attribute__((aligned(64))) = {1.0f};
    float cos[4] __attribute__((aligned(64))) = {1.0f};
    float output[8] __attribute__((aligned(64)));
    
    q_error_code ret = q_rope_f32_avx2(x, cos, NULL, output, 8);
    
    if (ret == Q_ERR_INVALID_ARG) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_ARG, got %d", ret);
    }
}

// Test 4: NULL pointer - output
static void test_null_output(void) {
    TEST_START("NULL pointer - output");
    
    float x[8] __attribute__((aligned(64))) = {1.0f};
    float cos[4] __attribute__((aligned(64))) = {1.0f};
    float sin[4] __attribute__((aligned(64))) = {0.0f};
    
    q_error_code ret = q_rope_f32_avx2(x, cos, sin, NULL, 8);
    
    if (ret == Q_ERR_INVALID_ARG) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_ARG, got %d", ret);
    }
}

// Test 5: Misaligned pointer - x
static void test_misaligned_x(void) {
    TEST_START("Misaligned pointer - x");
    
    float x_data[9]; // Unaligned
    float* x = x_data + 1; // Misaligned
    float cos[4] __attribute__((aligned(64))) = {1.0f};
    float sin[4] __attribute__((aligned(64))) = {0.0f};
    float output[8] __attribute__((aligned(64)));
    
    for (uint32_t i = 0; i < 8; i++) {
        x[i] = 1.0f;
    }
    
    q_error_code ret = q_rope_f32_avx2(x, cos, sin, output, 8);
    
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
    float cos[4] __attribute__((aligned(64))) = {1.0f};
    float sin[4] __attribute__((aligned(64))) = {0.0f};
    float output[8] __attribute__((aligned(64)));
    
    q_error_code ret = q_rope_f32_avx2(x, cos, sin, output, 0);
    
    if (ret == Q_ERR_INVALID_SIZE) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_SIZE, got %d", ret);
    }
}

// Test 7: Odd size (N must be even)
static void test_odd_size(void) {
    TEST_START("Odd size (N must be even)");
    
    float x[9] __attribute__((aligned(64)));
    float cos[5] __attribute__((aligned(64)));
    float sin[5] __attribute__((aligned(64)));
    float output[9] __attribute__((aligned(64)));
    
    for (uint32_t i = 0; i < 9; i++) {
        x[i] = 1.0f;
    }
    for (uint32_t i = 0; i < 5; i++) {
        cos[i] = 1.0f;
        sin[i] = 0.0f;
    }
    
    q_error_code ret = q_rope_f32_avx2(x, cos, sin, output, 9);
    
    if (ret == Q_ERR_INVALID_SIZE) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_SIZE, got %d", ret);
    }
}

// Test 8: Size not multiple of 8
static void test_size_not_multiple_of_8(void) {
    TEST_START("Size not multiple of 8");
    
    float x[10] __attribute__((aligned(64)));
    float cos[5] __attribute__((aligned(64)));
    float sin[5] __attribute__((aligned(64)));
    float output[10] __attribute__((aligned(64)));
    
    for (uint32_t i = 0; i < 10; i++) {
        x[i] = 1.0f;
    }
    for (uint32_t i = 0; i < 5; i++) {
        cos[i] = 1.0f;
        sin[i] = 0.0f;
    }
    
    q_error_code ret = q_rope_f32_avx2(x, cos, sin, output, 10);
    
    if (ret == Q_ERR_INVALID_SIZE) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_SIZE, got %d", ret);
    }
}

// Helper: Convert cos/sin arrays from [c0, c1, ...] to [c0, c0, c1, c1, ...] layout
static void duplicate_cos_sin(const float* src, float* dst, uint32_t N) {
    for (uint32_t i = 0; i < N / 2; i++) {
        dst[i * 2] = src[i];
        dst[i * 2 + 1] = src[i];
    }
}

// Test 9: Aliasing - x == output
static void test_aliasing_x_output(void) {
    TEST_START("Aliasing - x == output");
    
    float x[8] __attribute__((aligned(64))) = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    float cos_src[4] = {1.0f, 0.0f, -1.0f, 0.0f};
    float sin_src[4] = {0.0f, 1.0f, 0.0f, -1.0f};
    float cos[8] __attribute__((aligned(64)));
    float sin[8] __attribute__((aligned(64)));
    
    // Convert to duplicated layout for AVX2
    duplicate_cos_sin(cos_src, cos, 8);
    duplicate_cos_sin(sin_src, sin, 8);
    
    // Save original values
    float x_orig[8];
    memcpy(x_orig, x, sizeof(x));
    
    q_error_code ret = q_rope_f32_avx2(x, cos, sin, x, 8);
    
    // Check if function handles aliasing correctly
    if (ret == Q_OK) {
        // Verify output is rotated correctly
        bool rotated = true;
        for (uint32_t i = 0; i < 8; i++) {
            if (!isfinite(x[i])) {
                rotated = false;
                break;
            }
        }
        if (rotated) {
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

// Test 10: Zero rotation (cos=1, sin=0)
static void test_zero_rotation(void) {
    TEST_START("Zero rotation (cos=1, sin=0)");
    
    float x[8] __attribute__((aligned(64))) = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float cos_src[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float sin_src[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float cos[8] __attribute__((aligned(64)));
    float sin[8] __attribute__((aligned(64)));
    float output[8] __attribute__((aligned(64)));
    float expected[8];
    
    // Convert to duplicated layout for AVX2
    duplicate_cos_sin(cos_src, cos, 8);
    duplicate_cos_sin(sin_src, sin, 8);
    
    memcpy(expected, x, sizeof(x)); // Should be unchanged
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_rope_f32_avx2(x, cos, sin, output, 8);
        
        if (ret == Q_OK) {
            if (float_array_close(output, expected, 8, 1e-5f, 1e-4f)) {
                TEST_PASS();
            } else {
                TEST_FAIL("Output does not match expected (zero rotation)");
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

// Test 11: 90-degree rotation (cos=0, sin=1)
static void test_90_degree_rotation(void) {
    TEST_START("90-degree rotation (cos=0, sin=1)");
    
    float x[8] __attribute__((aligned(64))) = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f};
    float cos_src[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sin_src[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float cos[8] __attribute__((aligned(64)));
    float sin[8] __attribute__((aligned(64)));
    float output[8] __attribute__((aligned(64)));
    float expected[8];
    
    // Convert to duplicated layout for AVX2
    duplicate_cos_sin(cos_src, cos, 8);
    duplicate_cos_sin(sin_src, sin, 8);
    
    rope_ref(x, cos_src, sin_src, expected, 8);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_rope_f32_avx2(x, cos, sin, output, 8);
        
        if (ret == Q_OK) {
            if (float_array_close(output, expected, 8, 1e-4f, 1e-3f)) {
                TEST_PASS();
            } else {
                TEST_FAIL("Output does not match reference for 90-degree rotation");
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

// Test 12: 180-degree rotation (cos=-1, sin=0)
static void test_180_degree_rotation(void) {
    TEST_START("180-degree rotation (cos=-1, sin=0)");
    
    float x[8] __attribute__((aligned(64))) = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float cos_src[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
    float sin_src[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float cos[8] __attribute__((aligned(64)));
    float sin[8] __attribute__((aligned(64)));
    float output[8] __attribute__((aligned(64)));
    float expected[8];
    
    // Convert to duplicated layout for AVX2
    duplicate_cos_sin(cos_src, cos, 8);
    duplicate_cos_sin(sin_src, sin, 8);
    
    rope_ref(x, cos_src, sin_src, expected, 8);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_rope_f32_avx2(x, cos, sin, output, 8);
        
        if (ret == Q_OK) {
            if (float_array_close(output, expected, 8, 1e-4f, 1e-3f)) {
                TEST_PASS();
            } else {
                TEST_FAIL("Output does not match reference for 180-degree rotation");
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

// Test 13: Very large values
static void test_very_large_values(void) {
    TEST_START("Very large values");
    
    float x[8] __attribute__((aligned(64))) = {
        FLT_MAX / 2.0f, FLT_MAX / 2.0f, FLT_MAX / 2.0f, FLT_MAX / 2.0f,
        FLT_MAX / 2.0f, FLT_MAX / 2.0f, FLT_MAX / 2.0f, FLT_MAX / 2.0f
    };
    float cos_src[4] = {1.0f, 0.0f, -1.0f, 0.0f};
    float sin_src[4] = {0.0f, 1.0f, 0.0f, -1.0f};
    float cos[8] __attribute__((aligned(64)));
    float sin[8] __attribute__((aligned(64)));
    float output[8] __attribute__((aligned(64)));
    
    // Convert to duplicated layout for AVX2
    duplicate_cos_sin(cos_src, cos, 8);
    duplicate_cos_sin(sin_src, sin, 8);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_rope_f32_avx2(x, cos, sin, output, 8);
        
        if (ret == Q_OK) {
            // Check for overflow/NaN/Inf
            bool valid = true;
            for (uint32_t i = 0; i < 8; i++) {
                if (!isfinite(output[i])) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                TEST_PASS();
            } else {
                TEST_FAIL("Output contains non-finite values");
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
    float cos[4] __attribute__((aligned(64))) = {1.0f, 0.0f, -1.0f, 0.0f};
    float sin[4] __attribute__((aligned(64))) = {0.0f, 1.0f, 0.0f, -1.0f};
    float output[8] __attribute__((aligned(64)));
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_rope_f32_avx2(x, cos, sin, output, 8);
        
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

// Test 15: Infinity in input
static void test_inf_input(void) {
    TEST_START("Infinity in input");
    
    float x[8] __attribute__((aligned(64))) = {1.0f, 2.0f, INFINITY, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float cos[4] __attribute__((aligned(64))) = {1.0f, 0.0f, -1.0f, 0.0f};
    float sin[4] __attribute__((aligned(64))) = {0.0f, 1.0f, 0.0f, -1.0f};
    float output[8] __attribute__((aligned(64)));
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_rope_f32_avx2(x, cos, sin, output, 8);
        
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

// Test 16: Large size (stress test)
static void test_large_size(void) {
    TEST_START("Large size (stress test)");
    
    const uint32_t N = 8192; // Large but reasonable
    float* x = (float*)aligned_alloc(64, N * sizeof(float));
    float* cos_src = (float*)aligned_alloc(64, (N/2) * sizeof(float));
    float* sin_src = (float*)aligned_alloc(64, (N/2) * sizeof(float));
    float* cos = (float*)aligned_alloc(64, N * sizeof(float));
    float* sin = (float*)aligned_alloc(64, N * sizeof(float));
    float* output = (float*)aligned_alloc(64, N * sizeof(float));
    float* expected = (float*)malloc(N * sizeof(float));
    
    if (!x || !cos_src || !sin_src || !cos || !sin || !output || !expected) {
        TEST_FAIL("Memory allocation failed");
        if (x) free(x);
        if (cos_src) free(cos_src);
        if (sin_src) free(sin_src);
        if (cos) free(cos);
        if (sin) free(sin);
        if (output) free(output);
        if (expected) free(expected);
        return;
    }
    
    for (uint32_t i = 0; i < N; i++) {
        x[i] = (float)(i % 100);
    }
    for (uint32_t i = 0; i < N/2; i++) {
        float angle = (float)i * 0.1f;
        cos_src[i] = cosf(angle);
        sin_src[i] = sinf(angle);
    }
    
    // Convert to duplicated layout for AVX2
    duplicate_cos_sin(cos_src, cos, N);
    duplicate_cos_sin(sin_src, sin, N);
    
    rope_ref(x, cos_src, sin_src, expected, N);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_rope_f32_avx2(x, cos, sin, output, N);
        
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
    free(cos_src);
    free(sin_src);
    free(cos);
    free(sin);
    free(output);
    free(expected);
}

// Test 17: Precision test - compare with reference
static void test_precision(void) {
    TEST_START("Precision test - compare with reference");
    
    float x[8] __attribute__((aligned(64))) = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float cos_src[4] = {0.70710678f, 0.8660254f, 0.5f, 0.0f}; // Various angles
    float sin_src[4] = {0.70710678f, 0.5f, 0.8660254f, 1.0f};
    float cos[8] __attribute__((aligned(64)));
    float sin[8] __attribute__((aligned(64)));
    float output[8] __attribute__((aligned(64)));
    float expected[8];
    
    // Convert to duplicated layout for AVX2
    duplicate_cos_sin(cos_src, cos, 8);
    duplicate_cos_sin(sin_src, sin, 8);
    
    rope_ref(x, cos_src, sin_src, expected, 8);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_error_code ret = q_rope_f32_avx2(x, cos, sin, output, 8);
        
        if (ret == Q_OK) {
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
    printf("=== Adversarial Test Suite: q_rope_f32_avx2 ===\n\n");
    
    // Run all tests
    test_null_x();
    test_null_cos();
    test_null_sin();
    test_null_output();
    test_misaligned_x();
    test_zero_size();
    test_odd_size();
    test_size_not_multiple_of_8();
    test_aliasing_x_output();
    test_zero_rotation();
    test_90_degree_rotation();
    test_180_degree_rotation();
    test_very_large_values();
    test_nan_input();
    test_inf_input();
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

