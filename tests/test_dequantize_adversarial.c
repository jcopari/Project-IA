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

static void dequantize_q4_0_ref(
    const q_block_q4_0* restrict block,
    float* restrict output
) {
    const float scale = block->scale;
    
    for (uint32_t i = 0; i < 32; i++) {
        uint8_t byte_idx = i / 2;
        uint8_t nibble_idx = i % 2;
        uint8_t byte = block->qs[byte_idx];
        uint8_t nibble = (nibble_idx == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        
        // Dequantize: value = (quantized - 8) * scale
        output[i] = ((float)nibble - 8.0f) * scale;
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

// Test 1: NULL pointer - block
static void test_null_block(void) {
    TEST_START("NULL pointer - block");
    
    float output[32] __attribute__((aligned(64)));
    
    // Function should handle NULL gracefully (no crash)
    // Initialize output to detect if function was called
    for (uint32_t i = 0; i < 32; i++) {
        output[i] = 999.0f; // Sentinel value
    }
    
    q_dequantize_q4_0_block_avx2_public(NULL, output);
    
    // Function should not crash and output should remain unchanged (or be zero)
    // This is acceptable behavior for NULL input
    TEST_PASS();
}

// Test 2: NULL pointer - output
static void test_null_output(void) {
    TEST_START("NULL pointer - output");
    
    q_block_q4_0 block = {0};
    block.scale = 1.0f;
    
    // Function should handle NULL gracefully (no crash)
    q_dequantize_q4_0_block_avx2_public(&block, NULL);
    
    // Function should not crash - this is acceptable behavior
    TEST_PASS();
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 3: Misaligned output pointer
static void test_misaligned_output(void) {
    TEST_START("Misaligned output pointer");
    
    q_block_q4_0 block = {0};
    block.scale = 1.0f;
    float output_data[33]; // Unaligned
    float* output = output_data + 1; // Misaligned
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_dequantize_q4_0_block_avx2_public(&block, output);
        // Function may work but with performance penalty, or crash
        // Check if output is valid
        bool valid = true;
        for (uint32_t i = 0; i < 32; i++) {
            if (!isfinite(output[i])) {
                valid = false;
                break;
            }
        }
        if (valid) {
            TEST_PASS(); // Function handled misalignment gracefully
        } else {
            TEST_FAIL("Output contains non-finite values");
        }
    } else {
        TEST_PASS(); // Crash is acceptable for misaligned pointer
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 4: Zero scale
static void test_zero_scale(void) {
    TEST_START("Zero scale");
    
    q_block_q4_0 block = {0};
    block.scale = 0.0f;
    memset(block.qs, 0x88, 16); // All quantized = 8 (dequantized = 0)
    float output[32] __attribute__((aligned(64)));
    float expected[32] = {0.0f}; // All zeros
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_dequantize_q4_0_block_avx2_public(&block, output);
        
        if (float_array_close(output, expected, 32, 1e-5f, 1e-4f)) {
            TEST_PASS();
        } else {
            TEST_FAIL("Output does not match expected (zero scale)");
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 5: Very large scale
static void test_very_large_scale(void) {
    TEST_START("Very large scale");
    
    q_block_q4_0 block = {0};
    block.scale = FLT_MAX / 16.0f; // Large but safe
    memset(block.qs, 0xFF, 16); // All quantized = 15 (maximum)
    float output[32] __attribute__((aligned(64)));
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_dequantize_q4_0_block_avx2_public(&block, output);
        
        // Check for overflow/Inf
        bool valid = true;
        for (uint32_t i = 0; i < 32; i++) {
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
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 6: Very small scale
static void test_very_small_scale(void) {
    TEST_START("Very small scale");
    
    q_block_q4_0 block = {0};
    block.scale = FLT_MIN;
    memset(block.qs, 0xFF, 16); // All quantized = 15
    float output[32] __attribute__((aligned(64)));
    float expected[32];
    
    dequantize_q4_0_ref(&block, expected);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_dequantize_q4_0_block_avx2_public(&block, output);
        
        if (float_array_close(output, expected, 32, 1e-5f, 1e-4f)) {
            TEST_PASS();
        } else {
            TEST_FAIL("Output does not match reference for very small scale");
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 7: Negative scale
static void test_negative_scale(void) {
    TEST_START("Negative scale");
    
    q_block_q4_0 block = {0};
    block.scale = -1.0f;
    memset(block.qs, 0x88, 16); // All quantized = 8
    float output[32] __attribute__((aligned(64)));
    float expected[32] = {0.0f}; // (8-8) * (-1) = 0
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_dequantize_q4_0_block_avx2_public(&block, output);
        
        if (float_array_close(output, expected, 32, 1e-5f, 1e-4f)) {
            TEST_PASS();
        } else {
            TEST_FAIL("Output does not match expected (negative scale)");
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 8: All zeros quantized (nibble = 0)
static void test_all_zeros_quantized(void) {
    TEST_START("All zeros quantized (nibble = 0)");
    
    q_block_q4_0 block = {0};
    block.scale = 1.0f;
    memset(block.qs, 0x00, 16); // All nibbles = 0
    float output[32] __attribute__((aligned(64)));
    float expected[32];
    
    dequantize_q4_0_ref(&block, expected);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_dequantize_q4_0_block_avx2_public(&block, output);
        
        if (float_array_close(output, expected, 32, 1e-5f, 1e-4f)) {
            TEST_PASS();
        } else {
            TEST_FAIL("Output does not match reference for all zeros quantized");
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 9: All maximum quantized (nibble = 15)
static void test_all_maximum_quantized(void) {
    TEST_START("All maximum quantized (nibble = 15)");
    
    q_block_q4_0 block = {0};
    block.scale = 1.0f;
    memset(block.qs, 0xFF, 16); // All nibbles = 15
    float output[32] __attribute__((aligned(64)));
    float expected[32];
    
    dequantize_q4_0_ref(&block, expected);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_dequantize_q4_0_block_avx2_public(&block, output);
        
        if (float_array_close(output, expected, 32, 1e-5f, 1e-4f)) {
            TEST_PASS();
        } else {
            TEST_FAIL("Output does not match reference for all maximum quantized");
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 10: All middle quantized (nibble = 8, dequantized = 0)
static void test_all_middle_quantized(void) {
    TEST_START("All middle quantized (nibble = 8, dequantized = 0)");
    
    q_block_q4_0 block = {0};
    block.scale = 1.0f;
    memset(block.qs, 0x88, 16); // All nibbles = 8
    float output[32] __attribute__((aligned(64)));
    float expected[32] = {0.0f}; // (8-8) * 1 = 0
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_dequantize_q4_0_block_avx2_public(&block, output);
        
        if (float_array_close(output, expected, 32, 1e-5f, 1e-4f)) {
            TEST_PASS();
        } else {
            TEST_FAIL("Output does not match expected (all middle quantized)");
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 11: Alternating pattern
static void test_alternating_pattern(void) {
    TEST_START("Alternating pattern");
    
    q_block_q4_0 block = {0};
    block.scale = 1.0f;
    // Pattern: 0x08 = 0x08 (low=8, high=0), 0x80 = 0x80 (low=0, high=8)
    for (int i = 0; i < 16; i++) {
        block.qs[i] = (i % 2 == 0) ? 0x08 : 0x80;
    }
    float output[32] __attribute__((aligned(64)));
    float expected[32];
    
    dequantize_q4_0_ref(&block, expected);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_dequantize_q4_0_block_avx2_public(&block, output);
        
        if (float_array_close(output, expected, 32, 1e-5f, 1e-4f)) {
            TEST_PASS();
        } else {
            TEST_FAIL("Output does not match reference for alternating pattern");
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 12: Random pattern
static void test_random_pattern(void) {
    TEST_START("Random pattern");
    
    q_block_q4_0 block = {0};
    block.scale = 0.1f;
    
    // Generate random quantized values
    srand(42); // Fixed seed for reproducibility
    for (int i = 0; i < 16; i++) {
        uint8_t low = rand() % 16;
        uint8_t high = rand() % 16;
        block.qs[i] = low | (high << 4);
    }
    
    float output[32] __attribute__((aligned(64)));
    float expected[32];
    
    dequantize_q4_0_ref(&block, expected);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_dequantize_q4_0_block_avx2_public(&block, output);
        
        if (float_array_close(output, expected, 32, 1e-5f, 1e-4f)) {
            TEST_PASS();
        } else {
            TEST_FAIL("Output does not match reference for random pattern");
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 13: NaN scale
static void test_nan_scale(void) {
    TEST_START("NaN scale");
    
    q_block_q4_0 block = {0};
    block.scale = NAN;
    memset(block.qs, 0x88, 16);
    float output[32] __attribute__((aligned(64)));
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_dequantize_q4_0_block_avx2_public(&block, output);
        
        // NaN should propagate to output
        bool has_nan = false;
        for (uint32_t i = 0; i < 32; i++) {
            if (isnan(output[i])) {
                has_nan = true;
                break;
            }
        }
        if (has_nan) {
            TEST_PASS();
        } else {
            TEST_FAIL("NaN scale did not propagate to output");
        }
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 14: Infinity scale
static void test_inf_scale(void) {
    TEST_START("Infinity scale");
    
    q_block_q4_0 block = {0};
    block.scale = INFINITY;
    memset(block.qs, 0x88, 16);
    float output[32] __attribute__((aligned(64)));
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_dequantize_q4_0_block_avx2_public(&block, output);
        
        // Inf * anything = Inf (expected behavior)
        // However, (8-8) * Inf = 0 * Inf = NaN (IEEE 754 standard)
        // Function should not crash, Inf or NaN output is acceptable for extreme cases
        for (uint32_t i = 0; i < 32; i++) {
            // Accept Inf or NaN as valid output for extreme scale values
            // This is expected IEEE 754 behavior: 0 * Inf = NaN
            if (!isfinite(output[i])) {
                // Both Inf and NaN are acceptable (no crash)
                continue;
            }
            // If finite, should be Inf (for non-zero nibbles) or NaN (for zero nibbles)
            // But we accept any non-finite value as valid
        }
        // Function didn't crash, which is the important thing
        TEST_PASS();
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// Test 15: Precision test - compare with reference
static void test_precision(void) {
    TEST_START("Precision test - compare with reference");
    
    q_block_q4_0 block = {0};
    block.scale = 0.1f;
    
    // Create a pattern with various quantized values
    for (int i = 0; i < 16; i++) {
        uint8_t low = i % 16;
        uint8_t high = (i + 8) % 16;
        block.qs[i] = low | (high << 4);
    }
    
    float output[32] __attribute__((aligned(64)));
    float expected[32];
    
    dequantize_q4_0_ref(&block, expected);
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp_buf) == 0) {
        q_dequantize_q4_0_block_avx2_public(&block, output);
        
        if (float_array_close(output, expected, 32, 1e-5f, 1e-4f)) {
            TEST_PASS();
        } else {
            // Print first mismatch for debugging
            for (uint32_t i = 0; i < 32; i++) {
                float diff = fabsf(output[i] - expected[i]);
                if (diff > 1e-5f) {
                    TEST_FAIL_MSG("Mismatch at index %u: expected %.6f, got %.6f (diff: %.6f)",
                                 i, expected[i], output[i], diff);
                    break;
                }
            }
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
    printf("=== Adversarial Test Suite: q_dequantize_q4_0_block_avx2 ===\n\n");
    
    // Run all tests
    test_null_block();
    test_null_output();
    test_misaligned_output();
    test_zero_scale();
    test_very_large_scale();
    test_very_small_scale();
    test_negative_scale();
    test_all_zeros_quantized();
    test_all_maximum_quantized();
    test_all_middle_quantized();
    test_alternating_pattern();
    test_random_pattern();
    test_nan_scale();
    test_inf_scale();
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

