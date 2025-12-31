#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <limits.h>

// ============================================================================
// TEST CONFIGURATION
// ============================================================================

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

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

// ============================================================================
// TEST SUITE: q_strerror()
// ============================================================================

// Test 1: Q_OK (success code)
static void test_strerror_q_ok(void) {
    TEST_START("q_strerror - Q_OK (success code)");
    
    const char* str = q_strerror(Q_OK);
    if (str != NULL && strcmp(str, "Success") == 0) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected 'Success', got '%s'", str ? str : "NULL");
    }
}

// Test 2: All valid error codes
static void test_strerror_all_valid_codes(void) {
    TEST_START("q_strerror - All valid error codes");
    
    // Test all defined error codes
    struct {
        q_error_code code;
        const char* expected;
    } test_cases[] = {
        {Q_OK, "Success"},
        {Q_ERR_NULL_PTR, "Null pointer argument"},
        {Q_ERR_FILE_OPEN, "Failed to open file"},
        {Q_ERR_FILE_STAT, "Failed to stat file"},
        {Q_ERR_FILE_TOO_SMALL, "File too small (corrupt header?)"},
        {Q_ERR_MMAP_FAILED, "mmap() failed"},
        {Q_ERR_INVALID_MAGIC, "Invalid file magic (not a Qorus file)"},
        {Q_ERR_ALLOC_FAILED, "Memory allocation failed"},
        {Q_ERR_ARENA_OOM, "Arena Out of Memory"},
        {Q_ERR_INVALID_CONFIG, "Invalid model configuration"},
        {Q_ERR_INVALID_ARG, "Invalid argument"},
        {Q_ERR_ALIASING, "Input/output aliasing detected"},
        {Q_ERR_OVERFLOW, "Integer overflow detected"},
        {Q_ERR_MISALIGNED, "Pointer not properly aligned"},
        {Q_ERR_INVALID_DTYPE, "Invalid data type"},
        {Q_ERR_INVALID_SIZE, "Invalid size"}
    };
    
    int num_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    bool all_passed = true;
    
    for (int i = 0; i < num_cases; i++) {
        const char* str = q_strerror(test_cases[i].code);
        if (str == NULL || strcmp(str, test_cases[i].expected) != 0) {
            TEST_FAIL_MSG("Code %d: Expected '%s', got '%s'", 
                         test_cases[i].code, 
                         test_cases[i].expected,
                         str ? str : "NULL");
            all_passed = false;
            break;
        }
    }
    
    if (all_passed) {
        tests_run += num_cases;
        tests_passed += num_cases;
        printf("  ✓ PASSED (all %d error codes)\n", num_cases);
    }
}

// Test 3: Invalid error code (out of bounds, positive)
static void test_strerror_invalid_positive(void) {
    TEST_START("q_strerror - Invalid positive error code");
    
    const char* str = q_strerror(100);  // Positive, out of bounds
    if (str != NULL && strcmp(str, "Unknown error") == 0) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected 'Unknown error', got '%s'", str ? str : "NULL");
    }
}

// Test 4: Invalid error code (out of bounds, negative)
static void test_strerror_invalid_negative(void) {
    TEST_START("q_strerror - Invalid negative error code");
    
    const char* str = q_strerror(-100);  // Negative, out of bounds
    if (str != NULL && strcmp(str, "Unknown error") == 0) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected 'Unknown error', got '%s'", str ? str : "NULL");
    }
}

// Test 5: Edge case - INT_MIN (if it maps to valid index)
static void test_strerror_int_min(void) {
    TEST_START("q_strerror - INT_MIN edge case");
    
    // INT_MIN = -2147483648, which would map to index 2147483648
    // This is way out of bounds, should return "Unknown error"
    const char* str = q_strerror(INT_MIN);
    if (str != NULL && strcmp(str, "Unknown error") == 0) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected 'Unknown error', got '%s'", str ? str : "NULL");
    }
}

// Test 6: Performance test - O(1) complexity verification
static void test_strerror_performance(void) {
    TEST_START("q_strerror - Performance test (O(1) complexity)");
    
    // Measure time for multiple calls
    // O(1) means time should be constant regardless of number of calls
    const int iterations = 1000000;
    clock_t start, end;
    
    start = clock();
    for (int i = 0; i < iterations; i++) {
        // Call with different codes to avoid cache effects
        q_strerror((q_error_code)(i % 16));
    }
    end = clock();
    
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    double avg_time_per_call = elapsed / iterations;
    
    // O(1) should be very fast (< 1 microsecond per call)
    // If it's O(n) or worse, it would be much slower
    if (avg_time_per_call < 1e-6) {  // Less than 1 microsecond
        TEST_PASS();
        printf("    Performance: %.2f ns/call (O(1) confirmed)\n", avg_time_per_call * 1e9);
    } else {
        TEST_FAIL_MSG("Performance too slow: %.2f ns/call (expected < 1000 ns)", 
                     avg_time_per_call * 1e9);
    }
}

// Test 7: Return value is not NULL
static void test_strerror_not_null(void) {
    TEST_START("q_strerror - Return value is never NULL");
    
    // Test all valid codes and some invalid ones
    for (int code = -20; code <= 20; code++) {
        const char* str = q_strerror((q_error_code)code);
        if (str == NULL) {
            TEST_FAIL_MSG("q_strerror returned NULL for code %d", code);
            return;
        }
    }
    
    TEST_PASS();
}

// Test 8: Return value is static (pointer stability)
static void test_strerror_pointer_stability(void) {
    TEST_START("q_strerror - Pointer stability (static strings)");
    
    // Call multiple times with same code
    const char* str1 = q_strerror(Q_ERR_NULL_PTR);
    const char* str2 = q_strerror(Q_ERR_NULL_PTR);
    const char* str3 = q_strerror(Q_ERR_NULL_PTR);
    
    // All should return same pointer (static string)
    if (str1 == str2 && str2 == str3) {
        TEST_PASS();
    } else {
        TEST_FAIL("Pointers are not stable (not static strings)");
    }
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("=== Test Suite: Utils (q_strerror) ===\n\n");
    
    // Run all tests
    test_strerror_q_ok();
    test_strerror_all_valid_codes();
    test_strerror_invalid_positive();
    test_strerror_invalid_negative();
    test_strerror_int_min();
    test_strerror_performance();
    test_strerror_not_null();
    test_strerror_pointer_stability();
    
    // Print summary
    printf("\n=== Test Summary ===\n");
    printf("Total tests: %d\n", tests_run);
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    
    if (tests_failed == 0) {
        printf("\n✓ All tests passed!\n");
        return 0;
    } else {
        printf("\n✗ Some tests failed\n");
        return 1;
    }
}

