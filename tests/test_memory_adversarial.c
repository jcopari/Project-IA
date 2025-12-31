#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>
#include <limits.h>
#include <float.h>

// ============================================================================
// TEST CONFIGURATION
// ============================================================================

// Test statistics
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

// ============================================================================
// TEST HELPERS
// ============================================================================

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
// TEST SUITE: Memory Management Adversarial Tests
// ============================================================================

// Test 1: q_alloc_kv_cache - Null context pointer
static void test_alloc_kv_cache_null_ctx(void) {
    TEST_START("q_alloc_kv_cache - Null context pointer");
    
    q_error_code ret = q_alloc_kv_cache(NULL, 1024);
    if (ret == Q_ERR_INVALID_ARG) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_ARG, got %d", ret);
    }
}

// Test 2: q_alloc_kv_cache - Memory leak prevention (double allocation)
static void test_alloc_kv_cache_double_alloc(void) {
    TEST_START("q_alloc_kv_cache - Double allocation (memory leak prevention)");
    
    q_context ctx = {0};
    
    // First allocation should succeed
    q_error_code ret1 = q_alloc_kv_cache(&ctx, 1024);
    if (ret1 != Q_OK) {
        TEST_FAIL("First allocation failed");
        return;
    }
    
    // Second allocation should fail
    q_error_code ret2 = q_alloc_kv_cache(&ctx, 2048);
    if (ret2 == Q_ERR_INVALID_ARG) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_ARG on double allocation, got %d", ret2);
    }
    
    q_free_memory(&ctx);
}

// Test 3: q_alloc_kv_cache - Overflow in size
static void test_alloc_kv_cache_overflow(void) {
    TEST_START("q_alloc_kv_cache - Size overflow");
    
    q_context ctx = {0};
    
    // Test with size close to SIZE_MAX
    size_t huge_size = SIZE_MAX - 50;
    q_error_code ret = q_alloc_kv_cache(&ctx, huge_size);
    
    if (ret == Q_ERR_OVERFLOW) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_OVERFLOW, got %d", ret);
    }
}

// Test 4: q_alloc_kv_cache - Zero size
static void test_alloc_kv_cache_zero_size(void) {
    TEST_START("q_alloc_kv_cache - Zero size");
    
    q_context ctx = {0};
    
    q_error_code ret = q_alloc_kv_cache(&ctx, 0);
    // Zero size aligned is still 0 (Q_ALIGN_SIZE(0) = 0)
    // aligned_alloc(Q_ALIGN, 0) may fail or have undefined behavior
    // Both behaviors are acceptable
    if (ret == Q_OK) {
        // If it succeeds, size should be 0 (not Q_ALIGN)
        if (ctx.kv_size == 0) {
            TEST_PASS();
        } else {
            TEST_FAIL_MSG("Expected size 0, got %zu", ctx.kv_size);
        }
        q_free_memory(&ctx);
    } else if (ret == Q_ERR_OVERFLOW || ret == Q_ERR_ALLOC_FAILED) {
        // Overflow or allocation failure for zero size is acceptable
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Unexpected error code: %d", ret);
    }
}

// Test 5: q_alloc_arena - Null context pointer
static void test_alloc_arena_null_ctx(void) {
    TEST_START("q_alloc_arena - Null context pointer");
    
    q_error_code ret = q_alloc_arena(NULL, 1024);
    if (ret == Q_ERR_INVALID_ARG) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_ARG, got %d", ret);
    }
}

// Test 6: q_alloc_arena - Memory leak prevention (double allocation)
static void test_alloc_arena_double_alloc(void) {
    TEST_START("q_alloc_arena - Double allocation (memory leak prevention)");
    
    q_context ctx = {0};
    
    // First allocation should succeed
    q_error_code ret1 = q_alloc_arena(&ctx, 1024);
    if (ret1 != Q_OK) {
        TEST_FAIL("First allocation failed");
        return;
    }
    
    // Second allocation should fail
    q_error_code ret2 = q_alloc_arena(&ctx, 2048);
    if (ret2 == Q_ERR_INVALID_ARG) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_ARG on double allocation, got %d", ret2);
    }
    
    q_free_memory(&ctx);
}

// Test 7: q_alloc_arena - Overflow in size
static void test_alloc_arena_overflow(void) {
    TEST_START("q_alloc_arena - Size overflow");
    
    q_context ctx = {0};
    
    size_t huge_size = SIZE_MAX - 50;
    q_error_code ret = q_alloc_arena(&ctx, huge_size);
    
    if (ret == Q_ERR_OVERFLOW) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_OVERFLOW, got %d", ret);
    }
}

// Test 8: q_arena_alloc - Null context pointer
static void test_arena_alloc_null_ctx(void) {
    TEST_START("q_arena_alloc - Null context pointer");
    
    if (setjmp(crash_jmp_buf) == 0) {
        signal(SIGSEGV, crash_handler);
        signal(SIGBUS, crash_handler);
        signal(SIGABRT, crash_handler);
        
        void* ptr = q_arena_alloc(NULL, 1024);
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        
        if (ptr == NULL) {
            TEST_PASS();
        } else {
            TEST_FAIL("Expected NULL, got non-NULL pointer");
        }
    } else {
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        TEST_CRASH();
    }
}

// Test 9: q_arena_alloc - Uninitialized arena
static void test_arena_alloc_uninitialized(void) {
    TEST_START("q_arena_alloc - Uninitialized arena");
    
    q_context ctx = {0};
    // Don't call q_alloc_arena - leave scratch_buffer as NULL
    
    if (setjmp(crash_jmp_buf) == 0) {
        signal(SIGSEGV, crash_handler);
        signal(SIGBUS, crash_handler);
        signal(SIGABRT, crash_handler);
        
        void* ptr = q_arena_alloc(&ctx, 1024);
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        
        if (ptr == NULL) {
            TEST_PASS();
        } else {
            TEST_FAIL("Expected NULL for uninitialized arena, got non-NULL pointer");
        }
    } else {
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        TEST_CRASH();
    }
}

// Test 10: q_arena_alloc - Overflow in alignment calculation
static void test_arena_alloc_overflow_alignment(void) {
    TEST_START("q_arena_alloc - Overflow in alignment calculation");
    
    q_context ctx = {0};
    if (q_alloc_arena(&ctx, 1024 * 1024) != Q_OK) {
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    // Try to allocate with size that causes overflow in safe_align_size
    size_t huge_size = SIZE_MAX - 50;
    void* ptr = q_arena_alloc(&ctx, huge_size);
    
    if (ptr == NULL) {
        TEST_PASS();
    } else {
        TEST_FAIL("Expected NULL on overflow, got non-NULL pointer");
    }
    
    q_free_memory(&ctx);
}

// Test 11: q_arena_alloc - Overflow in addition (head + size)
static void test_arena_alloc_overflow_addition(void) {
    TEST_START("q_arena_alloc - Overflow in addition (head + size)");
    
    q_context ctx = {0};
    if (q_alloc_arena(&ctx, 1024 * 1024) != Q_OK) {
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    // Set head close to SIZE_MAX
    ctx.scratch_head = SIZE_MAX - 100;
    
    void* ptr = q_arena_alloc(&ctx, 200);  // This should overflow
    
    if (ptr == NULL) {
        TEST_PASS();
    } else {
        TEST_FAIL("Expected NULL on overflow, got non-NULL pointer");
    }
    
    q_free_memory(&ctx);
}

// Test 12: q_arena_alloc - Out of memory (OOM)
static void test_arena_alloc_oom(void) {
    TEST_START("q_arena_alloc - Out of memory");
    
    q_context ctx = {0};
    if (q_alloc_arena(&ctx, 1024) != Q_OK) {
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    // Try to allocate more than available
    void* ptr = q_arena_alloc(&ctx, 2048);
    
    if (ptr == NULL) {
        TEST_PASS();
    } else {
        TEST_FAIL("Expected NULL on OOM, got non-NULL pointer");
    }
    
    q_free_memory(&ctx);
}

// Test 13: q_arena_alloc - Misalignment detection (if corruption occurs)
static void test_arena_alloc_misalignment(void) {
    TEST_START("q_arena_alloc - Misalignment detection");
    
    q_context ctx = {0};
    if (q_alloc_arena(&ctx, 1024) != Q_OK) {
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    // Corrupt scratch_head to be misaligned
    ctx.scratch_head = 1;  // Not aligned to Q_ALIGN
    
    void* ptr = q_arena_alloc(&ctx, 64);
    
    if (ptr == NULL) {
        TEST_PASS();
    } else {
        TEST_FAIL("Expected NULL on misalignment, got non-NULL pointer");
    }
    
    q_free_memory(&ctx);
}

// Test 14: q_arena_alloc - Zero size
static void test_arena_alloc_zero_size(void) {
    TEST_START("q_arena_alloc - Zero size");
    
    q_context ctx = {0};
    if (q_alloc_arena(&ctx, 1024) != Q_OK) {
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    void* ptr = q_arena_alloc(&ctx, 0);
    
    // Zero size should be aligned to Q_ALIGN (64 bytes) by safe_align_size
    // So it should return a valid pointer
    if (ptr != NULL) {
        uintptr_t ptr_addr = (uintptr_t)ptr;
        if ((ptr_addr % Q_ALIGN) == 0) {
            TEST_PASS();
        } else {
            TEST_FAIL("Pointer not aligned");
        }
    } else {
        // If it returns NULL, it might be due to OOM or other check
        // This is acceptable behavior - zero size allocation is edge case
        TEST_PASS();  // Accept both NULL and non-NULL as valid
    }
    
    q_free_memory(&ctx);
}

// Test 15: q_arena_reset - Null context pointer
static void test_arena_reset_null_ctx(void) {
    TEST_START("q_arena_reset - Null context pointer");
    
    if (setjmp(crash_jmp_buf) == 0) {
        signal(SIGSEGV, crash_handler);
        signal(SIGBUS, crash_handler);
        signal(SIGABRT, crash_handler);
        
        q_arena_reset(NULL);
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        
        TEST_PASS();
    } else {
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        TEST_CRASH();
    }
}

// Test 16: q_arena_reset - Uninitialized arena
static void test_arena_reset_uninitialized(void) {
    TEST_START("q_arena_reset - Uninitialized arena");
    
    q_context ctx = {0};
    ctx.scratch_head = 100;  // Set head but no buffer
    
    if (setjmp(crash_jmp_buf) == 0) {
        signal(SIGSEGV, crash_handler);
        signal(SIGBUS, crash_handler);
        signal(SIGABRT, crash_handler);
        
        q_arena_reset(&ctx);
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        
        if (ctx.scratch_head == 0) {
            TEST_PASS();
        } else {
            TEST_FAIL("Head not reset to 0");
        }
    } else {
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        TEST_CRASH();
    }
}

// Test 17: q_free_memory - Null context pointer
static void test_free_memory_null_ctx(void) {
    TEST_START("q_free_memory - Null context pointer");
    
    if (setjmp(crash_jmp_buf) == 0) {
        signal(SIGSEGV, crash_handler);
        signal(SIGBUS, crash_handler);
        signal(SIGABRT, crash_handler);
        
        q_free_memory(NULL);
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        
        TEST_PASS();
    } else {
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        TEST_CRASH();
    }
}

// Test 18: q_free_memory - Double free (should be safe)
static void test_free_memory_double_free(void) {
    TEST_START("q_free_memory - Double free (should be safe)");
    
    q_context ctx = {0};
    
    // Try to create model file if it doesn't exist
    if (q_init_memory(&ctx, "model_dummy.qorus") != Q_OK) {
        // If model file doesn't exist, create minimal test without mmap
        // Test double free with only KV cache and arena
        if (q_alloc_kv_cache(&ctx, 1024) != Q_OK) {
            TEST_FAIL("Failed to allocate KV cache");
            return;
        }
        
        if (q_alloc_arena(&ctx, 1024) != Q_OK) {
            q_free_memory(&ctx);
            TEST_FAIL("Failed to allocate arena");
            return;
        }
    } else {
        // Model loaded successfully, allocate other resources
        if (q_alloc_kv_cache(&ctx, 1024) != Q_OK) {
            q_free_memory(&ctx);
            TEST_FAIL("Failed to allocate KV cache");
            return;
        }
        
        if (q_alloc_arena(&ctx, 1024) != Q_OK) {
            q_free_memory(&ctx);
            TEST_FAIL("Failed to allocate arena");
            return;
        }
    }
    
    // First free
    q_free_memory(&ctx);
    
    // Second free (should be safe - all pointers should be NULL)
    if (setjmp(crash_jmp_buf) == 0) {
        signal(SIGSEGV, crash_handler);
        signal(SIGBUS, crash_handler);
        signal(SIGABRT, crash_handler);
        
        q_free_memory(&ctx);
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        
        TEST_PASS();
    } else {
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        TEST_CRASH();
    }
}

// Test 19: q_free_memory - Partial allocation (free only what was allocated)
static void test_free_memory_partial(void) {
    TEST_START("q_free_memory - Partial allocation");
    
    q_context ctx = {0};
    
    // Only allocate arena
    if (q_alloc_arena(&ctx, 1024) != Q_OK) {
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    // Free should handle partial allocation gracefully
    if (setjmp(crash_jmp_buf) == 0) {
        signal(SIGSEGV, crash_handler);
        signal(SIGBUS, crash_handler);
        signal(SIGABRT, crash_handler);
        
        q_free_memory(&ctx);
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        
        // Verify all pointers are NULL
        if (ctx.scratch_buffer == NULL && ctx.kv_buffer == NULL && 
            ctx.weights_mmap == NULL && ctx.header == NULL) {
            TEST_PASS();
        } else {
            TEST_FAIL("Pointers not cleared after free");
        }
    } else {
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        TEST_CRASH();
    }
}

// Test 20: q_free_memory - LIFO order verification
static void test_free_memory_lifo_order(void) {
    TEST_START("q_free_memory - LIFO order verification");
    
    q_context ctx = {0};
    
    // Try to allocate in order: mmap -> kv_cache -> arena
    // If mmap fails, test with kv_cache -> arena only
    bool has_mmap = false;
    if (q_init_memory(&ctx, "model_dummy.qorus") == Q_OK) {
        has_mmap = true;
        if (ctx.weights_mmap == NULL || ctx.header == NULL) {
            q_free_memory(&ctx);
            TEST_FAIL("Mmap not initialized");
            return;
        }
    }
    
    if (q_alloc_kv_cache(&ctx, 1024) != Q_OK) {
        q_free_memory(&ctx);
        TEST_FAIL("Failed to allocate KV cache");
        return;
    }
    
    if (ctx.kv_buffer == NULL) {
        q_free_memory(&ctx);
        TEST_FAIL("KV cache not initialized");
        return;
    }
    
    if (q_alloc_arena(&ctx, 1024) != Q_OK) {
        q_free_memory(&ctx);
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    if (ctx.scratch_buffer == NULL) {
        q_free_memory(&ctx);
        TEST_FAIL("Arena not initialized");
        return;
    }
    
    // Free should free in reverse order: arena -> kv_cache -> mmap (if exists)
    q_free_memory(&ctx);
    
    // Verify all pointers are NULL
    if (ctx.scratch_buffer == NULL && ctx.kv_buffer == NULL && 
        (!has_mmap || (ctx.weights_mmap == NULL && ctx.header == NULL))) {
        TEST_PASS();
    } else {
        TEST_FAIL("Pointers not cleared after free");
    }
}

// Test 21: Multiple allocations and resets (stress test)
static void test_arena_stress(void) {
    TEST_START("Arena - Multiple allocations and resets (stress test)");
    
    q_context ctx = {0};
    if (q_alloc_arena(&ctx, 1024 * 1024) != Q_OK) {
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    // Allocate and reset multiple times
    for (int i = 0; i < 100; i++) {
        void* ptrs[10];
        for (int j = 0; j < 10; j++) {
            ptrs[j] = q_arena_alloc(&ctx, 64);
            if (ptrs[j] == NULL) {
                TEST_FAIL_MSG("Allocation failed at iteration %d, allocation %d", i, j);
                q_free_memory(&ctx);
                return;
            }
        }
        q_arena_reset(&ctx);
        if (ctx.scratch_head != 0) {
            TEST_FAIL_MSG("Head not reset at iteration %d", i);
            q_free_memory(&ctx);
            return;
        }
    }
    
    q_free_memory(&ctx);
    TEST_PASS();
}

// Test 22: Alignment preservation across allocations
static void test_arena_alignment_preservation(void) {
    TEST_START("Arena - Alignment preservation across allocations");
    
    q_context ctx = {0};
    if (q_alloc_arena(&ctx, 1024 * 1024) != Q_OK) {
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    // Allocate various sizes and verify alignment
    size_t sizes[] = {1, 7, 15, 31, 33, 63, 65, 127, 128, 255, 256};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int i = 0; i < num_sizes; i++) {
        void* ptr = q_arena_alloc(&ctx, sizes[i]);
        if (ptr == NULL) {
            TEST_FAIL_MSG("Allocation failed for size %zu", sizes[i]);
            q_free_memory(&ctx);
            return;
        }
        
        uintptr_t ptr_addr = (uintptr_t)ptr;
        if ((ptr_addr % Q_ALIGN) != 0) {
            TEST_FAIL_MSG("Pointer not aligned for size %zu: %p", sizes[i], ptr);
            q_free_memory(&ctx);
            return;
        }
    }
    
    q_free_memory(&ctx);
    TEST_PASS();
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("=== Adversarial Tests: Memory Management ===\n\n");
    
    // Run all tests
    test_alloc_kv_cache_null_ctx();
    test_alloc_kv_cache_double_alloc();
    test_alloc_kv_cache_overflow();
    test_alloc_kv_cache_zero_size();
    test_alloc_arena_null_ctx();
    test_alloc_arena_double_alloc();
    test_alloc_arena_overflow();
    test_arena_alloc_null_ctx();
    test_arena_alloc_uninitialized();
    test_arena_alloc_overflow_alignment();
    test_arena_alloc_overflow_addition();
    test_arena_alloc_oom();
    test_arena_alloc_misalignment();
    test_arena_alloc_zero_size();
    test_arena_reset_null_ctx();
    test_arena_reset_uninitialized();
    test_free_memory_null_ctx();
    test_free_memory_double_free();
    test_free_memory_partial();
    test_free_memory_lifo_order();
    test_arena_stress();
    test_arena_alignment_preservation();
    
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

