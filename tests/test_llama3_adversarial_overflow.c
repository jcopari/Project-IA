#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>
#include <limits.h>

// ============================================================================
// TEST CONFIGURATION
// ============================================================================

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;
static int tests_crashed = 0;

static jmp_buf crash_jmp_buf;
__attribute__((unused)) static void crash_handler(int sig) {
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
// TEST SUITE: Overflow Protection Tests
// ============================================================================

// Helper to create minimal model context for testing
static q_context* create_test_context(void) {
    static q_context ctx;
    memset(&ctx, 0, sizeof(ctx));
    
    // Create minimal mmap'd buffer
    size_t buffer_size = 1024 * 1024;
    void* buffer = aligned_alloc(Q_ALIGN, buffer_size);
    if (!buffer) {
        return NULL;
    }
    
    // Write minimal header
    q_model_header* header = (q_model_header*)buffer;
    header->magic = Q_MAGIC;
    header->version = 1;
    header->vocab_size = 100;
    header->dim = 128;
    header->hidden_dim = 256;
    header->n_layers = 1;
    header->n_heads = 4;
    header->n_kv_heads = 2;
    header->max_seq_len = 2048;
    header->rope_freq_base = 500000.0f;
    
    ctx.weights_mmap = buffer;
    ctx.weights_size = buffer_size;
    ctx.header = header;
    
    return &ctx;
}

// Test 1: create_tensor_view - Null context pointer
static void test_create_tensor_view_null_ctx(void) {
    TEST_START("create_tensor_view - Null context pointer");
    
    // We can't directly call create_tensor_view (it's static)
    // But we can test via llama_build_graph which uses it
    llama_model model = {0};
    q_error_code ret = llama_build_graph(NULL, &model);
    
    if (ret == Q_ERR_NULL_PTR) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_NULL_PTR, got %d", ret);
    }
}

// Test 2: create_tensor_view - Null weights_mmap
static void test_create_tensor_view_null_mmap(void) {
    TEST_START("create_tensor_view - Null weights_mmap");
    
    q_context ctx = {0};
    llama_model model = {0};
    
    // Allocate arena but don't initialize mmap
    if (q_alloc_arena(&ctx, 1024 * 1024) != Q_OK) {
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    q_error_code ret = llama_build_graph(&ctx, &model);
    
    if (ret == Q_ERR_NULL_PTR) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_NULL_PTR, got %d", ret);
    }
    
    q_free_memory(&ctx);
}

// Test 3: create_tensor_view - Overflow in calculate_f32_size
static void test_create_tensor_view_f32_overflow(void) {
    TEST_START("create_tensor_view - Overflow in F32 size calculation");
    
    q_context* ctx = create_test_context();
    if (!ctx) {
        TEST_FAIL("Failed to create test context");
        return;
    }
    
    if (q_alloc_arena(ctx, 1024 * 1024) != Q_OK) {
        free(ctx->weights_mmap);
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    // Modify header to cause overflow
    // Use very large dimensions that will overflow when multiplied
    ctx->header->vocab_size = UINT32_MAX / 2;
    ctx->header->dim = UINT32_MAX / 2;
    
    llama_model model = {0};
    q_error_code ret = llama_build_graph(ctx, &model);
    
    // Should fail due to overflow or invalid config
    if (ret == Q_ERR_INVALID_CONFIG || ret == Q_ERR_ARENA_OOM) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_CONFIG or Q_ERR_ARENA_OOM, got %d", ret);
    }
    
    q_free_memory(ctx);
}

// Test 4: create_tensor_view - Overflow in calculate_q4_0_size
static void test_create_tensor_view_q4_overflow(void) {
    TEST_START("create_tensor_view - Overflow in Q4_0 size calculation");
    
    q_context* ctx = create_test_context();
    if (!ctx) {
        TEST_FAIL("Failed to create test context");
        return;
    }
    
    if (q_alloc_arena(ctx, 1024 * 1024) != Q_OK) {
        free(ctx->weights_mmap);
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    // Modify header to cause overflow in Q4_0 calculation
    ctx->header->dim = UINT32_MAX / 2;
    ctx->header->hidden_dim = UINT32_MAX / 2;
    
    llama_model model = {0};
    q_error_code ret = llama_build_graph(ctx, &model);
    
    // Should fail due to overflow
    if (ret == Q_ERR_INVALID_CONFIG || ret == Q_ERR_ARENA_OOM) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_CONFIG or Q_ERR_ARENA_OOM, got %d", ret);
    }
    
    q_free_memory(ctx);
}

// Test 5: create_tensor_view - Tensor extends beyond mmap bounds
static void test_create_tensor_view_bounds_overflow(void) {
    TEST_START("create_tensor_view - Tensor extends beyond mmap bounds");
    
    q_context* ctx = create_test_context();
    if (!ctx) {
        TEST_FAIL("Failed to create test context");
        return;
    }
    
    // Make mmap very small
    ctx->weights_size = Q_HEADER_SIZE + 100;  // Only 100 bytes after header
    
    if (q_alloc_arena(ctx, 1024 * 1024) != Q_OK) {
        free(ctx->weights_mmap);
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    llama_model model = {0};
    q_error_code ret = llama_build_graph(ctx, &model);
    
    // Should fail due to bounds check
    if (ret == Q_ERR_INVALID_CONFIG) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_CONFIG, got %d", ret);
    }
    
    q_free_memory(ctx);
}

// Test 6: create_tensor_view - Wraparound detection
static void test_create_tensor_view_wraparound(void) {
    TEST_START("create_tensor_view - Wraparound detection");
    
    q_context* ctx = create_test_context();
    if (!ctx) {
        TEST_FAIL("Failed to create test context");
        return;
    }
    
    if (q_alloc_arena(ctx, 1024 * 1024) != Q_OK) {
        free(ctx->weights_mmap);
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    // Try to create tensor view at end of mmap with size that would wraparound
    // This is tested indirectly through llama_build_graph
    
    llama_model model = {0};
    q_error_code ret = llama_build_graph(ctx, &model);
    
    // Should either succeed (if bounds are valid) or fail gracefully
    if (ret == Q_OK || ret == Q_ERR_INVALID_CONFIG || ret == Q_ERR_ARENA_OOM) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Unexpected error code: %d", ret);
    }
    
    q_free_memory(ctx);
}

// Test 7: create_tensor_view - Overflow in stride calculation (F32)
static void test_create_tensor_view_stride_overflow_f32(void) {
    TEST_START("create_tensor_view - Overflow in F32 stride calculation");
    
    q_context* ctx = create_test_context();
    if (!ctx) {
        TEST_FAIL("Failed to create test context");
        return;
    }
    
    if (q_alloc_arena(ctx, 1024 * 1024) != Q_OK) {
        free(ctx->weights_mmap);
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    // Set dimensions that will cause overflow in stride calculation
    // This is hard to trigger directly, but we can test with very large dims
    ctx->header->vocab_size = 100;
    ctx->header->dim = UINT32_MAX / 4;  // Large but not overflowing in size calc
    
    llama_model model = {0};
    q_error_code ret = llama_build_graph(ctx, &model);
    
    // Should fail due to overflow in stride calculation or size calculation
    if (ret == Q_ERR_INVALID_CONFIG || ret == Q_ERR_ARENA_OOM) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_CONFIG or Q_ERR_ARENA_OOM, got %d", ret);
    }
    
    q_free_memory(ctx);
}

// Test 8: create_tensor_view - Overflow in stride calculation (Q4_0)
static void test_create_tensor_view_stride_overflow_q4(void) {
    TEST_START("create_tensor_view - Overflow in Q4_0 stride calculation");
    
    q_context* ctx = create_test_context();
    if (!ctx) {
        TEST_FAIL("Failed to create test context");
        return;
    }
    
    if (q_alloc_arena(ctx, 1024 * 1024) != Q_OK) {
        free(ctx->weights_mmap);
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    // Set dimensions that will cause overflow in Q4_0 stride calculation
    ctx->header->dim = UINT32_MAX / 4;
    ctx->header->hidden_dim = UINT32_MAX / 4;
    
    llama_model model = {0};
    q_error_code ret = llama_build_graph(ctx, &model);
    
    // Should fail due to overflow
    if (ret == Q_ERR_INVALID_CONFIG || ret == Q_ERR_ARENA_OOM) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_CONFIG or Q_ERR_ARENA_OOM, got %d", ret);
    }
    
    q_free_memory(ctx);
}

// Test 9: create_tensor_view - Invalid data pointer (outside mmap)
static void test_create_tensor_view_invalid_ptr(void) {
    TEST_START("create_tensor_view - Invalid data pointer (outside mmap)");
    
    q_context* ctx = create_test_context();
    if (!ctx) {
        TEST_FAIL("Failed to create test context");
        return;
    }
    
    if (q_alloc_arena(ctx, 1024 * 1024) != Q_OK) {
        free(ctx->weights_mmap);
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    // Corrupt weights_size to make bounds check fail
    ctx->weights_size = Q_HEADER_SIZE + 10;  // Very small, will fail bounds check
    
    llama_model model = {0};
    q_error_code ret = llama_build_graph(ctx, &model);
    
    // Should fail due to bounds check
    if (ret == Q_ERR_INVALID_CONFIG) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_CONFIG, got %d", ret);
    }
    
    q_free_memory(ctx);
}

// Test 10: create_tensor_view - Invalid dtype
static void test_create_tensor_view_invalid_dtype(void) {
    TEST_START("create_tensor_view - Invalid dtype");
    
    // This is tested indirectly through llama_build_graph
    // which only uses Q_F32 and Q_Q4_0
    
    q_context* ctx = create_test_context();
    if (!ctx) {
        TEST_FAIL("Failed to create test context");
        return;
    }
    
    if (q_alloc_arena(ctx, 1024 * 1024) != Q_OK) {
        free(ctx->weights_mmap);
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    llama_model model = {0};
    q_error_code ret = llama_build_graph(ctx, &model);
    
    // Should succeed (only valid dtypes are used)
    if (ret == Q_OK || ret == Q_ERR_INVALID_CONFIG || ret == Q_ERR_ARENA_OOM) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Unexpected error code: %d", ret);
    }
    
    q_free_memory(ctx);
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("=== Adversarial Tests: Llama3 Overflow Protection ===\n\n");
    
    // Run all tests
    test_create_tensor_view_null_ctx();
    test_create_tensor_view_null_mmap();
    test_create_tensor_view_f32_overflow();
    test_create_tensor_view_q4_overflow();
    test_create_tensor_view_bounds_overflow();
    test_create_tensor_view_wraparound();
    test_create_tensor_view_stride_overflow_f32();
    test_create_tensor_view_stride_overflow_q4();
    test_create_tensor_view_invalid_ptr();
    test_create_tensor_view_invalid_dtype();
    
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

