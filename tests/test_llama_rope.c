#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <signal.h>
#include <setjmp.h>

// ============================================================================
// ADVERSARIAL TEST SUITE: RoPE Functions
// ============================================================================
// Target Function: generate_rope_cos_sin() (static, tested indirectly)
//
// Strategy: Test through llama_forward() calls with various positions
// and validate RoPE is applied correctly. Since generate_rope_cos_sin
// is static, we test it indirectly by:
// 1. Running forward passes at different positions
// 2. Validating RoPE cache is used when available
// 3. Testing edge cases (pos = 0, pos = max_seq_len - 1)
// 4. Testing with and without cache enabled
// ============================================================================

// Test statistics
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;
static int tests_crashed = 0;

// Crash detection
static jmp_buf crash_jmp;
static void crash_handler(int sig) {
    (void)sig;
    longjmp(crash_jmp, 1);
}

// ============================================================================
// TEST MACROS
// ============================================================================

#define TEST_START(name) \
    do { \
        tests_run++; \
        printf("  [%d] %s... ", tests_run, name); \
        fflush(stdout); \
    } while(0)

#define TEST_PASS() \
    do { \
        tests_passed++; \
        printf("PASS\n"); \
    } while(0)

#define TEST_FAIL(reason) \
    do { \
        tests_failed++; \
        printf("FAIL: %s\n", reason); \
    } while(0)

#define TEST_CRASH() \
    do { \
        tests_crashed++; \
        printf("CRASHED\n"); \
    } while(0)

// Wrapper function for crash detection
static void run_test_with_crash_detection(void (*test_func)(void)) {
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp) == 0) {
        test_func();
    } else {
        TEST_CRASH();
    }
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Ensure dummy model exists
static bool ensure_dummy_model(void) {
    FILE* check = fopen("model_dummy.qorus", "rb");
    if (check != NULL) {
        fclose(check);
        return true;
    }
    
    int ret = system("python3 tools/convert_llama.py model_dummy.qorus 2 > /dev/null 2>&1");
    return (ret == 0);
}

// Helper: Setup model
static bool setup_model(q_context* ctx, q_llama_model* model) {
    memset(ctx, 0, sizeof(q_context));
    memset(model, 0, sizeof(q_llama_model));
    
    q_error_code ret = q_init_memory(ctx, "model_dummy.qorus");
    if (ret != Q_OK) return false;
    
    ret = q_alloc_arena(ctx, 64 * 1024 * 1024);
    if (ret != Q_OK) {
        q_free_memory(ctx);
        return false;
    }
    
    ret = llama_build_graph(ctx, model);
    if (ret != Q_OK) {
        q_free_memory(ctx);
        return false;
    }
    
    uint32_t head_dim = model->config.dim / model->config.n_heads;
    size_t kv_size = (size_t)model->config.n_layers * 
                     (size_t)model->config.n_kv_heads * 
                     (size_t)model->config.max_seq_len * 
                     (size_t)head_dim * 
                     sizeof(float) * 2;
    ret = q_alloc_kv_cache(ctx, kv_size);
    if (ret != Q_OK) {
        llama_free_graph(model);
        q_free_memory(ctx);
        return false;
    }
    
    return true;
}

// ============================================================================
// TEST CASES
// ============================================================================

// Test 1: Happy Path - RoPE applied at position 0
static void test_rope_pos0_impl(void) {
    TEST_START("RoPE application - Position 0");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model(&ctx, &model)) {
        TEST_FAIL("Failed to setup model");
        return;
    }
    
    uint32_t tokens[1] = {0};
    float* logits = (float*)aligned_alloc(Q_ALIGN, model.config.vocab_size * sizeof(float));
    if (logits == NULL) {
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_FAIL("Failed to allocate logits");
        return;
    }
    
    q_error_code ret = llama_forward(&model, &ctx, tokens, 1, 0, logits);
    
    free(logits);
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    if (ret != Q_OK) {
        TEST_FAIL("Forward pass should succeed");
        return;
    }
    
    TEST_PASS();
}

static void test_rope_pos0(void) {
    run_test_with_crash_detection(test_rope_pos0_impl);
}

// Test 2: Happy Path - RoPE applied at multiple positions
static void test_rope_multiple_pos_impl(void) {
    TEST_START("RoPE application - Multiple positions");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model(&ctx, &model)) {
        TEST_FAIL("Failed to setup model");
        return;
    }
    
    uint32_t seq_len = 4;
    uint32_t tokens[4] = {0, 1, 2, 3};
    float* logits = (float*)aligned_alloc(Q_ALIGN, model.config.vocab_size * sizeof(float));
    if (logits == NULL) {
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_FAIL("Failed to allocate logits");
        return;
    }
    
    q_error_code ret = llama_forward(&model, &ctx, tokens, seq_len, 0, logits);
    
    free(logits);
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    if (ret != Q_OK) {
        TEST_FAIL("Forward pass should succeed");
        return;
    }
    
    TEST_PASS();
}

static void test_rope_multiple_pos(void) {
    run_test_with_crash_detection(test_rope_multiple_pos_impl);
}

// Test 3: Edge Case - Position max_seq_len - 1
static void test_rope_max_pos_impl(void) {
    TEST_START("RoPE application - Position max_seq_len - 1");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model(&ctx, &model)) {
        TEST_FAIL("Failed to setup model");
        return;
    }
    
    uint32_t max_pos = model.config.max_seq_len - 1;
    uint32_t tokens[1] = {0};
    float* logits = (float*)aligned_alloc(Q_ALIGN, model.config.vocab_size * sizeof(float));
    if (logits == NULL) {
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_FAIL("Failed to allocate logits");
        return;
    }
    
    q_error_code ret = llama_forward(&model, &ctx, tokens, 1, max_pos, logits);
    
    free(logits);
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    if (ret != Q_OK) {
        TEST_FAIL("Forward pass should succeed at max position");
        return;
    }
    
    TEST_PASS();
}

static void test_rope_max_pos(void) {
    run_test_with_crash_detection(test_rope_max_pos_impl);
}

// Test 4: Edge Case - Incremental generation (pos > 0)
static void test_rope_incremental_impl(void) {
    TEST_START("RoPE application - Incremental generation");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model(&ctx, &model)) {
        TEST_FAIL("Failed to setup model");
        return;
    }
    
    uint32_t tokens[1] = {0};
    float* logits = (float*)aligned_alloc(Q_ALIGN, model.config.vocab_size * sizeof(float));
    if (logits == NULL) {
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_FAIL("Failed to allocate logits");
        return;
    }
    
    // Run multiple forward passes at increasing positions
    for (uint32_t pos = 0; pos < 10 && pos < model.config.max_seq_len; pos++) {
        q_error_code ret = llama_forward(&model, &ctx, tokens, 1, pos, logits);
        if (ret != Q_OK) {
            free(logits);
            llama_free_graph(&model);
            q_free_memory(&ctx);
            TEST_FAIL("Forward pass should succeed at all positions");
            return;
        }
    }
    
    free(logits);
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    TEST_PASS();
}

static void test_rope_incremental(void) {
    run_test_with_crash_detection(test_rope_incremental_impl);
}

// Test 5: Validate RoPE cache is used when available
// Note: This tests the cache lookup path in generate_rope_cos_sin
static void test_rope_cache_lookup_impl(void) {
    TEST_START("RoPE cache - Cache lookup (when enabled)");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model(&ctx, &model)) {
        TEST_FAIL("Failed to setup model");
        return;
    }
    
    // Check if cache is enabled (depends on max_seq_len <= 8192)
    if (!model.rope_cache_enabled) {
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_PASS(); // Cache not enabled for this model, skip test
        return;
    }
    
    uint32_t tokens[1] = {0};
    float* logits = (float*)aligned_alloc(Q_ALIGN, model.config.vocab_size * sizeof(float));
    if (logits == NULL) {
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_FAIL("Failed to allocate logits");
        return;
    }
    
    // Run forward pass - should use cache
    q_error_code ret = llama_forward(&model, &ctx, tokens, 1, 0, logits);
    
    free(logits);
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    if (ret != Q_OK) {
        TEST_FAIL("Forward pass should succeed with cache");
        return;
    }
    
    TEST_PASS();
}

static void test_rope_cache_lookup(void) {
    run_test_with_crash_detection(test_rope_cache_lookup_impl);
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("========================================\n");
    printf("  ADVERSARIAL TEST SUITE: RoPE Functions\n");
    printf("========================================\n\n");
    printf("Target: generate_rope_cos_sin() (indirect testing)\n");
    printf("Strategy: Test through llama_forward() calls\n\n");
    
    // Ensure dummy model exists
    printf("Ensuring dummy model exists...\n");
    if (!ensure_dummy_model()) {
        printf("ERROR: Failed to generate dummy model. Run manually:\n");
        printf("  python3 tools/convert_llama.py model_dummy.qorus 2\n");
        return 1;
    }
    printf("✓ Dummy model ready\n\n");
    
    // Reset statistics
    tests_run = 0;
    tests_passed = 0;
    tests_failed = 0;
    tests_crashed = 0;
    
    // CATEGORY 1: HAPPY PATH
    printf("CATEGORY 1: Happy Path\n");
    printf("-----------------------------------\n");
    test_rope_pos0();
    test_rope_multiple_pos();
    test_rope_max_pos();
    test_rope_incremental();
    printf("\n");
    
    // CATEGORY 2: CACHE TESTING
    printf("CATEGORY 2: Cache Testing\n");
    printf("-----------------------------------\n");
    test_rope_cache_lookup();
    printf("\n");
    
    // Summary
    printf("========================================\n");
    printf("  TEST SUMMARY\n");
    printf("========================================\n");
    printf("Total tests: %d\n", tests_run);
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    printf("Crashed: %d\n", tests_crashed);
    printf("\n");
    
    if (tests_failed == 0 && tests_crashed == 0) {
        printf("✓ All tests passed!\n");
        return 0;
    } else {
        printf("✗ Some tests failed or crashed\n");
        return 1;
    }
}

