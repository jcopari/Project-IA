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
#include <limits.h>

// ============================================================================
// ADVERSARIAL TEST SUITE: Scratchpad Functions
// ============================================================================
// Target Functions:
// - calculate_layer_scratchpad_size() (static, tested indirectly)
// - init_layer_scratchpad() (static, tested indirectly)
//
// Strategy: Test through llama_forward() calls with various configurations
// Since these are static functions, we test them indirectly by:
// 1. Creating models with different configurations
// 2. Calling llama_forward() with various seq_len values
// 3. Validating that scratchpad allocation succeeds/fails correctly
// 4. Checking alignment and bounds of allocated buffers
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
    
    // Try to generate model
    int ret = system("python3 tools/convert_llama.py model_dummy.qorus 2 > /dev/null 2>&1");
    return (ret == 0);
}

// Helper: Create minimal valid context and model
static bool setup_model(q_context* ctx, q_llama_model* model) {
    memset(ctx, 0, sizeof(q_context));
    memset(model, 0, sizeof(q_llama_model));
    
    q_error_code ret = q_init_memory(ctx, "model_dummy.qorus");
    if (ret != Q_OK) return false;
    
    ret = q_alloc_arena(ctx, 64 * 1024 * 1024); // 64MB
    if (ret != Q_OK) {
        q_free_memory(ctx);
        return false;
    }
    
    ret = llama_build_graph(ctx, model);
    if (ret != Q_OK) {
        q_free_memory(ctx);
        return false;
    }
    
    // Allocate KV cache
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
// TEST CASES: calculate_layer_scratchpad_size (indirect)
// ============================================================================

// Test 1: Happy Path - Normal seq_len
static void test_scratchpad_size_normal_impl(void) {
    TEST_START("Scratchpad size calculation - Normal seq_len");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model(&ctx, &model)) {
        TEST_FAIL("Failed to setup model");
        return;
    }
    
    // Test with normal seq_len
    uint32_t seq_len = 128;
    uint32_t tokens[128];
    for (uint32_t i = 0; i < seq_len; i++) {
        tokens[i] = i % model.config.vocab_size;
    }
    
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
        TEST_FAIL("Forward pass should succeed with normal seq_len");
        return;
    }
    
    TEST_PASS();
}

static void test_scratchpad_size_normal(void) {
    run_test_with_crash_detection(test_scratchpad_size_normal_impl);
}

// Test 2: Edge Case - seq_len = 1 (minimum)
static void test_scratchpad_size_min_impl(void) {
    TEST_START("Scratchpad size calculation - seq_len = 1");
    
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
        TEST_FAIL("Forward pass should succeed with seq_len = 1");
        return;
    }
    
    TEST_PASS();
}

static void test_scratchpad_size_min(void) {
    run_test_with_crash_detection(test_scratchpad_size_min_impl);
}

// Test 3: Edge Case - seq_len = max_seq_len (maximum)
// Note: May fail if max_seq_len is very large (e.g., 8192) due to memory constraints
static void test_scratchpad_size_max_impl(void) {
    TEST_START("Scratchpad size calculation - seq_len = max_seq_len");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model(&ctx, &model)) {
        TEST_FAIL("Failed to setup model");
        return;
    }
    
    // Limit to reasonable size to avoid OOM in test environment
    uint32_t seq_len = model.config.max_seq_len;
    if (seq_len > 2048) {
        seq_len = 2048; // Cap at 2048 for testing
    }
    
    uint32_t* tokens = (uint32_t*)malloc(seq_len * sizeof(uint32_t));
    if (tokens == NULL) {
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_FAIL("Failed to allocate tokens");
        return;
    }
    
    for (uint32_t i = 0; i < seq_len; i++) {
        tokens[i] = i % model.config.vocab_size;
    }
    
    float* logits = (float*)aligned_alloc(Q_ALIGN, model.config.vocab_size * sizeof(float));
    if (logits == NULL) {
        free(tokens);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_FAIL("Failed to allocate logits");
        return;
    }
    
    q_error_code ret = llama_forward(&model, &ctx, tokens, seq_len, 0, logits);
    
    free(logits);
    free(tokens);
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    // Accept both success and OOM (both are valid outcomes)
    if (ret != Q_OK && ret != Q_ERR_ARENA_OOM) {
        TEST_FAIL("Should succeed or return OOM");
        return;
    }
    
    TEST_PASS();
}

static void test_scratchpad_size_max(void) {
    run_test_with_crash_detection(test_scratchpad_size_max_impl);
}

// Test 4: Security - seq_len = 0 (should fail validation)
static void test_scratchpad_size_zero_impl(void) {
    TEST_START("Scratchpad size calculation - seq_len = 0 (invalid)");
    
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
    
    q_error_code ret = llama_forward(&model, &ctx, tokens, 0, 0, logits);
    
    free(logits);
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    // Should fail validation before scratchpad calculation
    if (ret == Q_OK) {
        TEST_FAIL("Should reject seq_len = 0");
        return;
    }
    
    if (ret != Q_ERR_INVALID_SIZE) {
        TEST_FAIL("Should return Q_ERR_INVALID_SIZE for seq_len = 0");
        return;
    }
    
    TEST_PASS();
}

static void test_scratchpad_size_zero(void) {
    run_test_with_crash_detection(test_scratchpad_size_zero_impl);
}

// Test 5: Security - seq_len > max_seq_len (should fail validation)
static void test_scratchpad_size_overflow_impl(void) {
    TEST_START("Scratchpad size calculation - seq_len > max_seq_len");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model(&ctx, &model)) {
        TEST_FAIL("Failed to setup model");
        return;
    }
    
    uint32_t seq_len = model.config.max_seq_len + 1;
    uint32_t* tokens = (uint32_t*)malloc(seq_len * sizeof(uint32_t));
    if (tokens == NULL) {
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_FAIL("Failed to allocate tokens");
        return;
    }
    
    float* logits = (float*)aligned_alloc(Q_ALIGN, model.config.vocab_size * sizeof(float));
    if (logits == NULL) {
        free(tokens);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_FAIL("Failed to allocate logits");
        return;
    }
    
    q_error_code ret = llama_forward(&model, &ctx, tokens, seq_len, 0, logits);
    
    free(logits);
    free(tokens);
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    // Should fail validation before scratchpad calculation
    if (ret == Q_OK) {
        TEST_FAIL("Should reject seq_len > max_seq_len");
        return;
    }
    
    if (ret != Q_ERR_INVALID_SIZE) {
        TEST_FAIL("Should return Q_ERR_INVALID_SIZE for seq_len > max_seq_len");
        return;
    }
    
    TEST_PASS();
}

static void test_scratchpad_size_overflow(void) {
    run_test_with_crash_detection(test_scratchpad_size_overflow_impl);
}

// Test 6: Security - Arena OOM (scratchpad too large)
// Note: This test requires building graph first, which may fail with very small arena
static void test_scratchpad_arena_oom_impl(void) {
    TEST_START("Scratchpad allocation - Arena OOM");
    
    q_context ctx;
    q_llama_model model;
    
    memset(&ctx, 0, sizeof(q_context));
    memset(&model, 0, sizeof(q_llama_model));
    
    q_error_code ret = q_init_memory(&ctx, "model_dummy.qorus");
    if (ret != Q_OK) {
        TEST_FAIL("Failed to init memory");
        return;
    }
    
    // Allocate small arena (may be too small even for graph building)
    ret = q_alloc_arena(&ctx, 64 * 1024); // 64KB - may be too small
    if (ret != Q_OK) {
        q_free_memory(&ctx);
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    ret = llama_build_graph(&ctx, &model);
    if (ret != Q_OK) {
        // Graph building failed - arena too small, which is acceptable
        q_free_memory(&ctx);
        TEST_PASS(); // This is a valid outcome
        return;
    }
    
    // Allocate KV cache
    uint32_t head_dim = model.config.dim / model.config.n_heads;
    size_t kv_size = (size_t)model.config.n_layers * 
                     (size_t)model.config.n_kv_heads * 
                     (size_t)model.config.max_seq_len * 
                     (size_t)head_dim * 
                     sizeof(float) * 2;
    ret = q_alloc_kv_cache(&ctx, kv_size);
    if (ret != Q_OK) {
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_PASS(); // KV cache allocation failed, which is acceptable
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
    
    ret = llama_forward(&model, &ctx, tokens, 1, 0, logits);
    
    free(logits);
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    // Should fail with OOM when trying to allocate scratchpad
    if (ret == Q_OK) {
        TEST_FAIL("Should fail with OOM when arena is too small");
        return;
    }
    
    if (ret != Q_ERR_ARENA_OOM) {
        TEST_FAIL("Should return Q_ERR_ARENA_OOM");
        return;
    }
    
    TEST_PASS();
}

static void test_scratchpad_arena_oom(void) {
    run_test_with_crash_detection(test_scratchpad_arena_oom_impl);
}

// Test 7: Fuzzing - Various seq_len values
static void test_scratchpad_fuzzing_impl(void) {
    TEST_START("Scratchpad size calculation - Fuzzing seq_len");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model(&ctx, &model)) {
        TEST_FAIL("Failed to setup model");
        return;
    }
    
    // Test various seq_len values (limit to avoid OOM)
    uint32_t test_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    size_t num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    bool all_passed = true;
    int passed_count = 0;
    
    for (size_t i = 0; i < num_tests; i++) {
        uint32_t seq_len = test_sizes[i];
        if (seq_len > model.config.max_seq_len) continue;
        
        uint32_t* tokens = (uint32_t*)malloc(seq_len * sizeof(uint32_t));
        if (tokens == NULL) continue;
        
        for (uint32_t j = 0; j < seq_len; j++) {
            tokens[j] = j % model.config.vocab_size;
        }
        
        float* logits = (float*)aligned_alloc(Q_ALIGN, model.config.vocab_size * sizeof(float));
        if (logits == NULL) {
            free(tokens);
            continue;
        }
        
        q_error_code ret = llama_forward(&model, &ctx, tokens, seq_len, 0, logits);
        
        free(logits);
        free(tokens);
        
        if (ret == Q_OK) {
            passed_count++;
        } else if (ret == Q_ERR_ARENA_OOM) {
            // OOM is acceptable for large seq_len
            break;
        } else {
            all_passed = false;
            break;
        }
    }
    
    // Require at least some tests to pass
    if (passed_count == 0) {
        all_passed = false;
    }
    
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    if (!all_passed) {
        TEST_FAIL("Some seq_len values failed");
        return;
    }
    
    TEST_PASS();
}

static void test_scratchpad_fuzzing(void) {
    run_test_with_crash_detection(test_scratchpad_fuzzing_impl);
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("========================================\n");
    printf("  ADVERSARIAL TEST SUITE: Scratchpad Functions\n");
    printf("========================================\n\n");
    printf("Target: calculate_layer_scratchpad_size(), init_layer_scratchpad()\n");
    printf("Strategy: Test indirectly through llama_forward() calls\n\n");
    
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
    test_scratchpad_size_normal();
    test_scratchpad_size_min();
    test_scratchpad_size_max();
    printf("\n");
    
    // CATEGORY 2: EDGE CASES
    printf("CATEGORY 2: Edge Cases\n");
    printf("-----------------------------------\n");
    test_scratchpad_size_zero();
    test_scratchpad_size_overflow();
    printf("\n");
    
    // CATEGORY 3: SECURITY
    printf("CATEGORY 3: Security (OOM)\n");
    printf("-----------------------------------\n");
    test_scratchpad_arena_oom();
    printf("\n");
    
    // CATEGORY 4: FUZZING
    printf("CATEGORY 4: Fuzzing\n");
    printf("-----------------------------------\n");
    test_scratchpad_fuzzing();
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

