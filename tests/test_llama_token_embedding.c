#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <signal.h>
#include <setjmp.h>

// ============================================================================
// ADVERSARIAL TEST SUITE: Token Embedding Functions
// ============================================================================
// Target Function: token_embedding_lookup() (static, tested indirectly)
//
// Strategy: Test through llama_forward() calls with various token inputs
// and validate embeddings are correct. Since token_embedding_lookup is
// static, we test it indirectly by:
// 1. Running forward passes with different token IDs
// 2. Validating invalid token IDs are rejected
// 3. Testing edge cases (token = 0, token = vocab_size - 1)
// 4. Testing with tokens >= vocab_size (should fail)
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

// Test 1: Happy Path - Valid token ID (0)
static void test_token_embedding_valid_min_impl(void) {
    TEST_START("Token embedding - Valid token ID (0)");
    
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
        TEST_FAIL("Forward pass should succeed with valid token");
        return;
    }
    
    TEST_PASS();
}

static void test_token_embedding_valid_min(void) {
    run_test_with_crash_detection(test_token_embedding_valid_min_impl);
}

// Test 2: Happy Path - Valid token ID (vocab_size - 1)
static void test_token_embedding_valid_max_impl(void) {
    TEST_START("Token embedding - Valid token ID (vocab_size - 1)");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model(&ctx, &model)) {
        TEST_FAIL("Failed to setup model");
        return;
    }
    
    uint32_t max_token = model.config.vocab_size - 1;
    uint32_t tokens[1] = {max_token};
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
        TEST_FAIL("Forward pass should succeed with max valid token");
        return;
    }
    
    TEST_PASS();
}

static void test_token_embedding_valid_max(void) {
    run_test_with_crash_detection(test_token_embedding_valid_max_impl);
}

// Test 3: Happy Path - Multiple valid tokens
static void test_token_embedding_multiple_valid_impl(void) {
    TEST_START("Token embedding - Multiple valid tokens");
    
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
        TEST_FAIL("Forward pass should succeed with multiple valid tokens");
        return;
    }
    
    TEST_PASS();
}

static void test_token_embedding_multiple_valid(void) {
    run_test_with_crash_detection(test_token_embedding_multiple_valid_impl);
}

// Test 4: Security - Invalid token ID (>= vocab_size)
static void test_token_embedding_invalid_impl(void) {
    TEST_START("Token embedding - Invalid token ID (>= vocab_size)");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model(&ctx, &model)) {
        TEST_FAIL("Failed to setup model");
        return;
    }
    
    uint32_t invalid_token = model.config.vocab_size; // Out of bounds
    uint32_t tokens[1] = {invalid_token};
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
    
    // Should fail validation in token_embedding_lookup
    if (ret == Q_OK) {
        TEST_FAIL("Should reject invalid token ID");
        return;
    }
    
    if (ret != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Should return Q_ERR_INVALID_ARG");
        return;
    }
    
    TEST_PASS();
}

static void test_token_embedding_invalid(void) {
    run_test_with_crash_detection(test_token_embedding_invalid_impl);
}

// Test 5: Security - Multiple tokens with one invalid
static void test_token_embedding_mixed_invalid_impl(void) {
    TEST_START("Token embedding - Multiple tokens with one invalid");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model(&ctx, &model)) {
        TEST_FAIL("Failed to setup model");
        return;
    }
    
    uint32_t seq_len = 4;
    uint32_t tokens[4] = {0, 1, model.config.vocab_size, 3}; // Third token invalid
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
    
    // Should fail validation
    if (ret == Q_OK) {
        TEST_FAIL("Should reject sequence with invalid token");
        return;
    }
    
    if (ret != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Should return Q_ERR_INVALID_ARG");
        return;
    }
    
    TEST_PASS();
}

static void test_token_embedding_mixed_invalid(void) {
    run_test_with_crash_detection(test_token_embedding_mixed_invalid_impl);
}

// Test 6: Edge Case - All tokens same (valid)
static void test_token_embedding_all_same_impl(void) {
    TEST_START("Token embedding - All tokens same (valid)");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model(&ctx, &model)) {
        TEST_FAIL("Failed to setup model");
        return;
    }
    
    uint32_t seq_len = 4;
    uint32_t tokens[4] = {5, 5, 5, 5}; // All same token
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
        TEST_FAIL("Forward pass should succeed with repeated tokens");
        return;
    }
    
    TEST_PASS();
}

static void test_token_embedding_all_same(void) {
    run_test_with_crash_detection(test_token_embedding_all_same_impl);
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("========================================\n");
    printf("  ADVERSARIAL TEST SUITE: Token Embedding Functions\n");
    printf("========================================\n\n");
    printf("Target: token_embedding_lookup() (indirect testing)\n");
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
    test_token_embedding_valid_min();
    test_token_embedding_valid_max();
    test_token_embedding_multiple_valid();
    test_token_embedding_all_same();
    printf("\n");
    
    // CATEGORY 2: SECURITY
    printf("CATEGORY 2: Security (Bounds Checking)\n");
    printf("-----------------------------------\n");
    test_token_embedding_invalid();
    test_token_embedding_mixed_invalid();
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

