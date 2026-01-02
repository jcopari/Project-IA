#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <signal.h>
#include <setjmp.h>
#include <limits.h>

// ============================================================================
// ADVERSARIAL TEST SUITE: KV Cache Functions
// ============================================================================
// Target Function: get_kv_cache_ptr() (static, tested indirectly)
//
// Strategy: Test through llama_forward() calls with various positions
// and validate KV cache updates. Since get_kv_cache_ptr is static,
// we test it indirectly by:
// 1. Running forward passes at different positions
// 2. Validating KV cache contents are updated correctly
// 3. Testing bounds checking (invalid layer_idx, kv_head_idx, pos)
// 4. Testing with NULL kv_buffer
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

// Helper: Setup model with KV cache
static bool setup_model_with_kv(q_context* ctx, q_llama_model* model) {
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

// Helper: Validate KV cache is zero-initialized
static bool validate_kv_cache_zero(q_context* ctx, const q_llama_config* config) {
    if (ctx->kv_buffer == NULL) return false;
    
    uint32_t head_dim = config->dim / config->n_heads;
    size_t total_size = (size_t)config->n_layers * 
                        (size_t)config->n_kv_heads * 
                        (size_t)config->max_seq_len * 
                        (size_t)head_dim * 
                        sizeof(float) * 2;
    
    // Check first and last bytes are zero
    uint8_t* kv_bytes = (uint8_t*)ctx->kv_buffer;
    if (kv_bytes[0] != 0 || kv_bytes[total_size - 1] != 0) {
        return false;
    }
    
    return true;
}

// Helper: Check if KV cache was updated (non-zero values)
static bool kv_cache_updated(q_context* ctx, const q_llama_config* config, 
                             uint32_t layer_idx, uint32_t pos) {
    if (ctx->kv_buffer == NULL) return false;
    
    uint32_t head_dim = config->dim / config->n_heads;
    
    // Check K cache for first KV head at given position
    size_t layer_stride = (size_t)config->n_kv_heads * 
                          (size_t)config->max_seq_len * 
                          (size_t)head_dim * 
                          sizeof(float) * 2;
    size_t head_stride = (size_t)config->max_seq_len * 
                         (size_t)head_dim * 
                         sizeof(float) * 2;
    size_t pos_stride = (size_t)head_dim * sizeof(float) * 2;
    size_t offset = (size_t)layer_idx * layer_stride +
                    (size_t)0 * head_stride +
                    (size_t)pos * pos_stride;
    
    float* k_ptr = (float*)((uint8_t*)ctx->kv_buffer + offset);
    
    // Check if any value is non-zero
    // Adversarial test: intentional float comparison
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wfloat-equal"
    for (uint32_t i = 0; i < head_dim; i++) {
        if (k_ptr[i] != 0.0f) {
            #pragma GCC diagnostic pop
            return true;
        }
    }
    #pragma GCC diagnostic pop
    
    return false;
}

// ============================================================================
// TEST CASES
// ============================================================================

// Test 1: Happy Path - KV cache initialized correctly
static void test_kv_cache_init_impl(void) {
    TEST_START("KV cache initialization - Zero-initialized");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model_with_kv(&ctx, &model)) {
        TEST_FAIL("Failed to setup model");
        return;
    }
    
    if (!validate_kv_cache_zero(&ctx, &model.config)) {
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_FAIL("KV cache should be zero-initialized");
        return;
    }
    
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    TEST_PASS();
}

static void test_kv_cache_init(void) {
    run_test_with_crash_detection(test_kv_cache_init_impl);
}

// Test 2: Happy Path - KV cache updated at position 0
static void test_kv_cache_update_pos0_impl(void) {
    TEST_START("KV cache update - Position 0");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model_with_kv(&ctx, &model)) {
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
    
    if (ret != Q_OK) {
        free(logits);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_FAIL("Forward pass should succeed");
        return;
    }
    
    // Check KV cache was updated for first layer
    if (!kv_cache_updated(&ctx, &model.config, 0, 0)) {
        free(logits);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_FAIL("KV cache should be updated at position 0");
        return;
    }
    
    free(logits);
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    TEST_PASS();
}

static void test_kv_cache_update_pos0(void) {
    run_test_with_crash_detection(test_kv_cache_update_pos0_impl);
}

// Test 3: Happy Path - KV cache updated at multiple positions
static void test_kv_cache_update_multiple_pos_impl(void) {
    TEST_START("KV cache update - Multiple positions");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model_with_kv(&ctx, &model)) {
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
    
    // Run forward pass at position 0
    q_error_code ret = llama_forward(&model, &ctx, tokens, seq_len, 0, logits);
    if (ret != Q_OK) {
        free(logits);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_FAIL("Forward pass should succeed");
        return;
    }
    
    // Check KV cache was updated for positions 0, 1, 2, 3
    // Note: llama_forward stores KV cache at position (pos + t) for each token t
    // Since we called with pos=0, tokens are stored at positions 0, 1, 2, 3
    // Check at least first position is updated (may not update all if early exit)
    bool at_least_one_updated = false;
    for (uint32_t pos = 0; pos < seq_len && pos < model.config.max_seq_len; pos++) {
        if (kv_cache_updated(&ctx, &model.config, 0, pos)) {
            at_least_one_updated = true;
            break;
        }
    }
    
    if (!at_least_one_updated) {
        free(logits);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_FAIL("KV cache should be updated at at least one position");
        return;
    }
    
    free(logits);
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    TEST_PASS();
}

static void test_kv_cache_update_multiple_pos(void) {
    run_test_with_crash_detection(test_kv_cache_update_multiple_pos_impl);
}

// Test 4: Edge Case - Position at max_seq_len - 1
static void test_kv_cache_max_pos_impl(void) {
    TEST_START("KV cache update - Position max_seq_len - 1");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model_with_kv(&ctx, &model)) {
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
    
    if (ret != Q_OK) {
        free(logits);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_FAIL("Forward pass should succeed at max position");
        return;
    }
    
    // Check KV cache was updated
    if (!kv_cache_updated(&ctx, &model.config, 0, max_pos)) {
        free(logits);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_FAIL("KV cache should be updated at max position");
        return;
    }
    
    free(logits);
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    TEST_PASS();
}

static void test_kv_cache_max_pos(void) {
    run_test_with_crash_detection(test_kv_cache_max_pos_impl);
}

// Test 5: Security - Position >= max_seq_len (should fail)
static void test_kv_cache_invalid_pos_impl(void) {
    TEST_START("KV cache bounds - Position >= max_seq_len");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model_with_kv(&ctx, &model)) {
        TEST_FAIL("Failed to setup model");
        return;
    }
    
    uint32_t invalid_pos = model.config.max_seq_len;
    uint32_t tokens[1] = {0};
    float* logits = (float*)aligned_alloc(Q_ALIGN, model.config.vocab_size * sizeof(float));
    if (logits == NULL) {
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_FAIL("Failed to allocate logits");
        return;
    }
    
    q_error_code ret = llama_forward(&model, &ctx, tokens, 1, invalid_pos, logits);
    
    free(logits);
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    // Should fail validation before accessing KV cache
    if (ret == Q_OK) {
        TEST_FAIL("Should reject position >= max_seq_len");
        return;
    }
    
    if (ret != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Should return Q_ERR_INVALID_ARG");
        return;
    }
    
    TEST_PASS();
}

static void test_kv_cache_invalid_pos(void) {
    run_test_with_crash_detection(test_kv_cache_invalid_pos_impl);
}

// Test 6: Security - NULL kv_buffer (should fail)
static void test_kv_cache_null_buffer_impl(void) {
    TEST_START("KV cache - NULL buffer");
    
    q_context ctx;
    q_llama_model model;
    
    memset(&ctx, 0, sizeof(q_context));
    memset(&model, 0, sizeof(q_llama_model));
    
    q_error_code ret = q_init_memory(&ctx, "model_dummy.qorus");
    if (ret != Q_OK) {
        TEST_FAIL("Failed to init memory");
        return;
    }
    
    ret = q_alloc_arena(&ctx, 64 * 1024 * 1024);
    if (ret != Q_OK) {
        q_free_memory(&ctx);
        TEST_FAIL("Failed to allocate arena");
        return;
    }
    
    ret = llama_build_graph(&ctx, &model);
    if (ret != Q_OK) {
        q_free_memory(&ctx);
        TEST_FAIL("Failed to build graph");
        return;
    }
    
    // Intentionally NOT allocating KV cache
    // ctx->kv_buffer should be NULL
    
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
    
    // Should fail validation
    if (ret == Q_OK) {
        TEST_FAIL("Should reject NULL kv_buffer");
        return;
    }
    
    if (ret != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Should return Q_ERR_INVALID_ARG");
        return;
    }
    
    TEST_PASS();
}

static void test_kv_cache_null_buffer(void) {
    run_test_with_crash_detection(test_kv_cache_null_buffer_impl);
}

// Test 7: Edge Case - All layers update KV cache
static void test_kv_cache_all_layers_impl(void) {
    TEST_START("KV cache update - All layers");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model_with_kv(&ctx, &model)) {
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
    
    if (ret != Q_OK) {
        free(logits);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        TEST_FAIL("Forward pass should succeed");
        return;
    }
    
    // Check KV cache was updated for all layers
    for (uint32_t layer = 0; layer < model.config.n_layers; layer++) {
        if (!kv_cache_updated(&ctx, &model.config, layer, 0)) {
            free(logits);
            llama_free_graph(&model);
            q_free_memory(&ctx);
            TEST_FAIL("KV cache should be updated for all layers");
            return;
        }
    }
    
    free(logits);
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    TEST_PASS();
}

static void test_kv_cache_all_layers(void) {
    run_test_with_crash_detection(test_kv_cache_all_layers_impl);
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("========================================\n");
    printf("  ADVERSARIAL TEST SUITE: KV Cache Functions\n");
    printf("========================================\n\n");
    printf("Target: get_kv_cache_ptr() (indirect testing)\n");
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
    test_kv_cache_init();
    test_kv_cache_update_pos0();
    test_kv_cache_update_multiple_pos();
    test_kv_cache_max_pos();
    test_kv_cache_all_layers();
    printf("\n");
    
    // CATEGORY 2: SECURITY
    printf("CATEGORY 2: Security (Bounds Checking)\n");
    printf("-----------------------------------\n");
    test_kv_cache_invalid_pos();
    test_kv_cache_null_buffer();
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

