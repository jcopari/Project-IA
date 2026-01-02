// ============================================================================
// END-TO-END TESTS: Text Generation (FASE 4.2)
// ============================================================================
// Testes completos do pipeline de geração de texto
// Valida: init → build → encode → generate → decode
// ============================================================================

#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>
#include <time.h>
#include <math.h>

// ============================================================================
// TEST CONFIGURATION
// ============================================================================

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;
static int tests_crashed = 0;

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

// Cleanup macros
#define CLEANUP_CONTEXT(ctx) do { \
    if ((ctx)->weights_mmap != NULL || (ctx)->scratch_buffer != NULL || (ctx)->kv_buffer != NULL) { \
        q_free_memory(ctx); \
    } \
} while(0)

#define CLEANUP_MODEL(model) do { \
    if ((model) != NULL) { \
        q_llama_model* _model_ptr = (model); \
        if (_model_ptr->token_embd != NULL || _model_ptr->layers != NULL) { \
            llama_free_graph(_model_ptr); \
        } \
    } \
} while(0)

#define CLEANUP_TOKENIZER(tok) do { \
    if ((tok) != NULL) { \
        q_tokenizer* _tok_ptr = (tok); \
        if (_tok_ptr->vocab != NULL || _tok_ptr->merges != NULL) { \
            q_tokenizer_free(_tok_ptr); \
        } \
    } \
} while(0)

#define CLEANUP_ALL(ctx, model, tok) do { \
    if ((model) != NULL) { CLEANUP_MODEL(model); } \
    if ((tok) != NULL) { CLEANUP_TOKENIZER(tok); } \
    CLEANUP_CONTEXT(ctx); \
} while(0)

// Helper: Ensure dummy model exists
static bool ensure_dummy_model(void) {
    FILE* f = fopen("model_dummy.qorus", "rb");
    if (f != NULL) {
        fclose(f);
        return true;
    }
    
    printf("  Generating dummy model...\n");
    int ret = system("python3 tools/convert_llama.py model_dummy.qorus 2 > /dev/null 2>&1");
    return (ret == 0);
}

// Helper: Ensure tokenizer exists
static bool ensure_tokenizer(void) {
    FILE* f = fopen("tokenizer.bin", "rb");
    if (f != NULL) {
        fclose(f);
        return true;
    }
    
    printf("  Generating tokenizer...\n");
    int ret = system("python3 tools/convert_llama.py --tokenizer tokenizer.bin > /dev/null 2>&1");
    return (ret == 0);
}

// Helper: Calculate KV cache size
static size_t calculate_kv_cache_size(const q_llama_config* config) {
    uint32_t head_dim = config->dim / config->n_heads;
    size_t kv_size = (size_t)config->n_layers * 
                     (size_t)config->n_kv_heads * 
                     (size_t)config->max_seq_len * 
                     (size_t)head_dim * 
                     sizeof(float) * 2; // K + V
    return Q_ALIGN_SIZE(kv_size);
}

// ============================================================================
// TEST CASES
// ============================================================================

// Test 1: Full generation pipeline
static void test_e2e_full_generation(void) {
    TEST_START("E2E - Full generation pipeline: init → build → encode → generate → decode");
    
    if (!ensure_dummy_model() || !ensure_tokenizer()) {
        TEST_FAIL("Cannot generate dummy model/tokenizer");
        return;
    }
    
    q_context ctx = {0};
    q_llama_model model = {0};
    q_tokenizer tokenizer = {0};
    q_error_code ret;
    
    // STEP 1: Initialize memory
    ret = q_init_memory(&ctx, "model_dummy.qorus");
    if (ret != Q_OK) {
        TEST_FAIL_MSG("q_init_memory failed: %d", ret);
        return;
    }
    
    // STEP 2: Allocate arena
    ret = q_alloc_arena(&ctx, 64 * 1024 * 1024);  // 64MB
    if (ret != Q_OK) {
        TEST_FAIL_MSG("q_alloc_arena failed: %d", ret);
        CLEANUP_CONTEXT(&ctx);
        return;
    }
    
    // STEP 3: Build graph
    ret = llama_build_graph(&ctx, &model);
    if (ret != Q_OK) {
        TEST_FAIL_MSG("llama_build_graph failed: %d", ret);
        CLEANUP_CONTEXT(&ctx);
        return;
    }
    
    // STEP 4: Allocate KV cache
    size_t kv_size = calculate_kv_cache_size(&model.config);
    ret = q_alloc_kv_cache(&ctx, kv_size);
    if (ret != Q_OK) {
        TEST_FAIL_MSG("q_alloc_kv_cache failed: %d", ret);
        CLEANUP_ALL(&ctx, &model, NULL);
        return;
    }
    
    // STEP 5: Load tokenizer
    ret = q_tokenizer_load(&tokenizer, "tokenizer.bin");
    if (ret != Q_OK) {
        TEST_FAIL_MSG("q_tokenizer_load failed: %d", ret);
        CLEANUP_ALL(&ctx, &model, NULL);
        return;
    }
    
    // STEP 6: Encode prompt
    const char* prompt = "Hello";
    uint32_t prompt_tokens[256];
    uint32_t num_prompt_tokens = 0;
    ret = q_tokenizer_encode(&tokenizer, prompt, prompt_tokens, &num_prompt_tokens, 256, true, false);
    if (ret != Q_OK) {
        TEST_FAIL_MSG("q_tokenizer_encode failed: %d", ret);
        CLEANUP_ALL(&ctx, &model, &tokenizer);
        return;
    }
    
    if (num_prompt_tokens == 0) {
        TEST_FAIL("Prompt encoded to zero tokens");
        CLEANUP_ALL(&ctx, &model, &tokenizer);
        return;
    }
    
    // STEP 7: Setup generation state
    uint32_t generated_tokens[256];
    q_generation_state gen_state = {
        .ctx = &ctx,
        .model = &model,
        .tokenizer = &tokenizer,
        .prompt_tokens = prompt_tokens,
        .num_prompt_tokens = num_prompt_tokens,
        .generated_tokens = generated_tokens,
        .num_generated_tokens = 0,
        .max_tokens = 10,  // Gerar até 10 tokens
        .temperature = 0.8f,
        .top_k = 40,
        .top_p = 0.9f,
        .current_pos = 0
    };
    
    // STEP 8: Generate text
    ret = q_generate(&gen_state);
    if (ret != Q_OK) {
        TEST_FAIL_MSG("q_generate failed: %d", ret);
        CLEANUP_ALL(&ctx, &model, &tokenizer);
        return;
    }
    
    // STEP 9: Validate generation
    if (gen_state.num_generated_tokens == 0) {
        TEST_FAIL("No tokens generated");
        CLEANUP_ALL(&ctx, &model, &tokenizer);
        return;
    }
    
    if (gen_state.num_generated_tokens > gen_state.max_tokens) {
        TEST_FAIL("Generated more tokens than max_tokens");
        CLEANUP_ALL(&ctx, &model, &tokenizer);
        return;
    }
    
    // STEP 10: Decode generated tokens
    char generated_text[2048];
    ret = q_tokenizer_decode(&tokenizer, gen_state.generated_tokens, 
                             gen_state.num_generated_tokens, generated_text, sizeof(generated_text));
    if (ret != Q_OK) {
        TEST_FAIL_MSG("q_tokenizer_decode failed: %d", ret);
        CLEANUP_ALL(&ctx, &model, &tokenizer);
        return;
    }
    
    printf("    Generated %u tokens: \"%s\"\n", gen_state.num_generated_tokens, generated_text);
    
    CLEANUP_ALL(&ctx, &model, &tokenizer);
    TEST_PASS();
}

// Test 2: Greedy sampling (temperature = 0.0)
static void test_e2e_greedy_sampling(void) {
    TEST_START("E2E - Greedy sampling (temperature = 0.0)");
    
    if (!ensure_dummy_model() || !ensure_tokenizer()) {
        TEST_FAIL("Cannot generate dummy model/tokenizer");
        return;
    }
    
    q_context ctx = {0};
    q_llama_model model = {0};
    q_tokenizer tokenizer = {0};
    q_error_code ret;
    
    ret = q_init_memory(&ctx, "model_dummy.qorus");
    if (ret != Q_OK) {
        TEST_FAIL("Cannot initialize memory");
        return;
    }
    
    ret = q_alloc_arena(&ctx, 64 * 1024 * 1024);
    if (ret != Q_OK) {
        TEST_FAIL("Cannot allocate arena");
        CLEANUP_CONTEXT(&ctx);
        return;
    }
    
    ret = llama_build_graph(&ctx, &model);
    if (ret != Q_OK) {
        TEST_FAIL("Cannot build graph");
        CLEANUP_CONTEXT(&ctx);
        return;
    }
    
    size_t kv_size = calculate_kv_cache_size(&model.config);
    ret = q_alloc_kv_cache(&ctx, kv_size);
    if (ret != Q_OK) {
        TEST_FAIL("Cannot allocate KV cache");
        CLEANUP_ALL(&ctx, &model, NULL);
        return;
    }
    
    ret = q_tokenizer_load(&tokenizer, "tokenizer.bin");
    if (ret != Q_OK) {
        TEST_FAIL("Cannot load tokenizer");
        CLEANUP_ALL(&ctx, &model, NULL);
        return;
    }
    
    const char* prompt = "Test";
    uint32_t prompt_tokens[256];
    uint32_t num_prompt_tokens = 0;
    ret = q_tokenizer_encode(&tokenizer, prompt, prompt_tokens, &num_prompt_tokens, 256, true, false);
    if (ret != Q_OK || num_prompt_tokens == 0) {
        TEST_FAIL("Cannot encode prompt");
        CLEANUP_ALL(&ctx, &model, &tokenizer);
        return;
    }
    
    uint32_t generated_tokens[256];
    q_generation_state gen_state = {
        .ctx = &ctx,
        .model = &model,
        .tokenizer = &tokenizer,
        .prompt_tokens = prompt_tokens,
        .num_prompt_tokens = num_prompt_tokens,
        .generated_tokens = generated_tokens,
        .num_generated_tokens = 0,
        .max_tokens = 5,
        .temperature = 0.0f,  // Greedy
        .top_k = 0,
        .top_p = 0.0f,
        .current_pos = 0
    };
    
    ret = q_generate(&gen_state);
    if (ret != Q_OK) {
        TEST_FAIL_MSG("q_generate failed: %d", ret);
        CLEANUP_ALL(&ctx, &model, &tokenizer);
        return;
    }
    
    if (gen_state.num_generated_tokens == 0) {
        TEST_FAIL("No tokens generated");
        CLEANUP_ALL(&ctx, &model, &tokenizer);
        return;
    }
    
    CLEANUP_ALL(&ctx, &model, &tokenizer);
    TEST_PASS();
}

// Test 3: Multiple generations (state reuse)
static void test_e2e_multiple_generations(void) {
    TEST_START("E2E - Multiple generations with state reuse");
    
    if (!ensure_dummy_model() || !ensure_tokenizer()) {
        TEST_FAIL("Cannot generate dummy model/tokenizer");
        return;
    }
    
    q_context ctx = {0};
    q_llama_model model = {0};
    q_tokenizer tokenizer = {0};
    q_error_code ret;
    
    ret = q_init_memory(&ctx, "model_dummy.qorus");
    if (ret != Q_OK) {
        TEST_FAIL("Cannot initialize memory");
        return;
    }
    
    ret = q_alloc_arena(&ctx, 64 * 1024 * 1024);
    if (ret != Q_OK) {
        TEST_FAIL("Cannot allocate arena");
        CLEANUP_CONTEXT(&ctx);
        return;
    }
    
    ret = llama_build_graph(&ctx, &model);
    if (ret != Q_OK) {
        TEST_FAIL("Cannot build graph");
        CLEANUP_CONTEXT(&ctx);
        return;
    }
    
    size_t kv_size = calculate_kv_cache_size(&model.config);
    ret = q_alloc_kv_cache(&ctx, kv_size);
    if (ret != Q_OK) {
        TEST_FAIL("Cannot allocate KV cache");
        CLEANUP_ALL(&ctx, &model, NULL);
        return;
    }
    
    ret = q_tokenizer_load(&tokenizer, "tokenizer.bin");
    if (ret != Q_OK) {
        TEST_FAIL("Cannot load tokenizer");
        CLEANUP_ALL(&ctx, &model, NULL);
        return;
    }
    
    // First generation
    const char* prompt1 = "Hello";
    uint32_t prompt_tokens1[256];
    uint32_t num_prompt_tokens1 = 0;
    ret = q_tokenizer_encode(&tokenizer, prompt1, prompt_tokens1, &num_prompt_tokens1, 256, true, false);
    if (ret != Q_OK || num_prompt_tokens1 == 0) {
        TEST_FAIL("Cannot encode prompt 1");
        CLEANUP_ALL(&ctx, &model, &tokenizer);
        return;
    }
    
    uint32_t generated_tokens1[256];
    q_generation_state gen_state1 = {
        .ctx = &ctx,
        .model = &model,
        .tokenizer = &tokenizer,
        .prompt_tokens = prompt_tokens1,
        .num_prompt_tokens = num_prompt_tokens1,
        .generated_tokens = generated_tokens1,
        .num_generated_tokens = 0,
        .max_tokens = 5,
        .temperature = 0.8f,
        .top_k = 40,
        .top_p = 0.9f,
        .current_pos = 0
    };
    
    ret = q_generate(&gen_state1);
    if (ret != Q_OK) {
        TEST_FAIL_MSG("First generation failed: %d", ret);
        CLEANUP_ALL(&ctx, &model, &tokenizer);
        return;
    }
    
    // Second generation (reuse context)
    const char* prompt2 = "World";
    uint32_t prompt_tokens2[256];
    uint32_t num_prompt_tokens2 = 0;
    ret = q_tokenizer_encode(&tokenizer, prompt2, prompt_tokens2, &num_prompt_tokens2, 256, true, false);
    if (ret != Q_OK || num_prompt_tokens2 == 0) {
        TEST_FAIL("Cannot encode prompt 2");
        CLEANUP_ALL(&ctx, &model, &tokenizer);
        return;
    }
    
    uint32_t generated_tokens2[256];
    q_generation_state gen_state2 = {
        .ctx = &ctx,
        .model = &model,
        .tokenizer = &tokenizer,
        .prompt_tokens = prompt_tokens2,
        .num_prompt_tokens = num_prompt_tokens2,
        .generated_tokens = generated_tokens2,
        .num_generated_tokens = 0,
        .max_tokens = 5,
        .temperature = 0.8f,
        .top_k = 40,
        .top_p = 0.9f,
        .current_pos = 0
    };
    
    ret = q_generate(&gen_state2);
    if (ret != Q_OK) {
        TEST_FAIL_MSG("Second generation failed: %d", ret);
        CLEANUP_ALL(&ctx, &model, &tokenizer);
        return;
    }
    
    if (gen_state1.num_generated_tokens == 0 || gen_state2.num_generated_tokens == 0) {
        TEST_FAIL("One or both generations produced no tokens");
        CLEANUP_ALL(&ctx, &model, &tokenizer);
        return;
    }
    
    CLEANUP_ALL(&ctx, &model, &tokenizer);
    TEST_PASS();
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstack-usage="
int main(void) {
    printf("========================================\n");
    printf("  END-TO-END GENERATION TEST SUITE\n");
    printf("========================================\n\n");
    printf("Strategy: Validate complete text generation pipeline\n");
    printf("Following TDD methodology: Full pipeline validation\n\n");
    
    // Cleanup handled by CLEANUP_ALL macros in each test
    
    tests_run = 0;
    tests_passed = 0;
    tests_failed = 0;
    tests_crashed = 0;
    
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGABRT, crash_handler);
    
    // Run tests
    if (setjmp(crash_jmp_buf) == 0) {
        test_e2e_full_generation();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_e2e_greedy_sampling();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_e2e_multiple_generations();
    } else {
        TEST_CRASH();
    }
    
    printf("\n========================================\n");
    printf("  TEST SUMMARY\n");
    printf("========================================\n");
    printf("Tests run:    %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_failed);
    printf("Tests crashed: %d\n", tests_crashed);
    printf("========================================\n");
    
    if (tests_failed == 0 && tests_crashed == 0) {
        printf("  ALL TESTS PASSED ✓\n");
        printf("========================================\n");
        return 0;
    } else {
        printf("  SOME TESTS FAILED ✗\n");
        printf("========================================\n");
        return 1;
    }
}
#pragma GCC diagnostic pop

