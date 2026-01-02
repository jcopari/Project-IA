#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <signal.h>
#include <setjmp.h>

// ============================================================================
// ADVERSARIAL TEST SUITE: llama_free_graph()
// ============================================================================
// Target Function: llama_free_graph()
//
// Strategy: Test cleanup and memory safety
// 1. Test normal cleanup
// 2. Test with NULL model (should not crash)
// 3. Test double-free protection (indirectly)
// 4. Test integration with q_arena_reset()
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
    
    return true;
}

// ============================================================================
// TEST CASES
// ============================================================================

// Test 1: Happy Path - Normal cleanup
static void test_free_graph_normal_impl(void) {
    TEST_START("llama_free_graph - Normal cleanup");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model(&ctx, &model)) {
        TEST_FAIL("Failed to setup model");
        return;
    }
    
    // Verify model is initialized
    if (model.token_embd == NULL || model.layers == NULL) {
        q_free_memory(&ctx);
        TEST_FAIL("Model should be initialized");
        return;
    }
    
    // Free graph
    llama_free_graph(&model);
    
    // Verify pointers are cleared
    if (model.token_embd != NULL || model.layers != NULL) {
        q_free_memory(&ctx);
        TEST_FAIL("Pointers should be cleared after free");
        return;
    }
    
    q_free_memory(&ctx);
    
    TEST_PASS();
}

static void test_free_graph_normal(void) {
    run_test_with_crash_detection(test_free_graph_normal_impl);
}

// Test 2: Security - NULL model pointer (should not crash)
static void test_free_graph_null_impl(void) {
    TEST_START("llama_free_graph - NULL model pointer");
    
    // Should not crash
    llama_free_graph(NULL);
    
    TEST_PASS();
}

static void test_free_graph_null(void) {
    run_test_with_crash_detection(test_free_graph_null_impl);
}

// Test 3: Security - Double free protection (indirect)
static void test_free_graph_double_free_impl(void) {
    TEST_START("llama_free_graph - Double free protection");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model(&ctx, &model)) {
        TEST_FAIL("Failed to setup model");
        return;
    }
    
    // Free first time
    llama_free_graph(&model);
    
    // Free second time - should not crash
    llama_free_graph(&model);
    
    q_free_memory(&ctx);
    
    TEST_PASS();
}

static void test_free_graph_double_free(void) {
    run_test_with_crash_detection(test_free_graph_double_free_impl);
}

// Test 4: Integration - Free after arena reset
static void test_free_graph_after_reset_impl(void) {
    TEST_START("llama_free_graph - After arena reset");
    
    q_context ctx;
    q_llama_model model;
    
    if (!setup_model(&ctx, &model)) {
        TEST_FAIL("Failed to setup model");
        return;
    }
    
    // Reset arena (should not affect model structures before watermark)
    q_arena_reset(&ctx);
    
    // Free graph - should still work
    llama_free_graph(&model);
    
    q_free_memory(&ctx);
    
    TEST_PASS();
}

static void test_free_graph_after_reset(void) {
    run_test_with_crash_detection(test_free_graph_after_reset_impl);
}

// Test 5: Edge Case - Free uninitialized model
static void test_free_graph_uninitialized_impl(void) {
    TEST_START("llama_free_graph - Uninitialized model");
    
    q_llama_model model;
    memset(&model, 0, sizeof(q_llama_model));
    
    // Should not crash
    llama_free_graph(&model);
    
    TEST_PASS();
}

static void test_free_graph_uninitialized(void) {
    run_test_with_crash_detection(test_free_graph_uninitialized_impl);
}

// Test 6: Edge Case - Free partially initialized model
static void test_free_graph_partial_impl(void) {
    TEST_START("llama_free_graph - Partially initialized model");
    
    q_llama_model model;
    memset(&model, 0, sizeof(q_llama_model));
    
    // Set some fields but not others
    model.token_embd = (q_tensor*)0xDEADBEEF; // Invalid pointer
    model.layers = NULL;
    
    // Should not crash (should clear pointers safely)
    llama_free_graph(&model);
    
    TEST_PASS();
}

static void test_free_graph_partial(void) {
    run_test_with_crash_detection(test_free_graph_partial_impl);
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("========================================\n");
    printf("  ADVERSARIAL TEST SUITE: llama_free_graph()\n");
    printf("========================================\n\n");
    
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
    test_free_graph_normal();
    printf("\n");
    
    // CATEGORY 2: NULL/UNDEFINED
    printf("CATEGORY 2: Null/Undefined Inputs\n");
    printf("-----------------------------------\n");
    test_free_graph_null();
    test_free_graph_uninitialized();
    test_free_graph_partial();
    printf("\n");
    
    // CATEGORY 3: SECURITY
    printf("CATEGORY 3: Security (Double Free)\n");
    printf("-----------------------------------\n");
    test_free_graph_double_free();
    printf("\n");
    
    // CATEGORY 4: INTEGRATION
    printf("CATEGORY 4: Integration\n");
    printf("-----------------------------------\n");
    test_free_graph_after_reset();
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

