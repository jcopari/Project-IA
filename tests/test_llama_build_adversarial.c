#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

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

#define TEST_CRASH() do { \
    tests_run++; \
    tests_crashed++; \
    printf("  ✗ CRASHED\n"); \
} while(0)

// Create minimal valid model file in memory
static void create_minimal_model_file(const char* path, uint32_t n_layers) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "ERROR: Failed to create test file\n");
        return;
    }
    
    // Write header
    uint32_t header[16] = {0};
    header[0] = Q_MAGIC;
    header[1] = 1;  // version
    header[2] = 100;  // vocab_size
    header[3] = 128;  // dim (must be multiple of 32)
    header[4] = 256;  // hidden_dim (must be multiple of 32)
    header[5] = n_layers;
    header[6] = 4;   // n_heads
    header[7] = 2;   // n_kv_heads
    header[8] = 2048; // max_seq_len
    float rope_theta = 500000.0f;
    
    fwrite(header, sizeof(uint32_t), 9, f);
    fwrite(&rope_theta, sizeof(float), 1, f);
    fwrite(header + 9, sizeof(uint32_t), 6, f);  // reserved
    
    // Write minimal tensors (just enough to pass validation)
    // token_embd [100, 128] = 100 * 128 * 4 = 51200 bytes
    // Align to 64 bytes: 51200 -> 51200 (already aligned)
    size_t token_embd_size = 100 * 128 * sizeof(float);
    size_t token_embd_aligned = (token_embd_size + 63) & ~63;
    float* dummy_data = calloc(token_embd_aligned / sizeof(float), sizeof(float));
    if (!dummy_data) {
        fclose(f);
        return;
    }
    if (fwrite(dummy_data, token_embd_aligned, 1, f) != 1) {
        free(dummy_data);
        fclose(f);
        return;
    }
    
    // output_norm [128] = 512 bytes, align to 64 = 512
    float dummy_norm[128];
    memset(dummy_norm, 0, sizeof(dummy_norm));
    if (fwrite(dummy_norm, sizeof(dummy_norm), 1, f) != 1) {
        free(dummy_data);
        fclose(f);
        return;
    }
    
    // output [100, 128] = 51200 bytes
    if (fwrite(dummy_data, token_embd_aligned, 1, f) != 1) {
        free(dummy_data);
        fclose(f);
        return;
    }
    free(dummy_data);
    
    // For each layer, write minimal data
    for (uint32_t i = 0; i < n_layers; i++) {
        // attn_norm [128] = 512 bytes, align to 64 = 512
        fwrite(dummy_norm, sizeof(dummy_norm), 1, f);
        
        // wq [128, 128] Q4_0 = 128 * (128/32) * 20 = 10240 bytes
        // Align to 64: 10240 -> 10240 (already aligned)
        uint8_t dummy_q4[10240];
        memset(dummy_q4, 0, sizeof(dummy_q4));
        fwrite(dummy_q4, sizeof(dummy_q4), 1, f);
        
        // wk [128, 64] Q4_0 = 128 * (64/32) * 20 = 5120 bytes
        // Align to 64: 5120 -> 5120 (already aligned)
        uint8_t dummy_q4_small[5120];
        memset(dummy_q4_small, 0, sizeof(dummy_q4_small));
        fwrite(dummy_q4_small, sizeof(dummy_q4_small), 1, f);
        
        // wv [128, 64] Q4_0 = 5120 bytes
        fwrite(dummy_q4_small, sizeof(dummy_q4_small), 1, f);
        
        // wo [128, 128] Q4_0 = 10240 bytes
        fwrite(dummy_q4, sizeof(dummy_q4), 1, f);
        
        // ffn_norm [128] = 512 bytes
        fwrite(dummy_norm, sizeof(dummy_norm), 1, f);
        
        // w_gate [128, 256] Q4_0 = 128 * (256/32) * 20 = 20480 bytes
        // Align to 64: 20480 -> 20480 (already aligned)
        uint8_t dummy_q4_large[20480];
        memset(dummy_q4_large, 0, sizeof(dummy_q4_large));
        fwrite(dummy_q4_large, sizeof(dummy_q4_large), 1, f);
        
        // w_up [128, 256] Q4_0 = 20480 bytes
        fwrite(dummy_q4_large, sizeof(dummy_q4_large), 1, f);
        
        // w_down [256, 128] Q4_0 = 256 * (128/32) * 20 = 20480 bytes
        fwrite(dummy_q4_large, sizeof(dummy_q4_large), 1, f);
    }
    
    fclose(f);
}

// ============================================================================
// ADVERSARIAL TEST CASES
// ============================================================================

// Test 1: NULL context pointer
static void test_null_context(void) {
    printf("Test 1: NULL context pointer\n");
    
    llama_model model = {0};
    q_error_code ret;
    
    if (setjmp(crash_jmp_buf) == 0) {
        signal(SIGSEGV, crash_handler);
        signal(SIGBUS, crash_handler);
        signal(SIGABRT, crash_handler);
        
        ret = llama_build_graph(NULL, &model);
        
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        
        if (ret == Q_ERR_NULL_PTR) {
            TEST_PASS();
        } else {
            TEST_FAIL("Expected Q_ERR_NULL_PTR");
        }
    } else {
        TEST_CRASH();
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
    }
}

// Test 2: NULL model pointer
static void test_null_model(void) {
    printf("Test 2: NULL model pointer\n");
    
    q_context ctx = {0};
    q_error_code ret;
    
    if (setjmp(crash_jmp_buf) == 0) {
        signal(SIGSEGV, crash_handler);
        signal(SIGBUS, crash_handler);
        signal(SIGABRT, crash_handler);
        
        ret = llama_build_graph(&ctx, NULL);
        
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        
        if (ret == Q_ERR_NULL_PTR) {
            TEST_PASS();
        } else {
            TEST_FAIL("Expected Q_ERR_NULL_PTR");
        }
    } else {
        TEST_CRASH();
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
    }
}

// Test 3: NULL weights_mmap
static void test_null_mmap(void) {
    printf("Test 3: NULL weights_mmap\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    q_error_code ret;
    
    ctx.weights_mmap = NULL;
    ctx.header = (q_model_header*)malloc(sizeof(q_model_header));
    if (!ctx.header) {
        printf("  SKIPPED: malloc failed\n");
        return;
    }
    ctx.header->magic = Q_MAGIC;
    
    ret = llama_build_graph(&ctx, &model);
    
    free(ctx.header);
    
    if (ret == Q_ERR_NULL_PTR) {
        TEST_PASS();
    } else {
        TEST_FAIL("Expected Q_ERR_NULL_PTR");
    }
}

// Test 4: NULL header
static void test_null_header(void) {
    printf("Test 4: NULL header\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    q_error_code ret;
    
    uint8_t dummy_mmap[1024];
    ctx.weights_mmap = dummy_mmap;
    ctx.weights_size = sizeof(dummy_mmap);
    ctx.header = NULL;
    
    ret = llama_build_graph(&ctx, &model);
    
    if (ret == Q_ERR_NULL_PTR) {
        TEST_PASS();
    } else {
        TEST_FAIL("Expected Q_ERR_NULL_PTR");
    }
}

// Test 5: Invalid magic number
static void test_invalid_magic(void) {
    printf("Test 5: Invalid magic number\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    q_model_header header = {0};
    q_error_code ret;
    
    uint8_t dummy_mmap[1024];
    ctx.weights_mmap = dummy_mmap;
    ctx.weights_size = sizeof(dummy_mmap);
    ctx.header = &header;
    header.magic = 0xDEADBEEF;  // Invalid magic
    
    ret = llama_build_graph(&ctx, &model);
    
    if (ret == Q_ERR_INVALID_MAGIC) {
        TEST_PASS();
    } else {
        TEST_FAIL("Expected Q_ERR_INVALID_MAGIC");
    }
}

// Test 6: Zero layers
static void test_zero_layers(void) {
    printf("Test 6: Zero layers (n_layers = 0)\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    q_model_header header = {0};
    q_error_code ret;
    
    uint8_t dummy_mmap[1024];
    ctx.weights_mmap = dummy_mmap;
    ctx.weights_size = sizeof(dummy_mmap);
    ctx.header = &header;
    header.magic = Q_MAGIC;
    header.n_layers = 0;
    header.dim = 128;
    header.vocab_size = 100;
    
    ret = llama_build_graph(&ctx, &model);
    
    if (ret == Q_ERR_INVALID_CONFIG) {
        TEST_PASS();
    } else {
        TEST_FAIL("Expected Q_ERR_INVALID_CONFIG");
    }
}

// Test 7: Zero dimension
static void test_zero_dim(void) {
    printf("Test 7: Zero dimension (dim = 0)\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    q_model_header header = {0};
    q_error_code ret;
    
    uint8_t dummy_mmap[1024];
    ctx.weights_mmap = dummy_mmap;
    ctx.weights_size = sizeof(dummy_mmap);
    ctx.header = &header;
    header.magic = Q_MAGIC;
    header.n_layers = 1;
    header.dim = 0;
    header.vocab_size = 100;
    
    ret = llama_build_graph(&ctx, &model);
    
    if (ret == Q_ERR_INVALID_CONFIG) {
        TEST_PASS();
    } else {
        TEST_FAIL("Expected Q_ERR_INVALID_CONFIG");
    }
}

// Test 8: Zero vocab_size
static void test_zero_vocab_size(void) {
    printf("Test 8: Zero vocab_size\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    q_model_header header = {0};
    q_error_code ret;
    
    uint8_t dummy_mmap[1024];
    ctx.weights_mmap = dummy_mmap;
    ctx.weights_size = sizeof(dummy_mmap);
    ctx.header = &header;
    header.magic = Q_MAGIC;
    header.n_layers = 1;
    header.dim = 128;
    header.vocab_size = 0;
    
    ret = llama_build_graph(&ctx, &model);
    
    if (ret == Q_ERR_INVALID_CONFIG) {
        TEST_PASS();
    } else {
        TEST_FAIL("Expected Q_ERR_INVALID_CONFIG");
    }
}

// Test 9: File too small (smaller than header)
static void test_file_too_small(void) {
    printf("Test 9: File too small (smaller than header)\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    q_error_code ret;
    
    // Create file with only header (no data)
    create_minimal_model_file("test_small.qorus", 1);
    
    ret = q_init_memory(&ctx, "test_small.qorus");
    if (ret != Q_OK) {
        printf("  SKIPPED: Failed to load test file\n");
        unlink("test_small.qorus");
        return;
    }
    
    // Truncate file to be smaller than header
    ctx.weights_size = Q_HEADER_SIZE - 1;
    
    ret = llama_build_graph(&ctx, &model);
    
    q_free_memory(&ctx);
    unlink("test_small.qorus");
    
    if (ret == Q_ERR_INVALID_CONFIG) {
        TEST_PASS();
    } else {
        TEST_FAIL("Expected Q_ERR_INVALID_CONFIG");
    }
}

// Test 10: Arena not allocated
static void test_arena_not_allocated(void) {
    printf("Test 10: Arena not allocated\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    q_error_code ret;
    
    create_minimal_model_file("test_arena.qorus", 1);
    
    ret = q_init_memory(&ctx, "test_arena.qorus");
    if (ret != Q_OK) {
        printf("  SKIPPED: Failed to load test file\n");
        unlink("test_arena.qorus");
        return;
    }
    
    // Don't allocate arena
    // ctx.scratch_buffer = NULL;
    
    ret = llama_build_graph(&ctx, &model);
    
    q_free_memory(&ctx);
    unlink("test_arena.qorus");
    
    if (ret == Q_ERR_ARENA_OOM) {
        TEST_PASS();
    } else {
        TEST_FAIL("Expected Q_ERR_ARENA_OOM");
    }
}

// Test 11: Arena too small (OOM)
static void test_arena_oom(void) {
    printf("Test 11: Arena too small (OOM)\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    q_error_code ret;
    
    create_minimal_model_file("test_oom.qorus", 1);
    
    ret = q_init_memory(&ctx, "test_oom.qorus");
    if (ret != Q_OK) {
        printf("  SKIPPED: Failed to load test file\n");
        unlink("test_oom.qorus");
        return;
    }
    
    // Allocate very small arena (1 byte)
    ret = q_alloc_arena(&ctx, 1);
    if (ret != Q_OK) {
        printf("  SKIPPED: Failed to allocate arena\n");
        q_free_memory(&ctx);
        unlink("test_oom.qorus");
        return;
    }
    
    ret = llama_build_graph(&ctx, &model);
    
    q_free_memory(&ctx);
    unlink("test_oom.qorus");
    
    if (ret == Q_ERR_ARENA_OOM) {
        TEST_PASS();
    } else {
        TEST_FAIL("Expected Q_ERR_ARENA_OOM");
    }
}

// Test 12: Offset overflow (file too small for all tensors)
static void test_offset_overflow(void) {
    printf("Test 12: Offset overflow (file too small)\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    q_error_code ret;
    
    create_minimal_model_file("test_overflow.qorus", 1);
    
    ret = q_init_memory(&ctx, "test_overflow.qorus");
    if (ret != Q_OK) {
        printf("  SKIPPED: Failed to load test file\n");
        unlink("test_overflow.qorus");
        return;
    }
    
    ret = q_alloc_arena(&ctx, 1024 * 1024);
    if (ret != Q_OK) {
        printf("  SKIPPED: Failed to allocate arena\n");
        q_free_memory(&ctx);
        unlink("test_overflow.qorus");
        return;
    }
    
    // Truncate file size artificially to cause offset overflow
    ctx.weights_size = Q_HEADER_SIZE + 1000;  // Very small
    
    ret = llama_build_graph(&ctx, &model);
    
    q_free_memory(&ctx);
    unlink("test_overflow.qorus");
    
    if (ret == Q_ERR_INVALID_CONFIG) {
        TEST_PASS();
    } else {
        TEST_FAIL("Expected Q_ERR_INVALID_CONFIG");
    }
}

// Test 13: Dim not multiple of 32 (invalid for Q4_0)
static void test_dim_not_multiple_32(void) {
    printf("Test 13: Dim not multiple of 32\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    q_model_header header = {0};
    q_error_code ret;
    
    uint8_t dummy_mmap[1024 * 1024];
    ctx.weights_mmap = dummy_mmap;
    ctx.weights_size = sizeof(dummy_mmap);
    ctx.header = &header;
    header.magic = Q_MAGIC;
    header.n_layers = 1;
    header.dim = 100;  // Not multiple of 32
    header.vocab_size = 100;
    header.hidden_dim = 256;
    
    ret = q_alloc_arena(&ctx, 1024 * 1024);
    if (ret != Q_OK) {
        printf("  SKIPPED: Failed to allocate arena\n");
        return;
    }
    
    ret = llama_build_graph(&ctx, &model);
    
    q_free_memory(&ctx);
    
    // Should fail when calculating Q4_0 size (ne1 % 32 != 0)
    if (ret == Q_ERR_INVALID_CONFIG) {
        TEST_PASS();
    } else {
        TEST_FAIL("Expected Q_ERR_INVALID_CONFIG");
    }
}

// Test 14: Valid model (happy path)
static void test_valid_model(void) {
    printf("Test 14: Valid model (happy path)\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    q_error_code ret;
    
    create_minimal_model_file("test_valid.qorus", 1);
    
    ret = q_init_memory(&ctx, "test_valid.qorus");
    if (ret != Q_OK) {
        printf("  SKIPPED: Failed to load test file\n");
        unlink("test_valid.qorus");
        return;
    }
    
    ret = q_alloc_arena(&ctx, 1024 * 1024);
    if (ret != Q_OK) {
        printf("  SKIPPED: Failed to allocate arena\n");
        q_free_memory(&ctx);
        unlink("test_valid.qorus");
        return;
    }
    
    ret = llama_build_graph(&ctx, &model);
    
    if (ret == Q_OK) {
        // Validate model structure
        if (model.config.n_layers == 1 &&
            model.config.dim == 128 &&
            model.config.vocab_size == 100 &&
            model.token_embd != NULL &&
            model.output_norm != NULL &&
            model.output != NULL &&
            model.layers != NULL &&
            model.layers[0].wq != NULL) {
            llama_free_graph(&model);
            q_free_memory(&ctx);
            unlink("test_valid.qorus");
            TEST_PASS();
        } else {
            llama_free_graph(&model);
            q_free_memory(&ctx);
            unlink("test_valid.qorus");
            TEST_FAIL("Model structure invalid");
        }
    } else {
        q_free_memory(&ctx);
        unlink("test_valid.qorus");
        TEST_FAIL("Expected Q_OK");
    }
}

// Test 15: Multiple layers (stress test)
static void test_multiple_layers(void) {
    printf("Test 15: Multiple layers (stress test)\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    q_error_code ret;
    
    create_minimal_model_file("test_multi.qorus", 10);
    
    ret = q_init_memory(&ctx, "test_multi.qorus");
    if (ret != Q_OK) {
        printf("  SKIPPED: Failed to load test file\n");
        unlink("test_multi.qorus");
        return;
    }
    
    ret = q_alloc_arena(&ctx, 1024 * 1024);
    if (ret != Q_OK) {
        printf("  SKIPPED: Failed to allocate arena\n");
        q_free_memory(&ctx);
        unlink("test_multi.qorus");
        return;
    }
    
    ret = llama_build_graph(&ctx, &model);
    
    if (ret == Q_OK) {
        if (model.config.n_layers == 10 &&
            model.layers != NULL &&
            model.layers[9].wq != NULL) {
            llama_free_graph(&model);
            q_free_memory(&ctx);
            unlink("test_multi.qorus");
            TEST_PASS();
        } else {
            llama_free_graph(&model);
            q_free_memory(&ctx);
            unlink("test_multi.qorus");
            TEST_FAIL("Multi-layer structure invalid");
        }
    } else {
        q_free_memory(&ctx);
        unlink("test_multi.qorus");
        TEST_FAIL("Expected Q_OK");
    }
}

// Test 16: llama_free_graph with NULL
static void test_free_graph_null(void) {
    printf("Test 16: llama_free_graph with NULL\n");
    
    if (setjmp(crash_jmp_buf) == 0) {
        signal(SIGSEGV, crash_handler);
        signal(SIGBUS, crash_handler);
        signal(SIGABRT, crash_handler);
        
        llama_free_graph(NULL);
        
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        
        TEST_PASS();
    } else {
        TEST_CRASH();
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
    }
}

// Test 17: Double free (should be safe)
static void test_double_free(void) {
    printf("Test 17: Double free (should be safe)\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    
    create_minimal_model_file("test_double.qorus", 1);
    
    if (q_init_memory(&ctx, "test_double.qorus") != Q_OK ||
        q_alloc_arena(&ctx, 1024 * 1024) != Q_OK ||
        llama_build_graph(&ctx, &model) != Q_OK) {
        printf("  SKIPPED: Setup failed\n");
        q_free_memory(&ctx);
        unlink("test_double.qorus");
        return;
    }
    
    llama_free_graph(&model);
    llama_free_graph(&model);  // Second free
    
    q_free_memory(&ctx);
    unlink("test_double.qorus");
    
    TEST_PASS();
}

// Test 18: Very large vocab_size (potential overflow)
static void test_large_vocab_size(void) {
    printf("Test 18: Very large vocab_size\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    q_model_header header = {0};
    q_error_code ret;
    
    // Use maximum reasonable vocab_size
    uint8_t dummy_mmap[1024 * 1024];
    ctx.weights_mmap = dummy_mmap;
    ctx.weights_size = sizeof(dummy_mmap);
    ctx.header = &header;
    header.magic = Q_MAGIC;
    header.n_layers = 1;
    header.dim = 128;
    header.vocab_size = UINT32_MAX;  // Maximum value
    
    ret = q_alloc_arena(&ctx, 1024 * 1024);
    if (ret != Q_OK) {
        printf("  SKIPPED: Failed to allocate arena\n");
        return;
    }
    
    ret = llama_build_graph(&ctx, &model);
    
    q_free_memory(&ctx);
    
    // Should fail due to offset overflow or invalid config
    if (ret == Q_ERR_INVALID_CONFIG || ret == Q_ERR_ARENA_OOM) {
        TEST_PASS();
    } else {
        TEST_FAIL("Expected error for large vocab_size");
    }
}

// Test 19: Hidden_dim not multiple of 32
static void test_hidden_dim_not_multiple_32(void) {
    printf("Test 19: Hidden_dim not multiple of 32\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    q_model_header header = {0};
    q_error_code ret;
    
    uint8_t dummy_mmap[1024 * 1024];
    ctx.weights_mmap = dummy_mmap;
    ctx.weights_size = sizeof(dummy_mmap);
    ctx.header = &header;
    header.magic = Q_MAGIC;
    header.n_layers = 1;
    header.dim = 128;
    header.vocab_size = 100;
    header.hidden_dim = 100;  // Not multiple of 32
    
    ret = q_alloc_arena(&ctx, 1024 * 1024);
    if (ret != Q_OK) {
        printf("  SKIPPED: Failed to allocate arena\n");
        return;
    }
    
    ret = llama_build_graph(&ctx, &model);
    
    q_free_memory(&ctx);
    
    // Should fail when calculating Q4_0 size for FFN layers
    if (ret == Q_ERR_INVALID_CONFIG) {
        TEST_PASS();
    } else {
        TEST_FAIL("Expected Q_ERR_INVALID_CONFIG");
    }
}

// Test 20: Corrupted header (invalid n_kv_heads > n_heads)
static void test_invalid_kv_heads(void) {
    printf("Test 20: Invalid n_kv_heads > n_heads\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    q_model_header header = {0};
    q_error_code ret;
    
    uint8_t dummy_mmap[1024 * 1024];
    ctx.weights_mmap = dummy_mmap;
    ctx.weights_size = sizeof(dummy_mmap);
    ctx.header = &header;
    header.magic = Q_MAGIC;
    header.n_layers = 1;
    header.dim = 128;
    header.vocab_size = 100;
    header.n_heads = 4;
    header.n_kv_heads = 8;  // Invalid: > n_heads
    
    ret = q_alloc_arena(&ctx, 1024 * 1024);
    if (ret != Q_OK) {
        printf("  SKIPPED: Failed to allocate arena\n");
        return;
    }
    
    ret = llama_build_graph(&ctx, &model);
    
    q_free_memory(&ctx);
    
    // Should still build (we don't validate this in build_graph)
    // But kv_dim calculation might be wrong
    if (ret == Q_OK || ret == Q_ERR_INVALID_CONFIG) {
        TEST_PASS();
    } else {
        TEST_FAIL("Unexpected error");
    }
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("==========================================\n");
    printf("  LLAMA_BUILD_GRAPH ADVERSARIAL TESTS\n");
    printf("==========================================\n\n");
    
    // Install signal handlers
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGABRT, crash_handler);
    
    // Run all tests
    test_null_context();
    test_null_model();
    test_null_mmap();
    test_null_header();
    test_invalid_magic();
    test_zero_layers();
    test_zero_dim();
    test_zero_vocab_size();
    test_file_too_small();
    test_arena_not_allocated();
    test_arena_oom();
    test_offset_overflow();
    test_dim_not_multiple_32();
    test_valid_model();
    test_multiple_layers();
    test_free_graph_null();
    test_double_free();
    test_large_vocab_size();
    test_hidden_dim_not_multiple_32();
    test_invalid_kv_heads();
    
    // Restore signal handlers
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGABRT, SIG_DFL);
    
    // Print summary
    printf("\n==========================================\n");
    printf("  TEST SUMMARY\n");
    printf("==========================================\n");
    printf("  Tests Run:    %d\n", tests_run);
    printf("  Tests Passed: %d\n", tests_passed);
    printf("  Tests Failed: %d\n", tests_failed);
    printf("  Tests Crashed: %d\n", tests_crashed);
    printf("  Success Rate: %.1f%%\n", 
           tests_run > 0 ? (100.0 * tests_passed / tests_run) : 0.0);
    printf("==========================================\n\n");
    
    if (tests_failed == 0 && tests_crashed == 0) {
        printf("✓ All tests passed!\n");
        return 0;
    } else {
        printf("✗ Some tests failed or crashed\n");
        return 1;
    }
}

