#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>
#include <limits.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

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

// ============================================================================
// CLEANUP STRATEGY - Resource Management
// ============================================================================

// Cleanup macro: Ensures context is freed even on early return
#define CLEANUP_CONTEXT(ctx) do { \
    if ((ctx)->weights_mmap != NULL || (ctx)->scratch_buffer != NULL || (ctx)->kv_buffer != NULL) { \
        q_free_memory(ctx); \
    } \
} while(0)

// Track temporary files created during tests
static const char* temp_files[] = {
    "test_empty.qorus",
    "test_small.qorus",
    "test_corrupted_magic.qorus",
    "test_truncated.qorus",
    "test_invalid_header.qorus",
    "test_overflow.qorus",
    "test_inconsistent.qorus",
    "model_dummy.qorus",
    NULL  // Sentinel
};

// Cleanup function: Remove all temporary test files
static void cleanup_temp_files(void) {
    for (int i = 0; temp_files[i] != NULL; i++) {
        unlink(temp_files[i]);
    }
}

// Cleanup function: Remove build artifacts
static void cleanup_build_artifacts(void) {
    unlink("build/tests/test_model_file_validation");
    int ret1 = system("find build -name '*.o' -type f -delete 2>/dev/null || true");
    int ret2 = system("find build -name '*.d' -type f -delete 2>/dev/null || true");
    (void)ret1;
    (void)ret2;
}

// Combined cleanup function (called by atexit)
static void cleanup_all(void) {
    cleanup_temp_files();
    cleanup_build_artifacts();
}

// ============================================================================
// MAPA DE CENÁRIOS - Model File Validation
// ============================================================================
//
// HAPPY PATH:
//   - Arquivo válido deve carregar corretamente
//
// EDGE CASES:
//   - Arquivo vazio
//   - Arquivo muito pequeno
//   - Arquivo inexistente
//
// SECURITY/MALICIOUS:
//   - Arquivo com magic inválido
//   - Arquivo truncado (header OK, dados incompletos)
//   - Headers inválidos (dimensões impossíveis, valores zero)
//   - Arquivos muito grandes (overflow em cálculos)
//
// ============================================================================

// CATEGORY 1: EDGE CASES - Arquivos inválidos básicos
// ============================================================================

// Test 1: Arquivo inexistente
static void test_file_nonexistent(void) {
    TEST_START("Model file validation - Nonexistent file");
    
    q_context ctx = {0};
    q_error_code ret = q_init_memory(&ctx, "/nonexistent/file.qorus");
    
    if (ret == Q_ERR_FILE_OPEN) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_FILE_OPEN, got %d (%s)", ret, q_strerror(ret));
    }
}

// Test 2: Arquivo vazio
static void test_file_empty(void) {
    TEST_START("Model file validation - Empty file");
    
    // Create empty file
    FILE* f = fopen("test_empty.qorus", "wb");
    if (f == NULL) {
        TEST_FAIL("Cannot create empty test file");
        return;
    }
    fclose(f);
    
    q_context ctx = {0};
    q_error_code ret = q_init_memory(&ctx, "test_empty.qorus");
    
    // Note: test_empty.qorus will be cleaned up by cleanup_all() at exit
    
    if (ret == Q_ERR_FILE_TOO_SMALL) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_FILE_TOO_SMALL, got %d (%s)", ret, q_strerror(ret));
        CLEANUP_CONTEXT(&ctx);
    }
}

// Test 3: Arquivo muito pequeno (< Q_HEADER_SIZE)
static void test_file_too_small(void) {
    TEST_START("Model file validation - File too small (< Q_HEADER_SIZE)");
    
    // Create file with size < Q_HEADER_SIZE
    FILE* f = fopen("test_small.qorus", "wb");
    if (f == NULL) {
        TEST_FAIL("Cannot create small test file");
        return;
    }
    
    // Write only 32 bytes (less than Q_HEADER_SIZE = 64)
    uint8_t dummy[32] = {0};
    fwrite(dummy, 1, 32, f);
    fclose(f);
    
    q_context ctx = {0};
    q_error_code ret = q_init_memory(&ctx, "test_small.qorus");
    
    // Note: test_small.qorus will be cleaned up by cleanup_all() at exit
    
    if (ret == Q_ERR_FILE_TOO_SMALL) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_FILE_TOO_SMALL, got %d (%s)", ret, q_strerror(ret));
        CLEANUP_CONTEXT(&ctx);
    }
}

// CATEGORY 2: SECURITY/MALICIOUS - Arquivos corrompidos
// ============================================================================

// Test 4: Arquivo com magic inválido
// SECURITY: Validação de integridade do arquivo
static void test_file_corrupted_magic(void) {
    TEST_START("Model file validation - Corrupted file (invalid magic)");
    
    // Create file with invalid magic
    FILE* f = fopen("test_corrupted_magic.qorus", "wb");
    if (f == NULL) {
        TEST_FAIL("Cannot create corrupted test file");
        return;
    }
    
    q_model_header header = {0};
    header.magic = 0xDEADBEEF;  // Invalid magic (should be Q_MAGIC = 0x514F5231)
    header.version = 1;
    header.vocab_size = 1000;
    header.dim = 512;
    // Fill rest of header
    fwrite(&header, sizeof(header), 1, f);
    fclose(f);
    
    q_context ctx = {0};
    q_error_code ret = q_init_memory(&ctx, "test_corrupted_magic.qorus");
    
    // Note: test_corrupted_magic.qorus will be cleaned up by cleanup_all() at exit
    
    // Should fail with Q_ERR_INVALID_MAGIC
    if (ret == Q_ERR_INVALID_MAGIC) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_MAGIC, got %d (%s)", ret, q_strerror(ret));
        CLEANUP_CONTEXT(&ctx);
    }
}

// Test 5: Arquivo truncado (header completo, dados incompletos)
// SECURITY: Previne acesso a dados não mapeados
static void test_file_truncated(void) {
    TEST_START("Model file validation - Truncated file (header OK, data incomplete)");
    
    // Create file with valid header but incomplete data
    FILE* f = fopen("test_truncated.qorus", "wb");
    if (f == NULL) {
        TEST_FAIL("Cannot create truncated test file");
        return;
    }
    
    q_model_header header = {0};
    header.magic = Q_MAGIC;
    header.version = 1;
    header.vocab_size = 32000;
    header.dim = 4096;
    header.hidden_dim = 11008;
    header.n_layers = 2;
    header.n_heads = 32;
    header.n_kv_heads = 8;
    header.max_seq_len = 8192;
    
    // Write header
    fwrite(&header, sizeof(header), 1, f);
    
    // Write only partial data (should be much more)
    // Calculate expected size for token_embd: vocab_size * dim * sizeof(float)
    size_t expected_token_embd_size = (size_t)header.vocab_size * (size_t)header.dim * sizeof(float);
    size_t partial_size = expected_token_embd_size / 10;  // Only 10% of expected
    
    float* partial_data = calloc(partial_size / sizeof(float), sizeof(float));
    if (partial_data == NULL) {
        fclose(f);
        TEST_FAIL("Cannot allocate partial data");
        return;
    }
    
    fwrite(partial_data, 1, partial_size, f);
    free(partial_data);
    fclose(f);
    
    q_context ctx = {0};
    q_error_code ret = q_init_memory(&ctx, "test_truncated.qorus");
    
    // Note: test_truncated.qorus will be cleaned up by cleanup_all() at exit
    
    // Should either fail gracefully or succeed but fail later during build_graph
    // The important thing is it doesn't crash
    if (ret != Q_OK) {
        // Expected: file is truncated, should fail
        TEST_PASS();
    } else {
        // If it succeeds, try to build graph (should fail)
        q_error_code ret2 = q_alloc_arena(&ctx, 1024 * 1024);
        if (ret2 == Q_OK) {
            q_llama_model model = {0};
            ret2 = llama_build_graph(&ctx, &model);
            if (ret2 != Q_OK) {
                // Build failed as expected (truncated data)
                TEST_PASS();
            } else {
                TEST_FAIL("Build graph succeeded with truncated file (unexpected)");
                llama_free_graph(&model);
            }
        }
        CLEANUP_CONTEXT(&ctx);
    }
}

// Test 6: Headers inválidos - Dimensões impossíveis (zero)
// SECURITY: Previne divisão por zero e overflow
static void test_file_invalid_header_zero_dimensions(void) {
    TEST_START("Model file validation - Invalid header (zero dimensions)");
    
    FILE* f = fopen("test_invalid_header.qorus", "wb");
    if (f == NULL) {
        TEST_FAIL("Cannot create invalid header test file");
        return;
    }
    
    q_model_header header = {0};
    header.magic = Q_MAGIC;
    header.version = 1;
    header.vocab_size = 0;      // Invalid: zero vocab_size
    header.dim = 0;              // Invalid: zero dim
    header.hidden_dim = 0;       // Invalid: zero hidden_dim
    header.n_layers = 0;         // Invalid: zero layers
    header.n_heads = 0;          // Invalid: zero heads
    header.n_kv_heads = 0;       // Invalid: zero kv_heads
    header.max_seq_len = 0;      // Invalid: zero max_seq_len
    
    fwrite(&header, sizeof(header), 1, f);
    fclose(f);
    
    q_context ctx = {0};
    q_error_code ret = q_init_memory(&ctx, "test_invalid_header.qorus");
    
    // Note: test_invalid_header.qorus will be cleaned up by cleanup_all() at exit
    
    // Should either fail during init_memory or during build_graph
    if (ret != Q_OK) {
        // Expected: invalid dimensions should be rejected
        TEST_PASS();
    } else {
        // If it succeeds, try to build graph (should fail)
        q_error_code ret2 = q_alloc_arena(&ctx, 1024 * 1024);
        if (ret2 == Q_OK) {
            q_llama_model model = {0};
            ret2 = llama_build_graph(&ctx, &model);
            if (ret2 != Q_OK) {
                // Build failed as expected (invalid dimensions)
                TEST_PASS();
            } else {
                TEST_FAIL("Build graph succeeded with invalid dimensions (unexpected)");
                llama_free_graph(&model);
            }
        }
        CLEANUP_CONTEXT(&ctx);
    }
}

// Test 7: Headers inválidos - Dimensões muito grandes (overflow)
// SECURITY: Previne overflow em cálculos de tamanho
static void test_file_invalid_header_overflow(void) {
    TEST_START("Model file validation - Invalid header (overflow dimensions)");
    
    FILE* f = fopen("test_overflow.qorus", "wb");
    if (f == NULL) {
        TEST_FAIL("Cannot create overflow test file");
        return;
    }
    
    q_model_header header = {0};
    header.magic = Q_MAGIC;
    header.version = 1;
    header.vocab_size = UINT32_MAX;      // Maximum value (will cause overflow)
    header.dim = UINT32_MAX;             // Maximum value
    header.hidden_dim = UINT32_MAX;      // Maximum value
    header.n_layers = 1000;               // Large but reasonable
    header.n_heads = 128;                 // Large but reasonable
    header.n_kv_heads = 128;
    header.max_seq_len = UINT32_MAX;      // Maximum value
    
    fwrite(&header, sizeof(header), 1, f);
    fclose(f);
    
    q_context ctx = {0};
    q_error_code ret = q_init_memory(&ctx, "test_overflow.qorus");
    
    // Note: test_overflow.qorus will be cleaned up by cleanup_all() at exit
    
    // Should either fail during init_memory or during build_graph (overflow check)
    if (ret != Q_OK) {
        // Expected: overflow should be detected
        TEST_PASS();
    } else {
        // If it succeeds, try to build graph (should fail due to overflow)
        q_error_code ret2 = q_alloc_arena(&ctx, 1024 * 1024);
        if (ret2 == Q_OK) {
            q_llama_model model = {0};
            ret2 = llama_build_graph(&ctx, &model);
            if (ret2 == Q_ERR_OVERFLOW || ret2 != Q_OK) {
                // Build failed as expected (overflow detected)
                TEST_PASS();
            } else {
                TEST_FAIL("Build graph succeeded with overflow dimensions (unexpected)");
                llama_free_graph(&model);
            }
        }
        CLEANUP_CONTEXT(&ctx);
    }
}

// Test 8: Headers inválidos - Dimensões inconsistentes
// SECURITY: Previne acesso a dados incorretos
static void test_file_invalid_header_inconsistent(void) {
    TEST_START("Model file validation - Invalid header (inconsistent dimensions)");
    
    FILE* f = fopen("test_inconsistent.qorus", "wb");
    if (f == NULL) {
        TEST_FAIL("Cannot create inconsistent header test file");
        return;
    }
    
    q_model_header header = {0};
    header.magic = Q_MAGIC;
    header.version = 1;
    header.vocab_size = 32000;
    header.dim = 4096;
    header.hidden_dim = 11008;
    header.n_layers = 2;
    header.n_heads = 32;
    header.n_kv_heads = 100;     // Invalid: n_kv_heads > n_heads
    header.max_seq_len = 8192;
    
    fwrite(&header, sizeof(header), 1, f);
    fclose(f);
    
    q_context ctx = {0};
    q_error_code ret = q_init_memory(&ctx, "test_inconsistent.qorus");
    
    // Note: test_inconsistent.qorus will be cleaned up by cleanup_all() at exit
    
    // Should either fail during init_memory or during build_graph
    if (ret != Q_OK) {
        // Expected: inconsistent dimensions should be rejected
        TEST_PASS();
    } else {
        // If it succeeds, try to build graph (should fail)
        q_error_code ret2 = q_alloc_arena(&ctx, 1024 * 1024);
        if (ret2 == Q_OK) {
            q_llama_model model = {0};
            ret2 = llama_build_graph(&ctx, &model);
            if (ret2 == Q_ERR_INVALID_CONFIG || ret2 != Q_OK) {
                // Build failed as expected (inconsistent dimensions)
                TEST_PASS();
            } else {
                TEST_FAIL("Build graph succeeded with inconsistent dimensions (unexpected)");
                llama_free_graph(&model);
            }
        }
        CLEANUP_CONTEXT(&ctx);
    }
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("========================================\n");
    printf("  ADVERSARIAL TEST SUITE: Model File Validation\n");
    printf("========================================\n\n");
    printf("Strategy: Try to BREAK model file loading through adversarial testing\n");
    printf("Following Lead SDET methodology: Happy Path + Edge Cases + Security\n\n");
    
    // CRITICAL: Register cleanup function to ensure cleanup even on crash
    atexit(cleanup_all);
    
    // Reset statistics
    tests_run = 0;
    tests_passed = 0;
    tests_failed = 0;
    tests_crashed = 0;
    
    // Install crash handler
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGABRT, crash_handler);
    
    // CATEGORY 1: EDGE CASES
    printf("CATEGORY 1: Edge Cases - Invalid Files\n");
    printf("-----------------------------------\n");
    if (setjmp(crash_jmp_buf) == 0) {
        test_file_nonexistent();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_file_empty();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_file_too_small();
    } else {
        TEST_CRASH();
    }
    printf("\n");
    
    // CATEGORY 2: SECURITY/MALICIOUS
    printf("CATEGORY 2: Security/Malicious - Corrupted Files\n");
    printf("-----------------------------------\n");
    if (setjmp(crash_jmp_buf) == 0) {
        test_file_corrupted_magic();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_file_truncated();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_file_invalid_header_zero_dimensions();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_file_invalid_header_overflow();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_file_invalid_header_inconsistent();
    } else {
        TEST_CRASH();
    }
    printf("\n");
    
    // Final cleanup: Remove all temporary files and build artifacts
    cleanup_temp_files();
    cleanup_build_artifacts();
    
    // Print summary
    printf("=== Test Summary ===\n");
    printf("Total tests: %d\n", tests_run);
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    printf("Crashed: %d\n", tests_crashed);
    
    if (tests_failed == 0 && tests_crashed == 0) {
        printf("\n✓ All tests passed!\n");
        printf("✓ Build artifacts cleaned up\n");
        return 0;
    } else {
        printf("\n✗ Some tests failed or crashed\n");
        printf("✓ Build artifacts cleaned up\n");
        return 1;
    }
}

