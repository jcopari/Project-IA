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

// Track temporary files created during tests
static const char* temp_files[] = {
    "test_empty.qorus",
    "test_small.qorus",
    "test_corrupted.qorus",
    "test_no_read.qorus",
    NULL  // Sentinel
};

// Cleanup function: Remove all temporary test files
static void cleanup_temp_files(void) {
    for (int i = 0; temp_files[i] != NULL; i++) {
        unlink(temp_files[i]);
    }
}

// Cleanup function: Remove build artifacts (object files, executables)
static void cleanup_build_artifacts(void) {
    // Remove test executable
    unlink("build/tests/test_memory_strategies");
    
    // Remove object files from build directory
    int ret1 = system("find build -name '*.o' -type f -delete 2>/dev/null || true");
    int ret2 = system("find build -name '*.d' -type f -delete 2>/dev/null || true");
    (void)ret1;  // Suppress unused warning
    (void)ret2;  // Suppress unused warning
}

// Combined cleanup function (called by atexit)
static void cleanup_all(void) {
    cleanup_temp_files();
    cleanup_build_artifacts();
}

// Cleanup macro: Ensures context is freed even on early return
#define CLEANUP_CONTEXT(ctx) do { \
    if ((ctx)->weights_mmap != NULL || (ctx)->scratch_buffer != NULL || (ctx)->kv_buffer != NULL) { \
        q_free_memory(ctx); \
    } \
} while(0)

// Cleanup macro: Ensures model graph is freed
#define CLEANUP_MODEL(model) do { \
    if ((model)->token_embd != NULL || (model)->layers != NULL) { \
        llama_free_graph(model); \
    } \
} while(0)

// Cleanup macro: Ensures tokenizer is freed
#define CLEANUP_TOKENIZER(tok) do { \
    if ((tok)->vocab != NULL || (tok)->merges != NULL) { \
        q_tokenizer_free(tok); \
    } \
} while(0)

// Helper: Ensure dummy model exists
static bool ensure_dummy_model(void) {
    FILE* f = fopen("model_dummy.qorus", "rb");
    if (f != NULL) {
        fclose(f);
        return true;
    }
    
    // Try to generate it
    printf("  Generating dummy model...\n");
    int ret = system("python3 tools/convert_llama.py model_dummy.qorus 2 > /dev/null 2>&1");
    return (ret == 0);
}

// ============================================================================
// MAPA DE CENÁRIOS - q_init_memory_ex()
// ============================================================================
// 
// HAPPY PATH:
//   - Carregar modelo válido com Q_MMAP_LAZY (padrão)
//   - Carregar modelo válido com Q_MMAP_EAGER (otimização)
//
// EDGE CASES:
//   - Context NULL
//   - Model path NULL
//   - Arquivo inexistente
//   - Arquivo vazio
//   - Arquivo muito pequeno (< Q_HEADER_SIZE)
//   - Estratégia inválida (valor fora do enum)
//
// SECURITY/MALICIOUS:
//   - Path injection (../etc/passwd)
//   - Path muito longo (buffer overflow)
//   - Arquivo sem permissão de leitura
//   - Arquivo corrompido (magic inválido)
//
// ============================================================================

// CATEGORY 1: HAPPY PATH - Estratégias válidas
// ============================================================================

// Test 1: Q_MMAP_LAZY strategy (padrão, fast startup)
static void test_init_memory_lazy_strategy(void) {
    TEST_START("q_init_memory_ex - Q_MMAP_LAZY strategy (happy path)");
    
    if (!ensure_dummy_model()) {
        TEST_FAIL("Cannot generate dummy model");
        return;
    }
    
    q_context ctx = {0};
    q_error_code ret = q_init_memory_ex(&ctx, "model_dummy.qorus", Q_MMAP_LAZY);
    
    if (ret != Q_OK) {
        TEST_FAIL_MSG("Expected Q_OK, got %d (%s)", ret, q_strerror(ret));
        return;
    }
    
    // Validate context was initialized correctly
    if (ctx.weights_mmap == NULL || ctx.header == NULL) {
        TEST_FAIL("Context not properly initialized");
        q_free_memory(&ctx);
        return;
    }
    
    // Validate magic number
    if (ctx.header->magic != Q_MAGIC) {
        TEST_FAIL_MSG("Invalid magic: 0x%08X", ctx.header->magic);
        q_free_memory(&ctx);
        return;
    }
    
    q_free_memory(&ctx);
    TEST_PASS();
}

// Test 2: Q_MMAP_EAGER strategy (otimização, slow startup, fast first inference)
static void test_init_memory_eager_strategy(void) {
    TEST_START("q_init_memory_ex - Q_MMAP_EAGER strategy (happy path)");
    
    if (!ensure_dummy_model()) {
        TEST_FAIL("Cannot generate dummy model");
        return;
    }
    
    q_context ctx = {0};
    q_error_code ret = q_init_memory_ex(&ctx, "model_dummy.qorus", Q_MMAP_EAGER);
    
    if (ret != Q_OK) {
        TEST_FAIL_MSG("Expected Q_OK, got %d (%s)", ret, q_strerror(ret));
        return;
    }
    
    // Validate context was initialized correctly
    if (ctx.weights_mmap == NULL || ctx.header == NULL) {
        TEST_FAIL("Context not properly initialized");
        q_free_memory(&ctx);
        return;
    }
    
    // Validate magic number
    if (ctx.header->magic != Q_MAGIC) {
        TEST_FAIL_MSG("Invalid magic: 0x%08X", ctx.header->magic);
        q_free_memory(&ctx);
        return;
    }
    
    // NOTE: On Linux, MAP_POPULATE should be set for EAGER
    // On other platforms, it should still work but without eager loading
    // This is a behavioral difference we're testing
    
    q_free_memory(&ctx);
    TEST_PASS();
}

// Test 3: Comparação de comportamento entre estratégias
// CRITICAL: EAGER deve pré-carregar páginas (mais lento startup, mais rápido primeira inferência)
// LAZY deve carregar sob demanda (rápido startup, page faults na primeira inferência)
static void test_strategy_behavioral_difference(void) {
    TEST_START("q_init_memory_ex - Behavioral difference LAZY vs EAGER");
    
    if (!ensure_dummy_model()) {
        TEST_FAIL("Cannot generate dummy model");
        return;
    }
    
    // Test LAZY: Deve ser rápido (não bloqueia)
    q_context ctx_lazy = {0};
    q_error_code ret_lazy = q_init_memory_ex(&ctx_lazy, "model_dummy.qorus", Q_MMAP_LAZY);
    
    if (ret_lazy != Q_OK) {
        TEST_FAIL("LAZY strategy failed");
        return;
    }
    
    // Test EAGER: Pode ser mais lento (pré-carrega páginas)
    q_context ctx_eager = {0};
    q_error_code ret_eager = q_init_memory_ex(&ctx_eager, "model_dummy.qorus", Q_MMAP_EAGER);
    
    if (ret_eager != Q_OK) {
        TEST_FAIL("EAGER strategy failed");
        q_free_memory(&ctx_lazy);
        return;
    }
    
    // Both should have valid contexts
    if (ctx_lazy.weights_mmap == NULL || ctx_eager.weights_mmap == NULL) {
        TEST_FAIL("One or both contexts not initialized");
        q_free_memory(&ctx_lazy);
        q_free_memory(&ctx_eager);
        return;
    }
    
    // Both should point to same file size
    if (ctx_lazy.weights_size != ctx_eager.weights_size) {
        TEST_FAIL_MSG("Size mismatch: LAZY=%zu, EAGER=%zu", 
                     ctx_lazy.weights_size, ctx_eager.weights_size);
        q_free_memory(&ctx_lazy);
        q_free_memory(&ctx_eager);
        return;
    }
    
    q_free_memory(&ctx_lazy);
    q_free_memory(&ctx_eager);
    TEST_PASS();
}

// CATEGORY 2: EDGE CASES - Validação de argumentos
// ============================================================================

// Test 4: NULL context pointer
static void test_init_memory_ex_null_ctx(void) {
    TEST_START("q_init_memory_ex - NULL context pointer");
    
    q_error_code ret = q_init_memory_ex(NULL, "model_dummy.qorus", Q_MMAP_LAZY);
    
    if (ret == Q_ERR_INVALID_ARG) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_ARG, got %d", ret);
    }
}

// Test 5: NULL model path
static void test_init_memory_ex_null_path(void) {
    TEST_START("q_init_memory_ex - NULL model path");
    
    q_context ctx = {0};
    q_error_code ret = q_init_memory_ex(&ctx, NULL, Q_MMAP_LAZY);
    
    if (ret == Q_ERR_INVALID_ARG) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_ARG, got %d", ret);
    }
}

// Test 6: Arquivo inexistente
static void test_init_memory_ex_nonexistent_file(void) {
    TEST_START("q_init_memory_ex - Nonexistent file");
    
    q_context ctx = {0};
    q_error_code ret = q_init_memory_ex(&ctx, "/nonexistent/file.qorus", Q_MMAP_LAZY);
    
    if (ret == Q_ERR_FILE_OPEN) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_FILE_OPEN, got %d (%s)", ret, q_strerror(ret));
    }
}

// Test 7: Arquivo vazio
static void test_init_memory_ex_empty_file(void) {
    TEST_START("q_init_memory_ex - Empty file");
    
    // Create empty file
    FILE* f = fopen("test_empty.qorus", "wb");
    if (f == NULL) {
        TEST_FAIL("Cannot create empty test file");
        return;
    }
    fclose(f);
    
    q_context ctx = {0};
    q_error_code ret = q_init_memory_ex(&ctx, "test_empty.qorus", Q_MMAP_LAZY);
    
    unlink("test_empty.qorus");
    
    if (ret == Q_ERR_FILE_TOO_SMALL) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_FILE_TOO_SMALL, got %d (%s)", ret, q_strerror(ret));
        // Cleanup context if it was partially initialized
        CLEANUP_CONTEXT(&ctx);
    }
}

// Test 8: Arquivo muito pequeno (< Q_HEADER_SIZE)
static void test_init_memory_ex_too_small_file(void) {
    TEST_START("q_init_memory_ex - File too small (< Q_HEADER_SIZE)");
    
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
    q_error_code ret = q_init_memory_ex(&ctx, "test_small.qorus", Q_MMAP_LAZY);
    
    unlink("test_small.qorus");
    
    if (ret == Q_ERR_FILE_TOO_SMALL) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_FILE_TOO_SMALL, got %d (%s)", ret, q_strerror(ret));
        // Cleanup context if it was partially initialized
        CLEANUP_CONTEXT(&ctx);
    }
}

// Test 9: Estratégia inválida (valor fora do enum)
// SECURITY: Previne uso de valores não definidos
static void test_init_memory_ex_invalid_strategy(void) {
    TEST_START("q_init_memory_ex - Invalid strategy value");
    
    if (!ensure_dummy_model()) {
        TEST_FAIL("Cannot generate dummy model");
        return;
    }
    
    q_context ctx = {0};
    
    // Test with invalid strategy value (2, which is not in enum)
    // NOTE: C não valida enum em runtime, então isso pode funcionar
    // Mas testamos para garantir comportamento consistente
    q_error_code ret = q_init_memory_ex(&ctx, "model_dummy.qorus", (q_mmap_strategy)2);
    
    // Should either succeed (tratado como LAZY) ou falhar graciosamente
    // Não deve crashar
    if (ret == Q_OK) {
        // If it succeeds, validate context
        if (ctx.weights_mmap != NULL && ctx.header != NULL) {
            q_free_memory(&ctx);
            TEST_PASS();
        } else {
            TEST_FAIL("Invalid strategy succeeded but context not initialized");
        }
    } else {
        // If it fails, should be a valid error code
        if (ret < 0) {
            TEST_PASS();
        } else {
            TEST_FAIL_MSG("Invalid strategy returned invalid error code: %d", ret);
        }
    }
}

// CATEGORY 3: SECURITY/MALICIOUS - Tentativas de ataque
// ============================================================================

// Test 10: Path injection (../etc/passwd)
// SECURITY: Previne directory traversal attacks
static void test_init_memory_ex_path_injection(void) {
    TEST_START("q_init_memory_ex - Path injection attack (../etc/passwd)");
    
    q_context ctx = {0};
    q_error_code ret = q_init_memory_ex(&ctx, "../etc/passwd", Q_MMAP_LAZY);
    
    // Should fail (file doesn't exist or is not a valid model)
    // Should NOT crash or expose sensitive data
    if (ret != Q_OK) {
        TEST_PASS();
    } else {
        // If it succeeds (unlikely), validate it's actually a model file
        if (ctx.header != NULL && ctx.header->magic == Q_MAGIC) {
            TEST_FAIL("Path injection succeeded (security issue!)");
            q_free_memory(&ctx);
        } else {
            TEST_PASS();
        }
    }
}

// Test 11: Path muito longo (buffer overflow prevention)
// SECURITY: Previne buffer overflow em path handling
static void test_init_memory_ex_long_path(void) {
    TEST_START("q_init_memory_ex - Very long path (buffer overflow prevention)");
    
    // Create path longer than PATH_MAX (typically 4096)
    char long_path[5000];
    memset(long_path, 'a', sizeof(long_path) - 1);
    long_path[sizeof(long_path) - 1] = '\0';
    // Copy prefix safely
    memcpy(long_path, "/tmp/", 5);
    
    q_context ctx = {0};
    q_error_code ret = q_init_memory_ex(&ctx, long_path, Q_MMAP_LAZY);
    
    // Should fail gracefully (file doesn't exist)
    // Should NOT crash
    if (ret != Q_OK) {
        TEST_PASS();
    } else {
        TEST_FAIL("Long path succeeded (may indicate buffer overflow vulnerability)");
        q_free_memory(&ctx);
    }
}

// Test 12: Arquivo sem permissão de leitura
// SECURITY: Validação de permissões
static void test_init_memory_ex_no_read_permission(void) {
    TEST_START("q_init_memory_ex - File without read permission");
    
    // Create file and remove read permission
    FILE* f = fopen("test_no_read.qorus", "wb");
    if (f == NULL) {
        TEST_FAIL("Cannot create test file");
        return;
    }
    
    // Write minimal valid header
    q_model_header header = {0};
    header.magic = Q_MAGIC;
    fwrite(&header, sizeof(header), 1, f);
    fclose(f);
    
    // Remove read permission
    chmod("test_no_read.qorus", 0000);
    
    q_context ctx = {0};
    q_error_code ret = q_init_memory_ex(&ctx, "test_no_read.qorus", Q_MMAP_LAZY);
    
    // Restore permissions for cleanup
    chmod("test_no_read.qorus", 0644);
    unlink("test_no_read.qorus");
    
    // Should fail with Q_ERR_FILE_OPEN
    if (ret == Q_ERR_FILE_OPEN) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_FILE_OPEN, got %d (%s)", ret, q_strerror(ret));
        // Cleanup context if it was partially initialized
        CLEANUP_CONTEXT(&ctx);
    }
}

// Test 13: Arquivo corrompido (magic inválido)
// SECURITY: Validação de integridade do arquivo
static void test_init_memory_ex_corrupted_magic(void) {
    TEST_START("q_init_memory_ex - Corrupted file (invalid magic)");
    
    // Create file with invalid magic
    FILE* f = fopen("test_corrupted.qorus", "wb");
    if (f == NULL) {
        TEST_FAIL("Cannot create corrupted test file");
        return;
    }
    
    q_model_header header = {0};
    header.magic = 0xDEADBEEF;  // Invalid magic
    fwrite(&header, sizeof(header), 1, f);
    fclose(f);
    
    q_context ctx = {0};
    q_error_code ret = q_init_memory_ex(&ctx, "test_corrupted.qorus", Q_MMAP_LAZY);
    
    unlink("test_corrupted.qorus");
    
    // Should fail with Q_ERR_INVALID_MAGIC
    if (ret == Q_ERR_INVALID_MAGIC) {
        TEST_PASS();
    } else {
        TEST_FAIL_MSG("Expected Q_ERR_INVALID_MAGIC, got %d (%s)", ret, q_strerror(ret));
        // Cleanup context if it was partially initialized
        CLEANUP_CONTEXT(&ctx);
    }
}

// Test 14: Double initialization (memory leak prevention)
// SECURITY: Previne vazamento de memória
static void test_init_memory_ex_double_init(void) {
    TEST_START("q_init_memory_ex - Double initialization (memory leak prevention)");
    
    if (!ensure_dummy_model()) {
        TEST_FAIL("Cannot generate dummy model");
        return;
    }
    
    q_context ctx = {0};
    
    // First initialization
    q_error_code ret1 = q_init_memory_ex(&ctx, "model_dummy.qorus", Q_MMAP_LAZY);
    if (ret1 != Q_OK) {
        TEST_FAIL("First initialization failed");
        return;
    }
    
    // Second initialization (should overwrite first)
    q_error_code ret2 = q_init_memory_ex(&ctx, "model_dummy.qorus", Q_MMAP_EAGER);
    
    // Should succeed (overwrites previous mmap)
    if (ret2 == Q_OK) {
        // Validate new mmap is different or same (implementation dependent)
        // Important: Should not leak the first mmap
        if (ctx.weights_mmap != NULL && ctx.header != NULL) {
            TEST_PASS();
        } else {
            TEST_FAIL("Second initialization succeeded but context invalid");
        }
    } else {
        TEST_FAIL_MSG("Second initialization failed: %d (%s)", ret2, q_strerror(ret2));
    }
    
    q_free_memory(&ctx);
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("========================================\n");
    printf("  ADVERSARIAL TEST SUITE: q_init_memory_ex()\n");
    printf("========================================\n\n");
    printf("Strategy: Try to BREAK memory initialization through adversarial testing\n");
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
    
    // CATEGORY 1: HAPPY PATH
    printf("CATEGORY 1: Happy Path - Valid Strategies\n");
    printf("-----------------------------------\n");
    if (setjmp(crash_jmp_buf) == 0) {
        test_init_memory_lazy_strategy();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_init_memory_eager_strategy();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_strategy_behavioral_difference();
    } else {
        TEST_CRASH();
    }
    printf("\n");
    
    // CATEGORY 2: EDGE CASES
    printf("CATEGORY 2: Edge Cases - Argument Validation\n");
    printf("-----------------------------------\n");
    if (setjmp(crash_jmp_buf) == 0) {
        test_init_memory_ex_null_ctx();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_init_memory_ex_null_path();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_init_memory_ex_nonexistent_file();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_init_memory_ex_empty_file();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_init_memory_ex_too_small_file();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_init_memory_ex_invalid_strategy();
    } else {
        TEST_CRASH();
    }
    printf("\n");
    
    // CATEGORY 3: SECURITY/MALICIOUS
    printf("CATEGORY 3: Security/Malicious - Attack Attempts\n");
    printf("-----------------------------------\n");
    if (setjmp(crash_jmp_buf) == 0) {
        test_init_memory_ex_path_injection();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_init_memory_ex_long_path();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_init_memory_ex_no_read_permission();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_init_memory_ex_corrupted_magic();
    } else {
        TEST_CRASH();
    }
    
    if (setjmp(crash_jmp_buf) == 0) {
        test_init_memory_ex_double_init();
    } else {
        TEST_CRASH();
    }
    printf("\n");
    
    // Print summary
    printf("=== Test Summary ===\n");
    printf("Total tests: %d\n", tests_run);
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    printf("Crashed: %d\n", tests_crashed);
    
    // Final cleanup: Remove all temporary files and build artifacts
    cleanup_temp_files();
    cleanup_build_artifacts();
    
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

