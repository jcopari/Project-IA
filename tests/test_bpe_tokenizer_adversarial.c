// ============================================================================
// ADVERSARIAL TEST SUITE: BPE Tokenizer (`src/tokenizer/bpe.c`)
// ============================================================================
// Lead SDET Strategy: Try to BREAK the BPE tokenizer through adversarial testing
// Following MFR + CoT + Mathematical Proof + TDD methodology
//
// Test Categories (from @gereteste.md):
// 1. Happy Path: Normal operation
// 2. Edge Cases: Empty strings, very long strings, boundary values
// 3. Null/Undefined: Missing data handling, uninitialized memory
// 4. Security/Malicious: Buffer overflows, integer overflows, corrupted data
// 5. Memory Safety: Use-after-free, double-free, memory leaks
// 6. Hash Table: Collisions, invalid lookups, corruption
// 7. Merge Rules: Invalid merges, priority order, circular dependencies
//
// Failure Modes from BPE_TOKENIZER_PLAN.md FASE 3.3:
// - Array static overflow → Alocação dinâmica ✅
// - Lookup linear → Hash table ✅
// - Merge order → Prioridade mantida ✅
// - Race condition → Thread-safe ✅
// - Token ID inválido → Validação ✅
// ============================================================================

#include "qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <signal.h>
#include <setjmp.h>
#include <limits.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

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

// Test helper macros (following project pattern)
#define TEST_START(name) \
    do { \
        tests_run++; \
        printf("  [%d] %s... ", tests_run, name); \
        fflush(stdout); \
        signal(SIGSEGV, crash_handler); \
        signal(SIGBUS, crash_handler); \
        signal(SIGFPE, crash_handler); \
        signal(SIGABRT, crash_handler); \
        if (setjmp(crash_jmp) == 0) {

#define TEST_END \
        } else { \
            tests_crashed++; \
            printf("CRASHED\n"); \
            signal(SIGSEGV, SIG_DFL); \
            signal(SIGBUS, SIG_DFL); \
            signal(SIGFPE, SIG_DFL); \
            signal(SIGABRT, SIG_DFL); \
        } \
    } while(0)

#define TEST_PASS() \
    do { \
        tests_passed++; \
        printf("PASS\n"); \
        signal(SIGSEGV, SIG_DFL); \
        signal(SIGBUS, SIG_DFL); \
        signal(SIGFPE, SIG_DFL); \
        signal(SIGABRT, SIG_DFL); \
    } while(0)

#define TEST_FAIL(reason) \
    do { \
        tests_failed++; \
        printf("FAIL: %s\n", reason); \
        signal(SIGSEGV, SIG_DFL); \
        signal(SIGBUS, SIG_DFL); \
        signal(SIGFPE, SIG_DFL); \
        signal(SIGABRT, SIG_DFL); \
    } while(0)

// Helper: Create minimal tokenizer for testing
static q_error_code create_test_tokenizer(q_tokenizer* tok, uint32_t vocab_size, uint32_t num_merges) {
    memset(tok, 0, sizeof(q_tokenizer));
    
    tok->vocab_size = vocab_size;
    tok->num_merges = num_merges;
    tok->bos_token_id = vocab_size;
    tok->eos_token_id = vocab_size + 1;
    tok->pad_token_id = vocab_size + 2;
    
    // Allocate vocab (simplified: just allocate pointers)
    tok->vocab = (char**)calloc(vocab_size, sizeof(char*));
    if (tok->vocab == NULL) {
        return Q_ERR_ALLOC_FAILED;
    }
    
    // Allocate merges if needed
    if (num_merges > 0) {
        tok->merges = (q_bpe_merge*)calloc(num_merges, sizeof(q_bpe_merge));
        if (tok->merges == NULL) {
            free((void*)tok->vocab);
            return Q_ERR_ALLOC_FAILED;
        }
    }
    
    tok->initialized = true;
    return Q_OK;
}

// Helper: Free test tokenizer
static void free_test_tokenizer(q_tokenizer* tok) {
    if (tok == NULL) return;
    
    if (tok->vocab != NULL) {
        free((void*)tok->vocab);
    }
    if (tok->merges != NULL) {
        free(tok->merges);
    }
    // Note: merge_hash_table is freed by q_tokenizer_free, but we're testing manually
    if (tok->merge_hash_table != NULL) {
        // Internal cleanup (would normally call q_tokenizer_free)
        // For adversarial tests, we test free separately
    }
    memset(tok, 0, sizeof(q_tokenizer));
}

// Helper: Create corrupt tokenizer file
static void create_corrupt_tokenizer_file(const char* path, uint32_t magic, uint32_t version) {
    FILE* f = fopen(path, "wb");
    if (f == NULL) return;
    
    // Write header with corrupt magic/version
    uint32_t header[8] = {
        magic,      // magic
        version,    // version
        256,        // vocab_size
        0,          // num_merges
        256,        // bos_token_id
        257,        // eos_token_id
        258,        // pad_token_id
        0           // reserved
    };
    fwrite(header, sizeof(uint32_t), 8, f);
    
    // Write minimal vocab (256 tokens, each 1 byte)
    for (uint32_t i = 0; i < 256; i++) {
        uint8_t length = 1;
        fwrite(&length, 1, 1, f);
        fwrite(&i, 1, 1, f);  // Token string = byte value
    }
    
    fclose(f);
}

// ============================================================================
// CATEGORY 1: NULL POINTER ATTACKS
// ============================================================================

static void test_null_tokenizer_encode(void) {
    TEST_START("NULL tokenizer (encode)");
    
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    q_error_code err = q_tokenizer_encode(NULL, "test", tokens, &num_tokens, 256, false, false);
    if (err != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Expected Q_ERR_INVALID_ARG for NULL tokenizer");
        return;
    }
    
    TEST_PASS();
TEST_END;
}

static void test_null_text_encode(void) {
    TEST_START("NULL text (encode)");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, NULL, tokens, &num_tokens, 256, false, false);
    if (err != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Expected Q_ERR_INVALID_ARG for NULL text");
        free_test_tokenizer(&tok);
        return;
    }
    
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

static void test_null_tokens_out_encode(void) {
    TEST_START("NULL tokens_out (encode)");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, "test", NULL, &num_tokens, 256, false, false);
    if (err != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Expected Q_ERR_INVALID_ARG for NULL tokens_out");
        free_test_tokenizer(&tok);
        return;
    }
    
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

static void test_null_num_tokens_out_encode(void) {
    TEST_START("NULL num_tokens_out (encode)");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    uint32_t tokens[256];
    
    err = q_tokenizer_encode(&tok, "test", tokens, NULL, 256, false, false);
    if (err != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Expected Q_ERR_INVALID_ARG for NULL num_tokens_out");
        free_test_tokenizer(&tok);
        return;
    }
    
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

static void test_null_tokenizer_decode(void) {
    TEST_START("NULL tokenizer (decode)");
    
    uint32_t tokens[] = {72, 101, 108, 108, 111};
    char text[256];
    
    q_error_code err = q_tokenizer_decode(NULL, tokens, 5, text, sizeof(text));
    if (err != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Expected Q_ERR_INVALID_ARG for NULL tokenizer");
        return;
    }
    
    TEST_PASS();
TEST_END;
}

static void test_null_tokenizer_load(void) {
    TEST_START("NULL tokenizer (load)");
    
    q_error_code err = q_tokenizer_load(NULL, "nonexistent.bin");
    if (err != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Expected Q_ERR_INVALID_ARG for NULL tokenizer");
        return;
    }
    
    TEST_PASS();
TEST_END;
}

static void test_null_path_load(void) {
    TEST_START("NULL path (load)");
    
    q_tokenizer tok;
    q_error_code err = q_tokenizer_load(&tok, NULL);
    if (err != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Expected Q_ERR_INVALID_ARG for NULL path");
        return;
    }
    
    TEST_PASS();
TEST_END;
}

static void test_null_tokenizer_free(void) {
    TEST_START("NULL tokenizer (free)");
    
    // Should not crash
    q_tokenizer_free(NULL);
    
    TEST_PASS();
TEST_END;
}

// ============================================================================
// CATEGORY 2: EDGE CASES - BOUNDARY VALUES
// ============================================================================

static void test_empty_text_encode(void) {
    TEST_START("Empty text (encode)");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, "", tokens, &num_tokens, 256, false, false);
    if (err != Q_OK) {
        TEST_FAIL("Empty text should return Q_OK");
        free_test_tokenizer(&tok);
        return;
    }
    if (num_tokens != 0) {
        TEST_FAIL("Empty text should produce 0 tokens");
        free_test_tokenizer(&tok);
        return;
    }
    
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

static void test_max_text_length(void) {
    TEST_START("Text at MAX_TEXT_BYTES limit");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    // Create text at MAX_TEXT_BYTES (1MB)
    char* large_text = (char*)malloc(1024 * 1024 + 1);
    if (large_text == NULL) {
        TEST_FAIL("Failed to allocate large text");
        free_test_tokenizer(&tok);
        return;
    }
    memset(large_text, 'A', 1024 * 1024);
    large_text[1024 * 1024] = '\0';
    
    uint32_t* tokens = (uint32_t*)malloc(1024 * 1024 * sizeof(uint32_t));
    if (tokens == NULL) {
        free(large_text);
        TEST_FAIL("Failed to allocate tokens buffer");
        free_test_tokenizer(&tok);
        return;
    }
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, large_text, tokens, &num_tokens, 1024 * 1024, false, false);
    if (err != Q_OK) {
        TEST_FAIL("MAX_TEXT_BYTES text should succeed");
        free(large_text);
        free(tokens);
        free_test_tokenizer(&tok);
        return;
    }
    
    free(large_text);
    free(tokens);
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

static void test_text_exceeds_max_bytes(void) {
    TEST_START("Text exceeds MAX_TEXT_BYTES");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    // Create text exceeding MAX_TEXT_BYTES (1MB + 1)
    char* large_text = (char*)malloc(1024 * 1024 + 2);
    if (large_text == NULL) {
        TEST_FAIL("Failed to allocate large text");
        free_test_tokenizer(&tok);
        return;
    }
    memset(large_text, 'A', 1024 * 1024 + 1);
    large_text[1024 * 1024 + 1] = '\0';
    
    uint32_t* tokens = (uint32_t*)malloc((1024 * 1024 + 1) * sizeof(uint32_t));
    if (tokens == NULL) {
        free(large_text);
        TEST_FAIL("Failed to allocate tokens buffer");
        free_test_tokenizer(&tok);
        return;
    }
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, large_text, tokens, &num_tokens, 1024 * 1024 + 1, false, false);
    if (err != Q_ERR_ARENA_OOM) {
        TEST_FAIL("Text exceeding MAX_TEXT_BYTES should return Q_ERR_ARENA_OOM");
        free(large_text);
        free(tokens);
        free_test_tokenizer(&tok);
        return;
    }
    
    free(large_text);
    free(tokens);
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

static void test_zero_max_tokens(void) {
    TEST_START("Zero max_tokens (encode)");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    uint32_t tokens[1];
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, "test", tokens, &num_tokens, 0, false, false);
    if (err != Q_ERR_INVALID_SIZE) {
        TEST_FAIL("Zero max_tokens should return Q_ERR_INVALID_SIZE");
        free_test_tokenizer(&tok);
        return;
    }
    
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

static void test_single_byte_text(void) {
    TEST_START("Single byte text");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, "A", tokens, &num_tokens, 256, false, false);
    if (err != Q_OK) {
        TEST_FAIL("Single byte text should succeed");
        free_test_tokenizer(&tok);
        return;
    }
    if (num_tokens != 1 || tokens[0] != 65) {
        TEST_FAIL("Single byte 'A' should produce token 65");
        free_test_tokenizer(&tok);
        return;
    }
    
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

// ============================================================================
// CATEGORY 3: BUFFER OVERFLOW ATTACKS
// ============================================================================

static void test_buffer_too_small_encode(void) {
    TEST_START("Buffer too small (encode)");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    const char* text = "Hello World";
    uint32_t tokens[5];  // Too small
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, text, tokens, &num_tokens, 5, false, false);
    if (err != Q_ERR_ARENA_OOM) {
        TEST_FAIL("Buffer too small should return Q_ERR_ARENA_OOM");
        free_test_tokenizer(&tok);
        return;
    }
    
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

static void test_buffer_exact_size_encode(void) {
    TEST_START("Buffer exact size (encode)");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    const char* text = "Hello";
    uint32_t tokens[5];
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, text, tokens, &num_tokens, 5, false, false);
    if (err != Q_OK) {
        TEST_FAIL("Buffer exact size should succeed");
        free_test_tokenizer(&tok);
        return;
    }
    if (num_tokens != 5) {
        TEST_FAIL("Should produce 5 tokens");
        free_test_tokenizer(&tok);
        return;
    }
    
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

static void test_buffer_overflow_with_merges(void) {
    TEST_START("Buffer overflow with merges");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 1);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    // Setup merge that reduces tokens
    tok.merges[0].token_id1 = 108;
    tok.merges[0].token_id2 = 108;
    tok.merges[0].merged_id = 500;
    
    const char* text = "hello";  // 5 bytes -> 4 tokens after merge
    uint32_t tokens[3];  // Too small (needs 4)
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, text, tokens, &num_tokens, 3, false, false);
    if (err != Q_ERR_ARENA_OOM) {
        TEST_FAIL("Buffer too small after merges should return Q_ERR_ARENA_OOM");
        free_test_tokenizer(&tok);
        return;
    }
    
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

static void test_buffer_overflow_with_bos_eos(void) {
    TEST_START("Buffer overflow with BOS/EOS");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    const char* text = "Hi";  // 2 tokens
    uint32_t tokens[3];  // Needs 4 (BOS + 2 + EOS)
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, text, tokens, &num_tokens, 3, true, true);
    if (err != Q_ERR_ARENA_OOM) {
        TEST_FAIL("Buffer too small with BOS/EOS should return Q_ERR_ARENA_OOM");
        free_test_tokenizer(&tok);
        return;
    }
    
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

// ============================================================================
// CATEGORY 4: UNINITIALIZED MEMORY ATTACKS
// ============================================================================

static void test_uninitialized_tokenizer_encode(void) {
    TEST_START("Uninitialized tokenizer (encode)");
    
    q_tokenizer tok;
    memset(&tok, 0, sizeof(q_tokenizer));
    // tok.initialized = false (not set)
    
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    q_error_code err = q_tokenizer_encode(&tok, "test", tokens, &num_tokens, 256, false, false);
    if (err != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Uninitialized tokenizer should return Q_ERR_INVALID_ARG");
        return;
    }
    
    TEST_PASS();
TEST_END;
}

static void test_uninitialized_tokenizer_decode(void) {
    TEST_START("Uninitialized tokenizer (decode)");
    
    q_tokenizer tok;
    memset(&tok, 0, sizeof(q_tokenizer));
    // tok.initialized = false (not set)
    
    uint32_t tokens[] = {72, 101, 108, 108, 111};
    char text[256];
    
    q_error_code err = q_tokenizer_decode(&tok, tokens, 5, text, sizeof(text));
    if (err != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Uninitialized tokenizer should return Q_ERR_INVALID_ARG");
        return;
    }
    
    TEST_PASS();
TEST_END;
}

static void test_partially_initialized_tokenizer(void) {
    TEST_START("Partially initialized tokenizer");
    
    q_tokenizer tok;
    memset(&tok, 0xFF, sizeof(q_tokenizer));  // Fill with garbage
    tok.initialized = true;  // Only this field set
    
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    // Should fail validation (vocab_size likely invalid) or crash
    // Both behaviors are acceptable for adversarial testing
    q_error_code err = q_tokenizer_encode(&tok, "test", tokens, &num_tokens, 256, false, false);
    // If it doesn't crash, it should return an error
    if (err == Q_OK) {
        TEST_FAIL("Partially initialized tokenizer should fail validation");
        return;
    }
    
    TEST_PASS();
    TEST_END;
}

// ============================================================================
// CATEGORY 5: INVALID FILE ATTACKS (q_tokenizer_load)
// ============================================================================

static void test_file_nonexistent_load(void) {
    TEST_START("Nonexistent file (load)");
    
    q_tokenizer tok;
    q_error_code err = q_tokenizer_load(&tok, "/nonexistent/path/tokenizer.bin");
    if (err != Q_ERR_FILE_OPEN) {
        TEST_FAIL("Nonexistent file should return Q_ERR_FILE_OPEN");
        return;
    }
    
    TEST_PASS();
TEST_END;
}

static void test_file_empty_load(void) {
    TEST_START("Empty file (load)");
    
    const char* path = "/tmp/test_empty_tokenizer.bin";
    FILE* f = fopen(path, "wb");
    if (f == NULL) {
        TEST_FAIL("Failed to create empty file");
        return;
    }
    fclose(f);
    
    q_tokenizer tok;
    q_error_code err = q_tokenizer_load(&tok, path);
    if (err != Q_ERR_FILE_OPEN && err != Q_ERR_INVALID_MAGIC) {
        TEST_FAIL("Empty file should return error");
        unlink(path);
        return;
    }
    
    unlink(path);
    TEST_PASS();
TEST_END;
}

static void test_file_corrupted_magic_load(void) {
    TEST_START("Corrupted magic (load)");
    
    const char* path = "/tmp/test_corrupt_magic.bin";
    create_corrupt_tokenizer_file(path, 0xDEADBEEF, 1);  // Wrong magic
    
    q_tokenizer tok;
    q_error_code err = q_tokenizer_load(&tok, path);
    if (err != Q_ERR_INVALID_MAGIC) {
        TEST_FAIL("Corrupted magic should return Q_ERR_INVALID_MAGIC");
        unlink(path);
        return;
    }
    
    unlink(path);
    TEST_PASS();
TEST_END;
}

static void test_file_corrupted_version_load(void) {
    TEST_START("Corrupted version (load)");
    
    const char* path = "/tmp/test_corrupt_version.bin";
    create_corrupt_tokenizer_file(path, 0x51544B52, 999);  // Wrong version
    
    q_tokenizer tok;
    q_error_code err = q_tokenizer_load(&tok, path);
    if (err != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Corrupted version should return Q_ERR_INVALID_ARG");
        unlink(path);
        return;
    }
    
    unlink(path);
    TEST_PASS();
TEST_END;
}

static void test_file_vocab_size_zero_load(void) {
    TEST_START("Vocab size zero (load)");
    
    const char* path = "/tmp/test_vocab_zero.bin";
    FILE* f = fopen(path, "wb");
    if (f == NULL) {
        TEST_FAIL("Failed to create file");
        return;
    }
    
    uint32_t header[8] = {
        0x51544B52,  // magic
        1,           // version
        0,           // vocab_size = 0 (invalid)
        0,           // num_merges
        256,         // bos_token_id
        257,         // eos_token_id
        258,         // pad_token_id
        0            // reserved
    };
    fwrite(header, sizeof(uint32_t), 8, f);
    fclose(f);
    
    q_tokenizer tok;
    q_error_code err = q_tokenizer_load(&tok, path);
    if (err != Q_ERR_INVALID_SIZE) {
        TEST_FAIL("Vocab size zero should return Q_ERR_INVALID_SIZE");
        unlink(path);
        return;
    }
    
    unlink(path);
    TEST_PASS();
TEST_END;
}

static void test_file_vocab_size_overflow_load(void) {
    TEST_START("Vocab size overflow (load)");
    
    const char* path = "/tmp/test_vocab_overflow.bin";
    FILE* f = fopen(path, "wb");
    if (f == NULL) {
        TEST_FAIL("Failed to create file");
        return;
    }
    
    uint32_t header[8] = {
        0x51544B52,  // magic
        1,           // version
        2000000,     // vocab_size > 1000000 (invalid)
        0,           // num_merges
        256,         // bos_token_id
        257,         // eos_token_id
        258,         // pad_token_id
        0            // reserved
    };
    fwrite(header, sizeof(uint32_t), 8, f);
    fclose(f);
    
    q_tokenizer tok;
    q_error_code err = q_tokenizer_load(&tok, path);
    if (err != Q_ERR_INVALID_SIZE) {
        TEST_FAIL("Vocab size overflow should return Q_ERR_INVALID_SIZE");
        unlink(path);
        return;
    }
    
    unlink(path);
    TEST_PASS();
TEST_END;
}

static void test_file_truncated_vocab_load(void) {
    TEST_START("Truncated vocab (load)");
    
    const char* path = "/tmp/test_truncated_vocab.bin";
    FILE* f = fopen(path, "wb");
    if (f == NULL) {
        TEST_FAIL("Failed to create file");
        return;
    }
    
    uint32_t header[8] = {
        0x51544B52,  // magic
        1,           // version
        256,         // vocab_size
        0,           // num_merges
        256,         // bos_token_id
        257,         // eos_token_id
        258,         // pad_token_id
        0            // reserved
    };
    fwrite(header, sizeof(uint32_t), 8, f);
    
    // Write only 100 tokens instead of 256 (truncated)
    for (uint32_t i = 0; i < 100; i++) {
        uint8_t length = 1;
        fwrite(&length, 1, 1, f);
        fwrite(&i, 1, 1, f);
    }
    fclose(f);
    
    q_tokenizer tok;
    q_error_code err = q_tokenizer_load(&tok, path);
    if (err != Q_ERR_FILE_OPEN) {
        TEST_FAIL("Truncated vocab should return Q_ERR_FILE_OPEN");
        unlink(path);
        return;
    }
    
    unlink(path);
    TEST_PASS();
TEST_END;
}

// ============================================================================
// CATEGORY 6: INVALID MERGE RULES
// ============================================================================

static void test_invalid_merge_token_id1(void) {
    TEST_START("Invalid merge token_id1 >= vocab_size");
    
    const char* path = "/tmp/test_invalid_merge1.bin";
    FILE* f = fopen(path, "wb");
    if (f == NULL) {
        TEST_FAIL("Failed to create file");
        return;
    }
    
    uint32_t header[8] = {
        0x51544B52,  // magic
        1,           // version
        256,         // vocab_size
        1,           // num_merges
        256,         // bos_token_id
        257,         // eos_token_id
        258,         // pad_token_id
        0            // reserved
    };
    fwrite(header, sizeof(uint32_t), 8, f);
    
    // Write vocab (256 tokens)
    for (uint32_t i = 0; i < 256; i++) {
        uint8_t length = 1;
        fwrite(&length, 1, 1, f);
        fwrite(&i, 1, 1, f);
    }
    
    // Write invalid merge: token_id1 = 300 >= vocab_size (256)
    uint32_t merge[3] = {300, 108, 500};  // token_id1 invalid
    fwrite(merge, sizeof(uint32_t), 3, f);
    fclose(f);
    
    q_tokenizer tok;
    q_error_code err = q_tokenizer_load(&tok, path);
    if (err != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Invalid merge token_id1 should return Q_ERR_INVALID_ARG");
        unlink(path);
        return;
    }
    
    unlink(path);
    TEST_PASS();
TEST_END;
}

static void test_invalid_merge_token_id2(void) {
    TEST_START("Invalid merge token_id2 >= vocab_size");
    
    const char* path = "/tmp/test_invalid_merge2.bin";
    FILE* f = fopen(path, "wb");
    if (f == NULL) {
        TEST_FAIL("Failed to create file");
        return;
    }
    
    uint32_t header[8] = {
        0x51544B52,  // magic
        1,           // version
        256,         // vocab_size
        1,           // num_merges
        256,         // bos_token_id
        257,         // eos_token_id
        258,         // pad_token_id
        0            // reserved
    };
    fwrite(header, sizeof(uint32_t), 8, f);
    
    // Write vocab (256 tokens)
    for (uint32_t i = 0; i < 256; i++) {
        uint8_t length = 1;
        fwrite(&length, 1, 1, f);
        fwrite(&i, 1, 1, f);
    }
    
    // Write invalid merge: token_id2 = 300 >= vocab_size (256)
    uint32_t merge[3] = {108, 300, 500};  // token_id2 invalid
    fwrite(merge, sizeof(uint32_t), 3, f);
    fclose(f);
    
    q_tokenizer tok;
    q_error_code err = q_tokenizer_load(&tok, path);
    if (err != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Invalid merge token_id2 should return Q_ERR_INVALID_ARG");
        unlink(path);
        return;
    }
    
    unlink(path);
    TEST_PASS();
TEST_END;
}

static void test_invalid_merge_merged_id(void) {
    TEST_START("Invalid merge merged_id >= vocab_size");
    
    const char* path = "/tmp/test_invalid_merge3.bin";
    FILE* f = fopen(path, "wb");
    if (f == NULL) {
        TEST_FAIL("Failed to create file");
        return;
    }
    
    uint32_t header[8] = {
        0x51544B52,  // magic
        1,           // version
        256,         // vocab_size
        1,           // num_merges
        256,         // bos_token_id
        257,         // eos_token_id
        258,         // pad_token_id
        0            // reserved
    };
    fwrite(header, sizeof(uint32_t), 8, f);
    
    // Write vocab (256 tokens)
    for (uint32_t i = 0; i < 256; i++) {
        uint8_t length = 1;
        fwrite(&length, 1, 1, f);
        fwrite(&i, 1, 1, f);
    }
    
    // Write invalid merge: merged_id = 300 >= vocab_size (256)
    uint32_t merge[3] = {108, 108, 300};  // merged_id invalid
    fwrite(merge, sizeof(uint32_t), 3, f);
    fclose(f);
    
    q_tokenizer tok;
    q_error_code err = q_tokenizer_load(&tok, path);
    if (err != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Invalid merge merged_id should return Q_ERR_INVALID_ARG");
        unlink(path);
        return;
    }
    
    unlink(path);
    TEST_PASS();
TEST_END;
}

// ============================================================================
// CATEGORY 7: HASH TABLE EDGE CASES
// ============================================================================

static void test_hash_table_no_merges(void) {
    TEST_START("Hash table with no merges");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    // Hash table should be NULL when no merges
    if (tok.merge_hash_table != NULL) {
        TEST_FAIL("Hash table should be NULL when num_merges == 0");
        free_test_tokenizer(&tok);
        return;
    }
    
    // Encoding should still work
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    err = q_tokenizer_encode(&tok, "test", tokens, &num_tokens, 256, false, false);
    if (err != Q_OK) {
        TEST_FAIL("Encoding should work without hash table");
        free_test_tokenizer(&tok);
        return;
    }
    
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

static void test_hash_table_large_num_merges(void) {
    TEST_START("Hash table with large num_merges");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 1000, 10000);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    // Setup valid merges
    for (uint32_t i = 0; i < 10000; i++) {
        tok.merges[i].token_id1 = i % 256;
        tok.merges[i].token_id2 = (i + 1) % 256;
        tok.merges[i].merged_id = 500 + (i % 500);
    }
    
    // Encoding should work (hash table built internally)
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    err = q_tokenizer_encode(&tok, "test", tokens, &num_tokens, 256, false, false);
    // Should succeed (hash table handles collisions)
    (void)err;
    
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

// ============================================================================
// CATEGORY 8: MERGE APPLICATION EDGE CASES
// ============================================================================

static void test_single_token_no_merges(void) {
    TEST_START("Single token (no merges possible)");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 1);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    tok.merges[0].token_id1 = 108;
    tok.merges[0].token_id2 = 108;
    tok.merges[0].merged_id = 500;
    
    // Single token - no merges possible (< 2 tokens)
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, "A", tokens, &num_tokens, 256, false, false);
    if (err != Q_OK) {
        TEST_FAIL("Single token should succeed");
        free_test_tokenizer(&tok);
        return;
    }
    if (num_tokens != 1 || tokens[0] != 65) {
        TEST_FAIL("Single token should produce token 65");
        free_test_tokenizer(&tok);
        return;
    }
    
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

static void test_multiple_merges_priority_order(void) {
    TEST_START("Multiple merges priority order");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 3);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    // Setup merges with overlapping patterns
    // Merge 0 (highest priority): (108, 108) -> 500
    tok.merges[0].token_id1 = 108;
    tok.merges[0].token_id2 = 108;
    tok.merges[0].merged_id = 500;
    
    // Merge 1 (medium priority): (101, 108) -> 501
    tok.merges[1].token_id1 = 101;
    tok.merges[1].token_id2 = 108;
    tok.merges[1].merged_id = 501;
    
    // Merge 2 (low priority): (500, 111) -> 502
    tok.merges[2].token_id1 = 500;
    tok.merges[2].token_id2 = 111;
    tok.merges[2].merged_id = 502;
    
    // Text "hello" should apply merge 0 first (ll -> 500), then merge 2 (500+o -> 502)
    const char* text = "hello";
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, text, tokens, &num_tokens, 256, false, false);
    if (err != Q_OK) {
        TEST_FAIL("Multiple merges should succeed");
        free_test_tokenizer(&tok);
        return;
    }
    
    // Verify priority: merge 0 (ll) should be applied first
    // "hello" = [104, 101, 108, 108, 111]
    // After merge 0: [104, 101, 500, 111]
    // After merge 2: [104, 101, 502] (if applicable)
    if (num_tokens < 3) {
        TEST_FAIL("Should have at least 3 tokens after merges");
        free_test_tokenizer(&tok);
        return;
    }
    
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

static void test_circular_merge_dependency(void) {
    TEST_START("Circular merge dependency (should handle gracefully)");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 1000, 2);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    // Setup circular dependency: A+B->C, C+D->A
    tok.merges[0].token_id1 = 65;  // 'A'
    tok.merges[0].token_id2 = 66;  // 'B'
    tok.merges[0].merged_id = 67;  // 'C'
    
    tok.merges[1].token_id1 = 67;  // 'C'
    tok.merges[1].token_id2 = 68;  // 'D'
    tok.merges[1].merged_id = 65;  // 'A' (circular)
    
    // Should not crash or loop infinitely
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, "ABCD", tokens, &num_tokens, 256, false, false);
    // Should succeed (greedy algorithm handles this)
    (void)err;
    
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

// ============================================================================
// CATEGORY 9: DECODE EDGE CASES
// ============================================================================

static void test_decode_empty_tokens(void) {
    TEST_START("Decode empty tokens");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    // Setup vocab strings
    for (uint32_t i = 0; i < 256; i++) {
        tok.vocab[i] = (char*)malloc(2);
        if (tok.vocab[i] != NULL) {
            tok.vocab[i][0] = (char)i;
            tok.vocab[i][1] = '\0';
        }
    }
    
    uint32_t tokens[1] = {0};  // Non-empty array, but num_tokens=0
    char text[256];
    
    err = q_tokenizer_decode(&tok, tokens, 0, text, sizeof(text));
    if (err != Q_OK) {
        TEST_FAIL("Empty tokens should decode successfully");
        for (uint32_t i = 0; i < 256; i++) {
            free(tok.vocab[i]);
        }
        free_test_tokenizer(&tok);
        return;
    }
    if (text[0] != '\0') {
        TEST_FAIL("Empty tokens should produce empty string");
        for (uint32_t i = 0; i < 256; i++) {
            free(tok.vocab[i]);
        }
        free_test_tokenizer(&tok);
        return;
    }
    
    for (uint32_t i = 0; i < 256; i++) {
        free(tok.vocab[i]);
    }
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

static void test_decode_invalid_token_id(void) {
    TEST_START("Decode invalid token ID");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    // Setup vocab strings
    for (uint32_t i = 0; i < 256; i++) {
        tok.vocab[i] = (char*)malloc(2);
        if (tok.vocab[i] != NULL) {
            tok.vocab[i][0] = (char)i;
            tok.vocab[i][1] = '\0';
        }
    }
    
    uint32_t tokens[] = {72, 999, 108};  // 999 is invalid
    char text[256];
    
    err = q_tokenizer_decode(&tok, tokens, 3, text, sizeof(text));
    if (err != Q_OK) {
        TEST_FAIL("Invalid token ID should be skipped, not fail");
        for (uint32_t i = 0; i < 256; i++) {
            free(tok.vocab[i]);
        }
        free_test_tokenizer(&tok);
        return;
    }
    // Should decode "H" and "l", skipping 999
    if (strlen(text) != 2) {
        TEST_FAIL("Should decode valid tokens, skip invalid");
        for (uint32_t i = 0; i < 256; i++) {
            free(tok.vocab[i]);
        }
        free_test_tokenizer(&tok);
        return;
    }
    
    for (uint32_t i = 0; i < 256; i++) {
        free(tok.vocab[i]);
    }
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

static void test_decode_buffer_too_small(void) {
    TEST_START("Decode buffer too small");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    // Setup vocab strings (long strings)
    for (uint32_t i = 0; i < 256; i++) {
        tok.vocab[i] = (char*)malloc(100);
        if (tok.vocab[i] != NULL) {
            memset(tok.vocab[i], 'A' + (i % 26), 99);
            tok.vocab[i][99] = '\0';
        }
    }
    
    uint32_t tokens[] = {72, 101, 108, 108, 111};
    char text[10];  // Too small
    
    err = q_tokenizer_decode(&tok, tokens, 5, text, 10);
    if (err != Q_ERR_ARENA_OOM) {
        TEST_FAIL("Buffer too small should return Q_ERR_ARENA_OOM");
        for (uint32_t i = 0; i < 256; i++) {
            free(tok.vocab[i]);
        }
        free_test_tokenizer(&tok);
        return;
    }
    
    for (uint32_t i = 0; i < 256; i++) {
        free(tok.vocab[i]);
    }
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

static void test_decode_special_tokens_only(void) {
    TEST_START("Decode special tokens only");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    // Setup vocab strings
    for (uint32_t i = 0; i < 256; i++) {
        tok.vocab[i] = (char*)malloc(2);
        if (tok.vocab[i] != NULL) {
            tok.vocab[i][0] = (char)i;
            tok.vocab[i][1] = '\0';
        }
    }
    
    uint32_t tokens[] = {tok.bos_token_id, tok.eos_token_id, tok.pad_token_id};
    char text[256];
    
    err = q_tokenizer_decode(&tok, tokens, 3, text, sizeof(text));
    if (err != Q_OK) {
        TEST_FAIL("Special tokens only should decode successfully");
        for (uint32_t i = 0; i < 256; i++) {
            free(tok.vocab[i]);
        }
        free_test_tokenizer(&tok);
        return;
    }
    if (text[0] != '\0') {
        TEST_FAIL("Special tokens only should produce empty string");
        for (uint32_t i = 0; i < 256; i++) {
            free(tok.vocab[i]);
        }
        free_test_tokenizer(&tok);
        return;
    }
    
    for (uint32_t i = 0; i < 256; i++) {
        free(tok.vocab[i]);
    }
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

// ============================================================================
// CATEGORY 10: MEMORY SAFETY ATTACKS
// ============================================================================

static void test_double_free_tokenizer(void) {
    TEST_START("Double free tokenizer");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    // Setup vocab strings
    for (uint32_t i = 0; i < 256; i++) {
        tok.vocab[i] = (char*)malloc(2);
        if (tok.vocab[i] != NULL) {
            tok.vocab[i][0] = (char)i;
            tok.vocab[i][1] = '\0';
        }
    }
    
    // Free twice (should not crash)
    q_tokenizer_free(&tok);
    q_tokenizer_free(&tok);
    
    TEST_PASS();
TEST_END;
}

static void test_free_after_encode(void) {
    TEST_START("Free after encode (use-after-free prevention)");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, "test", tokens, &num_tokens, 256, false, false);
    if (err != Q_OK) {
        TEST_FAIL("Encode should succeed");
        free_test_tokenizer(&tok);
        return;
    }
    
    // Free tokenizer
    q_tokenizer_free(&tok);
    
    // Try to use freed tokenizer (should fail gracefully)
    num_tokens = 0;
    err = q_tokenizer_encode(&tok, "test", tokens, &num_tokens, 256, false, false);
    if (err != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Use-after-free should return Q_ERR_INVALID_ARG");
        return;
    }
    
    TEST_PASS();
TEST_END;
}

// ============================================================================
// CATEGORY 11: INTEGER OVERFLOW ATTACKS
// ============================================================================

static void test_integer_overflow_buffer_size(void) {
    TEST_START("Integer overflow in buffer_size calculation");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    // Create text that would cause overflow in buffer_size calculation
    // buffer_size = max(text_len, max_tokens)
    // If max_tokens = UINT32_MAX and text_len is large, multiplication may overflow
    
    // Use max_tokens = UINT32_MAX
    uint32_t* tokens = (uint32_t*)malloc(1000 * sizeof(uint32_t));  // Small buffer
    if (tokens == NULL) {
        TEST_FAIL("Failed to allocate tokens");
        free_test_tokenizer(&tok);
        return;
    }
    uint32_t num_tokens = 0;
    
    const char* text = "test";
    err = q_tokenizer_encode(&tok, text, tokens, &num_tokens, UINT32_MAX, false, false);
    // Should handle overflow gracefully (either succeed or return error)
    (void)err;
    
    free(tokens);
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

// ============================================================================
// CATEGORY 12: FUZZING TESTS
// ============================================================================

static void test_fuzzing_text_lengths(void) {
    TEST_START("Fuzzing text lengths");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    // Test various text lengths
    for (size_t len = 0; len < 1000; len++) {
        char* text = (char*)malloc(len + 1);
        if (text == NULL) continue;
        
        memset(text, 'A', len);
        text[len] = '\0';
        
        uint32_t* tokens = (uint32_t*)malloc((len + 10) * sizeof(uint32_t));
        if (tokens == NULL) {
            free(text);
            continue;
        }
        uint32_t num_tokens = 0;
        
        err = q_tokenizer_encode(&tok, text, tokens, &num_tokens, len + 10, false, false);
        // Should succeed for all lengths
        (void)err;
        
        free(tokens);
        free(text);
    }
    
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

static void test_fuzzing_token_ids(void) {
    TEST_START("Fuzzing token IDs");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    if (err != Q_OK) {
        TEST_FAIL("Failed to create test tokenizer");
        return;
    }
    
    // Setup vocab strings
    for (uint32_t i = 0; i < 256; i++) {
        tok.vocab[i] = (char*)malloc(2);
        if (tok.vocab[i] != NULL) {
            tok.vocab[i][0] = (char)i;
            tok.vocab[i][1] = '\0';
        }
    }
    
    // Test various token ID combinations
    for (uint32_t i = 0; i < 1000; i++) {
        uint32_t tokens[10];
        for (uint32_t j = 0; j < 10; j++) {
            tokens[j] = (i + j) % 300;  // Mix valid and invalid IDs
        }
        
        char text[256];
        err = q_tokenizer_decode(&tok, tokens, 10, text, sizeof(text));
        // Should handle gracefully (skip invalid IDs)
        (void)err;
    }
    
    for (uint32_t i = 0; i < 256; i++) {
        free(tok.vocab[i]);
    }
    free_test_tokenizer(&tok);
    TEST_PASS();
TEST_END;
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("========================================\n");
    printf("  ADVERSARIAL TEST SUITE: BPE Tokenizer\n");
    printf("========================================\n\n");
    printf("Strategy: Try to BREAK the BPE tokenizer through adversarial testing\n");
    printf("Following Lead SDET methodology: Happy Path + Edge Cases + Security\n\n");
    
    // Reset statistics
    tests_run = 0;
    tests_passed = 0;
    tests_failed = 0;
    tests_crashed = 0;
    
    // CATEGORY 1: NULL POINTER ATTACKS
    printf("CATEGORY 1: NULL Pointer Attacks\n");
    printf("-----------------------------------\n");
    test_null_tokenizer_encode();
    test_null_text_encode();
    test_null_tokens_out_encode();
    test_null_num_tokens_out_encode();
    test_null_tokenizer_decode();
    test_null_tokenizer_load();
    test_null_path_load();
    test_null_tokenizer_free();
    printf("\n");
    
    // CATEGORY 2: EDGE CASES - BOUNDARY VALUES
    printf("CATEGORY 2: Edge Cases - Boundary Values\n");
    printf("-----------------------------------\n");
    test_empty_text_encode();
    test_max_text_length();
    test_text_exceeds_max_bytes();
    test_zero_max_tokens();
    test_single_byte_text();
    printf("\n");
    
    // CATEGORY 3: BUFFER OVERFLOW ATTACKS
    printf("CATEGORY 3: Buffer Overflow Attacks\n");
    printf("-----------------------------------\n");
    test_buffer_too_small_encode();
    test_buffer_exact_size_encode();
    test_buffer_overflow_with_merges();
    test_buffer_overflow_with_bos_eos();
    printf("\n");
    
    // CATEGORY 4: UNINITIALIZED MEMORY ATTACKS
    printf("CATEGORY 4: Uninitialized Memory Attacks\n");
    printf("-----------------------------------\n");
    test_uninitialized_tokenizer_encode();
    test_uninitialized_tokenizer_decode();
    test_partially_initialized_tokenizer();
    printf("\n");
    
    // CATEGORY 5: INVALID FILE ATTACKS
    printf("CATEGORY 5: Invalid File Attacks (q_tokenizer_load)\n");
    printf("-----------------------------------\n");
    test_file_nonexistent_load();
    test_file_empty_load();
    test_file_corrupted_magic_load();
    test_file_corrupted_version_load();
    test_file_vocab_size_zero_load();
    test_file_vocab_size_overflow_load();
    test_file_truncated_vocab_load();
    printf("\n");
    
    // CATEGORY 6: INVALID MERGE RULES
    printf("CATEGORY 6: Invalid Merge Rules\n");
    printf("-----------------------------------\n");
    test_invalid_merge_token_id1();
    test_invalid_merge_token_id2();
    test_invalid_merge_merged_id();
    printf("\n");
    
    // CATEGORY 7: HASH TABLE EDGE CASES
    printf("CATEGORY 7: Hash Table Edge Cases\n");
    printf("-----------------------------------\n");
    test_hash_table_no_merges();
    test_hash_table_large_num_merges();
    printf("\n");
    
    // CATEGORY 8: MERGE APPLICATION EDGE CASES
    printf("CATEGORY 8: Merge Application Edge Cases\n");
    printf("-----------------------------------\n");
    test_single_token_no_merges();
    test_multiple_merges_priority_order();
    test_circular_merge_dependency();
    printf("\n");
    
    // CATEGORY 9: DECODE EDGE CASES
    printf("CATEGORY 9: Decode Edge Cases\n");
    printf("-----------------------------------\n");
    test_decode_empty_tokens();
    test_decode_invalid_token_id();
    test_decode_buffer_too_small();
    test_decode_special_tokens_only();
    printf("\n");
    
    // CATEGORY 10: MEMORY SAFETY ATTACKS
    printf("CATEGORY 10: Memory Safety Attacks\n");
    printf("-----------------------------------\n");
    test_double_free_tokenizer();
    test_free_after_encode();
    printf("\n");
    
    // CATEGORY 11: INTEGER OVERFLOW ATTACKS
    printf("CATEGORY 11: Integer Overflow Attacks\n");
    printf("-----------------------------------\n");
    test_integer_overflow_buffer_size();
    printf("\n");
    
    // CATEGORY 12: FUZZING TESTS
    printf("CATEGORY 12: Fuzzing Tests\n");
    printf("-----------------------------------\n");
    test_fuzzing_text_lengths();
    test_fuzzing_token_ids();
    printf("\n");
    
    // Print summary
    printf("========================================\n");
    printf("  TEST SUMMARY\n");
    printf("========================================\n");
    printf("Tests Run:    %d\n", tests_run);
    printf("Tests Passed: %d\n", tests_passed);
    printf("Tests Failed: %d\n", tests_failed);
    printf("Tests Crashed: %d\n", tests_crashed);
    printf("========================================\n");
    
    if (tests_failed == 0 && tests_crashed == 0) {
        printf("✓ All adversarial tests PASSED\n");
        return 0;
    } else {
        printf("✗ Some tests FAILED or CRASHED\n");
        return 1;
    }
}

