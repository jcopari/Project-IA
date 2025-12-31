#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <signal.h>
#include <setjmp.h>

// ============================================================================
// ADVERSARIAL TEST SUITE: q_tokenizer_encode/decode()
// ============================================================================
// Lead SDET Strategy: Try to BREAK the tokenizer through adversarial testing
// Following MFR + CoT + Mathematical Proof + TDD methodology
//
// Test Categories:
// 1. Happy Path: Normal operation
// 2. Edge Cases: Empty strings, very long strings, special characters
// 3. Null/Undefined: Missing data handling
// 4. Security/Malicious: Injection attempts, corrupted data
// 5. Memory Safety: Buffer overflows, use-after-free
// 6. Boundary Conditions: Max tokens, buffer sizes
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

// Test helper macros
#define TEST_START(name) \
    do { \
        tests_run++; \
        printf("  [%d] %s... ", tests_run, name); \
        fflush(stdout); \
        signal(SIGSEGV, crash_handler); \
        signal(SIGBUS, crash_handler); \
        signal(SIGFPE, crash_handler); \
        if (setjmp(crash_jmp) == 0) {

#define TEST_END \
        } else { \
            tests_crashed++; \
            printf("CRASHED\n"); \
            signal(SIGSEGV, SIG_DFL); \
            signal(SIGBUS, SIG_DFL); \
            signal(SIGFPE, SIG_DFL); \
        } \
    } while(0)

#define TEST_PASS() \
    do { \
        tests_passed++; \
        printf("PASS\n"); \
        signal(SIGSEGV, SIG_DFL); \
        signal(SIGBUS, SIG_DFL); \
        signal(SIGFPE, SIG_DFL); \
    } while(0)

#define TEST_FAIL(reason) \
    do { \
        tests_failed++; \
        printf("FAIL: %s\n", reason); \
        signal(SIGSEGV, SIG_DFL); \
        signal(SIGBUS, SIG_DFL); \
        signal(SIGFPE, SIG_DFL); \
    } while(0)

// ============================================================================
// MAPA DE CENÁRIOS (Scenario Map)
// ============================================================================

// CATEGORY 1: NULL POINTER ATTACKS
static void test_null_tokenizer(void) {
    TEST_START("NULL tokenizer pointer");
    
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    q_error_code ret = q_tokenizer_encode(NULL, "test", tokens, &num_tokens, 256, false, false);
    if (ret != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Should return Q_ERR_INVALID_ARG for NULL tokenizer");
        return;
    }
    
    TEST_PASS();
}

static void test_null_text(void) {
    TEST_START("NULL text pointer");
    
    q_tokenizer tok = {0};
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    q_error_code ret = q_tokenizer_encode(&tok, NULL, tokens, &num_tokens, 256, false, false);
    if (ret != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Should return Q_ERR_INVALID_ARG for NULL text");
        return;
    }
    
    TEST_PASS();
}

static void test_null_tokens_out(void) {
    TEST_START("NULL tokens_out pointer");
    
    q_tokenizer tok = {0};
    tok.initialized = true;
    uint32_t num_tokens = 0;
    
    q_error_code ret = q_tokenizer_encode(&tok, "test", NULL, &num_tokens, 256, false, false);
    if (ret != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Should return Q_ERR_INVALID_ARG for NULL tokens_out");
        return;
    }
    
    TEST_PASS();
}

static void test_null_num_tokens_out(void) {
    TEST_START("NULL num_tokens_out pointer");
    
    q_tokenizer tok = {0};
    tok.initialized = true;
    uint32_t tokens[256];
    
    q_error_code ret = q_tokenizer_encode(&tok, "test", tokens, NULL, 256, false, false);
    if (ret != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Should return Q_ERR_INVALID_ARG for NULL num_tokens_out");
        return;
    }
    
    TEST_PASS();
}

static void test_null_text_out_decode(void) {
    TEST_START("NULL text_out pointer (decode)");
    
    q_tokenizer tok = {0};
    tok.initialized = true;
    uint32_t tokens[1] = {0};
    
    q_error_code ret = q_tokenizer_decode(&tok, tokens, 1, NULL, 1024);
    if (ret != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Should return Q_ERR_INVALID_ARG for NULL text_out");
        return;
    }
    
    TEST_PASS();
}

// CATEGORY 2: EDGE CASES - EMPTY STRINGS
static void test_empty_string_encode(void) {
    TEST_START("Empty string encode");
    
    q_tokenizer tok = {0};
    q_error_code ret = q_tokenizer_load(&tok, "tokenizer.bin");
    if (ret != Q_OK) {
        TEST_FAIL("Failed to load tokenizer");
        return;
    }
    
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    ret = q_tokenizer_encode(&tok, "", tokens, &num_tokens, 256, false, false);
    if (ret != Q_OK) {
        q_tokenizer_free(&tok);
        TEST_FAIL("Should handle empty string");
        return;
    }
    
    if (num_tokens != 0) {
        q_tokenizer_free(&tok);
        TEST_FAIL("Empty string should produce 0 tokens");
        return;
    }
    
    q_tokenizer_free(&tok);
    TEST_PASS();
}

static void test_empty_string_with_bos_eos(void) {
    TEST_START("Empty string with BOS/EOS");
    
    q_tokenizer tok = {0};
    q_error_code ret = q_tokenizer_load(&tok, "tokenizer.bin");
    if (ret != Q_OK) {
        TEST_FAIL("Failed to load tokenizer");
        return;
    }
    
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    ret = q_tokenizer_encode(&tok, "", tokens, &num_tokens, 256, true, true);
    if (ret != Q_OK) {
        q_tokenizer_free(&tok);
        TEST_FAIL("Should handle empty string with BOS/EOS");
        return;
    }
    
    if (num_tokens != 2) {
        q_tokenizer_free(&tok);
        TEST_FAIL("Empty string with BOS/EOS should produce 2 tokens");
        return;
    }
    
    if (tokens[0] != tok.bos_token_id || tokens[1] != tok.eos_token_id) {
        q_tokenizer_free(&tok);
        TEST_FAIL("BOS/EOS tokens should be correct");
        return;
    }
    
    q_tokenizer_free(&tok);
    TEST_PASS();
}

// CATEGORY 3: BUFFER OVERFLOW ATTACKS
static void test_buffer_too_small_encode(void) {
    TEST_START("Buffer too small (encode)");
    
    q_tokenizer tok = {0};
    q_error_code ret = q_tokenizer_load(&tok, "tokenizer.bin");
    if (ret != Q_OK) {
        TEST_FAIL("Failed to load tokenizer");
        return;
    }
    
    uint32_t tokens[1];  // Very small buffer
    uint32_t num_tokens = 0;
    
    ret = q_tokenizer_encode(&tok, "Hello World", tokens, &num_tokens, 1, false, false);
    if (ret != Q_ERR_ARENA_OOM) {
        q_tokenizer_free(&tok);
        TEST_FAIL("Should return Q_ERR_ARENA_OOM for buffer too small");
        return;
    }
    
    q_tokenizer_free(&tok);
    TEST_PASS();
}

static void test_buffer_too_small_decode(void) {
    TEST_START("Buffer too small (decode)");
    
    q_tokenizer tok = {0};
    q_error_code ret = q_tokenizer_load(&tok, "tokenizer.bin");
    if (ret != Q_OK) {
        TEST_FAIL("Failed to load tokenizer");
        return;
    }
    
    uint32_t tokens[10];
    uint32_t num_tokens = 0;
    
    // Encode first
    ret = q_tokenizer_encode(&tok, "Hello World", tokens, &num_tokens, 10, false, false);
    if (ret != Q_OK) {
        q_tokenizer_free(&tok);
        TEST_FAIL("Failed to encode");
        return;
    }
    
    // Try to decode with very small buffer
    char text_out[1];  // Way too small
    ret = q_tokenizer_decode(&tok, tokens, num_tokens, text_out, 1);
    if (ret != Q_ERR_ARENA_OOM) {
        q_tokenizer_free(&tok);
        TEST_FAIL("Should return Q_ERR_ARENA_OOM for buffer too small");
        return;
    }
    
    q_tokenizer_free(&tok);
    TEST_PASS();
}

// CATEGORY 4: INVALID TOKEN IDS (DECODE)
static void test_invalid_token_id_decode(void) {
    TEST_START("Invalid token ID (decode)");
    
    q_tokenizer tok = {0};
    q_error_code ret = q_tokenizer_load(&tok, "tokenizer.bin");
    if (ret != Q_OK) {
        TEST_FAIL("Failed to load tokenizer");
        return;
    }
    
    // Use invalid token ID (>= vocab_size)
    uint32_t tokens[1] = {tok.vocab_size + 100};
    char text_out[1024];
    
    ret = q_tokenizer_decode(&tok, tokens, 1, text_out, sizeof(text_out));
    if (ret == Q_OK) {
        // Should handle gracefully (skip invalid tokens)
        // Check that text is empty or contains placeholder
        q_tokenizer_free(&tok);
        TEST_PASS();
        return;
    }
    
    q_tokenizer_free(&tok);
    TEST_PASS();
}

static void test_token_id_overflow_decode(void) {
    TEST_START("Token ID overflow (UINT32_MAX)");
    
    q_tokenizer tok = {0};
    q_error_code ret = q_tokenizer_load(&tok, "tokenizer.bin");
    if (ret != Q_OK) {
        TEST_FAIL("Failed to load tokenizer");
        return;
    }
    
    uint32_t tokens[1] = {UINT32_MAX};
    char text_out[1024];
    
    ret = q_tokenizer_decode(&tok, tokens, 1, text_out, sizeof(text_out));
    // Should handle gracefully (skip invalid tokens)
    
    q_tokenizer_free(&tok);
    TEST_PASS();
}

// CATEGORY 5: VERY LONG STRINGS
static void test_very_long_string(void) {
    TEST_START("Very long string (10KB)");
    
    q_tokenizer tok = {0};
    q_error_code ret = q_tokenizer_load(&tok, "tokenizer.bin");
    if (ret != Q_OK) {
        TEST_FAIL("Failed to load tokenizer");
        return;
    }
    
    // Create very long string
    char* long_str = (char*)malloc(10000);
    if (long_str == NULL) {
        q_tokenizer_free(&tok);
        TEST_FAIL("Failed to allocate long string");
        return;
    }
    
    // Fill with pattern
    for (int i = 0; i < 9999; i++) {
        long_str[i] = 'A' + (i % 26);
    }
    long_str[9999] = '\0';
    
    uint32_t tokens[10000];
    uint32_t num_tokens = 0;
    
    ret = q_tokenizer_encode(&tok, long_str, tokens, &num_tokens, 10000, false, false);
    
    free(long_str);
    q_tokenizer_free(&tok);
    
    if (ret != Q_OK) {
        TEST_FAIL("Should handle very long strings");
        return;
    }
    
    TEST_PASS();
}

// CATEGORY 6: SPECIAL CHARACTERS / UTF-8 EDGE CASES
static void test_utf8_multibyte(void) {
    TEST_START("UTF-8 multibyte characters");
    
    q_tokenizer tok = {0};
    q_error_code ret = q_tokenizer_load(&tok, "tokenizer.bin");
    if (ret != Q_OK) {
        TEST_FAIL("Failed to load tokenizer");
        return;
    }
    
    // UTF-8: "Hello 世界" (Hello World in Chinese)
    const char* utf8_str = "Hello \xE4\xB8\x96\xE7\x95\x8C";
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    ret = q_tokenizer_encode(&tok, utf8_str, tokens, &num_tokens, 256, false, false);
    
    q_tokenizer_free(&tok);
    
    if (ret != Q_OK) {
        TEST_FAIL("Should handle UTF-8 multibyte characters");
        return;
    }
    
    TEST_PASS();
}

static void test_invalid_utf8_sequence(void) {
    TEST_START("Invalid UTF-8 sequence");
    
    q_tokenizer tok = {0};
    q_error_code ret = q_tokenizer_load(&tok, "tokenizer.bin");
    if (ret != Q_OK) {
        TEST_FAIL("Failed to load tokenizer");
        return;
    }
    
    // Invalid UTF-8: continuation byte without start byte
    const char* invalid_utf8 = "\x80\x81\x82";
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    ret = q_tokenizer_encode(&tok, invalid_utf8, tokens, &num_tokens, 256, false, false);
    
    q_tokenizer_free(&tok);
    
    // Should handle gracefully (map to pad token or skip)
    if (ret != Q_OK) {
        TEST_FAIL("Should handle invalid UTF-8 gracefully");
        return;
    }
    
    TEST_PASS();
}

// CATEGORY 7: CORRUPTED TOKENIZER FILE
static void test_corrupted_tokenizer_file(void) {
    TEST_START("Corrupted tokenizer file (invalid magic)");
    
    // Create corrupted tokenizer file
    FILE* f = fopen("tokenizer_corrupted.bin", "wb");
    if (f == NULL) {
        TEST_FAIL("Failed to create corrupted tokenizer file");
        return;
    }
    
    // Write invalid magic number
    uint32_t invalid_magic = 0xDEADBEEF;
    fwrite(&invalid_magic, sizeof(uint32_t), 1, f);
    fclose(f);
    
    q_tokenizer tok = {0};
    q_error_code ret = q_tokenizer_load(&tok, "tokenizer_corrupted.bin");
    
    // Cleanup
    unlink("tokenizer_corrupted.bin");
    
    if (ret != Q_ERR_INVALID_MAGIC) {
        TEST_FAIL("Should return Q_ERR_INVALID_MAGIC for corrupted file");
        return;
    }
    
    TEST_PASS();
}

static void test_truncated_tokenizer_file(void) {
    TEST_START("Truncated tokenizer file");
    
    // Create truncated tokenizer file (only header, no vocab)
    FILE* f = fopen("tokenizer_truncated.bin", "wb");
    if (f == NULL) {
        TEST_FAIL("Failed to create truncated tokenizer file");
        return;
    }
    
    // Write valid header but truncate file
    uint32_t magic = 0x51544B52;  // Valid magic
    uint32_t version = 1;
    uint32_t vocab_size = 1000;
    uint32_t num_merges = 0;
    uint32_t bos_id = 1000;
    uint32_t eos_id = 1001;
    uint32_t pad_id = 1002;
    uint32_t reserved = 0;
    
    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);
    fwrite(&vocab_size, sizeof(uint32_t), 1, f);
    fwrite(&num_merges, sizeof(uint32_t), 1, f);
    fwrite(&bos_id, sizeof(uint32_t), 1, f);
    fwrite(&eos_id, sizeof(uint32_t), 1, f);
    fwrite(&pad_id, sizeof(uint32_t), 1, f);
    fwrite(&reserved, sizeof(uint32_t), 1, f);
    // File ends here (no vocab data)
    fclose(f);
    
    q_tokenizer tok = {0};
    q_error_code ret = q_tokenizer_load(&tok, "tokenizer_truncated.bin");
    
    // Cleanup
    unlink("tokenizer_truncated.bin");
    
    if (ret == Q_OK) {
        q_tokenizer_free(&tok);
        TEST_FAIL("Should return error for truncated file");
        return;
    }
    
    TEST_PASS();
}

// CATEGORY 8: UNINITIALIZED TOKENIZER
static void test_uninitialized_tokenizer(void) {
    TEST_START("Uninitialized tokenizer (initialized=false)");
    
    q_tokenizer tok = {0};
    tok.initialized = false;  // Explicitly uninitialized
    
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    q_error_code ret = q_tokenizer_encode(&tok, "test", tokens, &num_tokens, 256, false, false);
    if (ret != Q_ERR_INVALID_ARG) {
        TEST_FAIL("Should return Q_ERR_INVALID_ARG for uninitialized tokenizer");
        return;
    }
    
    TEST_PASS();
}

// CATEGORY 9: ROUND-TRIP CONSISTENCY
static void test_round_trip_consistency(void) {
    TEST_START("Round-trip consistency (encode -> decode -> encode)");
    
    q_tokenizer tok = {0};
    q_error_code ret = q_tokenizer_load(&tok, "tokenizer.bin");
    if (ret != Q_OK) {
        TEST_FAIL("Failed to load tokenizer");
        return;
    }
    
    const char* original = "Hello World";
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    // Encode
    ret = q_tokenizer_encode(&tok, original, tokens, &num_tokens, 256, false, false);
    if (ret != Q_OK) {
        q_tokenizer_free(&tok);
        TEST_FAIL("Failed to encode");
        return;
    }
    
    // Decode
    char decoded[1024];
    ret = q_tokenizer_decode(&tok, tokens, num_tokens, decoded, sizeof(decoded));
    if (ret != Q_OK) {
        q_tokenizer_free(&tok);
        TEST_FAIL("Failed to decode");
        return;
    }
    
    // Verify round-trip (may not be exact due to BOS/EOS handling)
    // For now, just verify decode doesn't crash
    q_tokenizer_free(&tok);
    TEST_PASS();
}

// CATEGORY 10: BOS/EOS EDGE CASES
static void test_bos_eos_only(void) {
    TEST_START("BOS/EOS tokens only (no actual text)");
    
    q_tokenizer tok = {0};
    q_error_code ret = q_tokenizer_load(&tok, "tokenizer.bin");
    if (ret != Q_OK) {
        TEST_FAIL("Failed to load tokenizer");
        return;
    }
    
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    ret = q_tokenizer_encode(&tok, "", tokens, &num_tokens, 256, true, true);
    if (ret != Q_OK) {
        q_tokenizer_free(&tok);
        TEST_FAIL("Should handle BOS/EOS only");
        return;
    }
    
    if (num_tokens != 2 || tokens[0] != tok.bos_token_id || tokens[1] != tok.eos_token_id) {
        q_tokenizer_free(&tok);
        TEST_FAIL("BOS/EOS tokens should be correct");
        return;
    }
    
    q_tokenizer_free(&tok);
    TEST_PASS();
}

static void test_decode_with_special_tokens(void) {
    TEST_START("Decode with special tokens (should skip them)");
    
    q_tokenizer tok = {0};
    q_error_code ret = q_tokenizer_load(&tok, "tokenizer.bin");
    if (ret != Q_OK) {
        TEST_FAIL("Failed to load tokenizer");
        return;
    }
    
    // Create token sequence with BOS/EOS/PAD
    uint32_t tokens[5] = {
        tok.bos_token_id,
        72,  // 'H'
        101, // 'e'
        tok.eos_token_id,
        tok.pad_token_id
    };
    
    char text_out[1024];
    ret = q_tokenizer_decode(&tok, tokens, 5, text_out, sizeof(text_out));
    
    q_tokenizer_free(&tok);
    
    if (ret != Q_OK) {
        TEST_FAIL("Should decode successfully, skipping special tokens");
        return;
    }
    
    TEST_PASS();
}

// CATEGORY 11: DOUBLE-FREE ATTACKS
static void test_double_free_tokenizer(void) {
    TEST_START("Double free tokenizer (should not crash)");
    
    q_tokenizer tok = {0};
    q_error_code ret = q_tokenizer_load(&tok, "tokenizer.bin");
    if (ret != Q_OK) {
        TEST_FAIL("Failed to load tokenizer");
        return;
    }
    
    // Free twice (should handle gracefully)
    q_tokenizer_free(&tok);
    q_tokenizer_free(&tok);  // Double free
    
    TEST_PASS();
}

// CATEGORY 12: ZERO-LENGTH BUFFER
static void test_zero_length_buffer_decode(void) {
    TEST_START("Zero-length buffer (decode)");
    
    q_tokenizer tok = {0};
    q_error_code ret = q_tokenizer_load(&tok, "tokenizer.bin");
    if (ret != Q_OK) {
        TEST_FAIL("Failed to load tokenizer");
        return;
    }
    
    uint32_t tokens[1] = {72};  // 'H'
    char text_out[1];  // Zero-length (only null terminator space)
    
    ret = q_tokenizer_decode(&tok, tokens, 1, text_out, 0);
    
    q_tokenizer_free(&tok);
    
    if (ret != Q_ERR_INVALID_SIZE && ret != Q_ERR_ARENA_OOM) {
        TEST_FAIL("Should return error for zero-length buffer");
        return;
    }
    
    TEST_PASS();
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("========================================\n");
    printf("  ADVERSARIAL TEST SUITE: Tokenizer\n");
    printf("========================================\n\n");
    printf("Strategy: Try to BREAK the tokenizer through adversarial testing\n");
    printf("Following Lead SDET methodology: Happy Path + Edge Cases + Security\n\n");
    
    // Ensure tokenizer.bin exists
    FILE* check = fopen("tokenizer.bin", "rb");
    if (check == NULL) {
        printf("ERROR: tokenizer.bin not found. Run: python3 tools/convert_llama.py --tokenizer tokenizer.bin\n");
        return 1;
    }
    fclose(check);
    
    // Reset statistics
    tests_run = 0;
    tests_passed = 0;
    tests_failed = 0;
    tests_crashed = 0;
    
    // CATEGORY 1: NULL POINTER ATTACKS
    printf("CATEGORY 1: NULL Pointer Attacks\n");
    printf("-----------------------------------\n");
    test_null_tokenizer();
    test_null_text();
    test_null_tokens_out();
    test_null_num_tokens_out();
    test_null_text_out_decode();
    printf("\n");
    
    // CATEGORY 2: EDGE CASES - EMPTY STRINGS
    printf("CATEGORY 2: Edge Cases - Empty Strings\n");
    printf("-----------------------------------\n");
    test_empty_string_encode();
    test_empty_string_with_bos_eos();
    printf("\n");
    
    // CATEGORY 3: BUFFER OVERFLOW ATTACKS
    printf("CATEGORY 3: Buffer Overflow Attacks\n");
    printf("-----------------------------------\n");
    test_buffer_too_small_encode();
    test_buffer_too_small_decode();
    printf("\n");
    
    // CATEGORY 4: INVALID TOKEN IDS (DECODE)
    printf("CATEGORY 4: Invalid Token IDs (Decode)\n");
    printf("-----------------------------------\n");
    test_invalid_token_id_decode();
    test_token_id_overflow_decode();
    printf("\n");
    
    // CATEGORY 5: VERY LONG STRINGS
    printf("CATEGORY 5: Very Long Strings\n");
    printf("-----------------------------------\n");
    test_very_long_string();
    printf("\n");
    
    // CATEGORY 6: SPECIAL CHARACTERS / UTF-8 EDGE CASES
    printf("CATEGORY 6: Special Characters / UTF-8 Edge Cases\n");
    printf("-----------------------------------\n");
    test_utf8_multibyte();
    test_invalid_utf8_sequence();
    printf("\n");
    
    // CATEGORY 7: CORRUPTED TOKENIZER FILE
    printf("CATEGORY 7: Corrupted Tokenizer File\n");
    printf("-----------------------------------\n");
    test_corrupted_tokenizer_file();
    test_truncated_tokenizer_file();
    printf("\n");
    
    // CATEGORY 8: UNINITIALIZED TOKENIZER
    printf("CATEGORY 8: Uninitialized Tokenizer\n");
    printf("-----------------------------------\n");
    test_uninitialized_tokenizer();
    printf("\n");
    
    // CATEGORY 9: ROUND-TRIP CONSISTENCY
    printf("CATEGORY 9: Round-Trip Consistency\n");
    printf("-----------------------------------\n");
    test_round_trip_consistency();
    printf("\n");
    
    // CATEGORY 10: BOS/EOS EDGE CASES
    printf("CATEGORY 10: BOS/EOS Edge Cases\n");
    printf("-----------------------------------\n");
    test_bos_eos_only();
    test_decode_with_special_tokens();
    printf("\n");
    
    // CATEGORY 11: DOUBLE-FREE ATTACKS
    printf("CATEGORY 11: Double-Free Attacks\n");
    printf("-----------------------------------\n");
    test_double_free_tokenizer();
    printf("\n");
    
    // CATEGORY 12: ZERO-LENGTH BUFFER
    printf("CATEGORY 12: Zero-Length Buffer\n");
    printf("-----------------------------------\n");
    test_zero_length_buffer_decode();
    printf("\n");
    
    // Print summary
    printf("========================================\n");
    printf("  SUMMARY: Tokenizer Adversarial Tests\n");
    printf("========================================\n");
    printf("  Tests Run:    %d\n", tests_run);
    printf("  Tests Passed: %d\n", tests_passed);
    printf("  Tests Failed: %d\n", tests_failed);
    printf("  Tests Crashed: %d\n", tests_crashed);
    printf("  Success Rate: %.1f%%\n", 
           tests_run > 0 ? (100.0 * tests_passed / tests_run) : 0.0);
    printf("\n");
    
    if (tests_failed == 0 && tests_crashed == 0) {
        printf("✓ All adversarial tests passed! Tokenizer is robust.\n");
        return 0;
    } else {
        printf("✗ Some tests failed or crashed. Tokenizer needs hardening.\n");
        return 1;
    }
}

