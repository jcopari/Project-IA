// ============================================================================
// TEST: BPE Tokenizer - Specification Tests (TDD)
// ============================================================================
// Following docs/BPE_TOKENIZER_PLAN.md FASE 4.2
// Tests validate the mathematical specification from FASE 3.4
//
// Test Strategy:
// 1. Basic test: "Hello" → [72, 101, 108, 108, 111] (no merges)
// 2. Merge test: "hello" with merge (108,108)→500 → [104, 101, 500, 111]
// 3. BOS/EOS test: "Hi" with add_bos/add_eos → [bos, 72, 105, eos]
// 4. Buffer overflow test: should return Q_ERR_ARENA_OOM
// 5. Empty text test: should return empty tokens (or BOS/EOS only)
// ============================================================================

#include "qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>

// Test helper: Create a minimal tokenizer for testing
static q_error_code create_test_tokenizer(q_tokenizer* tok, uint32_t vocab_size, uint32_t num_merges) {
    memset(tok, 0, sizeof(q_tokenizer));
    
    tok->vocab_size = vocab_size;
    tok->num_merges = num_merges;
    tok->bos_token_id = vocab_size;
    tok->eos_token_id = vocab_size + 1;
    tok->pad_token_id = vocab_size + 2;
    
    // Allocate vocab (simplified: just allocate pointers, don't fill strings)
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

// Test helper: Free test tokenizer
static void free_test_tokenizer(q_tokenizer* tok) {
    if (tok->vocab != NULL) {
        free((void*)tok->vocab);
    }
    if (tok->merges != NULL) {
        free(tok->merges);
    }
    memset(tok, 0, sizeof(q_tokenizer));
}

// Test 1: Basic encoding (no merges)
// Specification: "Hello" → [72, 101, 108, 108, 111]
static void test_basic_encode(void) {
    printf("Test 1: Basic encoding (no merges)\n");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    assert(err == Q_OK);
    
    const char* text = "Hello";
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, text, tokens, &num_tokens, 256, false, false);
    assert(err == Q_OK);
    assert(num_tokens == 5);
    assert(tokens[0] == 72);   // 'H'
    assert(tokens[1] == 101);  // 'e'
    assert(tokens[2] == 108);  // 'l'
    assert(tokens[3] == 108);  // 'l'
    assert(tokens[4] == 111);  // 'o'
    
    free_test_tokenizer(&tok);
    printf("  ✓ PASSED\n\n");
}

// Test 2: Encoding with merge
// Specification: "hello" with merge (108,108)→500 → [104, 101, 500, 111]
static void test_encode_with_merge(void) {
    printf("Test 2: Encoding with merge\n");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 1);
    assert(err == Q_OK);
    
    // Setup merge: (108, 108) -> 500 (merge two 'l' characters)
    tok.merges[0].token_id1 = 108;
    tok.merges[0].token_id2 = 108;
    tok.merges[0].merged_id = 500;
    
    const char* text = "hello";
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, text, tokens, &num_tokens, 256, false, false);
    assert(err == Q_OK);
    assert(num_tokens == 4);  // Reduced from 5 to 4 due to merge
    assert(tokens[0] == 104);  // 'h'
    assert(tokens[1] == 101);  // 'e'
    assert(tokens[2] == 500);  // merged 'll'
    assert(tokens[3] == 111);  // 'o'
    
    free_test_tokenizer(&tok);
    printf("  ✓ PASSED\n\n");
}

// Test 3: Encoding with BOS/EOS
// Specification: "Hi" with add_bos/add_eos → [bos, 72, 105, eos]
static void test_encode_with_special_tokens(void) {
    printf("Test 3: Encoding with BOS/EOS\n");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    assert(err == Q_OK);
    
    const char* text = "Hi";
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, text, tokens, &num_tokens, 256, true, true);
    assert(err == Q_OK);
    assert(num_tokens == 4);
    assert(tokens[0] == tok.bos_token_id);  // BOS
    assert(tokens[1] == 72);   // 'H'
    assert(tokens[2] == 105);  // 'i'
    assert(tokens[3] == tok.eos_token_id);  // EOS
    
    free_test_tokenizer(&tok);
    printf("  ✓ PASSED\n\n");
}

// Test 4: Buffer overflow
// Specification: Should return Q_ERR_ARENA_OOM if buffer too small
static void test_buffer_overflow(void) {
    printf("Test 4: Buffer overflow\n");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    assert(err == Q_OK);
    
    const char* text = "Hello World";
    uint32_t tokens[5];  // Too small buffer
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, text, tokens, &num_tokens, 5, false, false);
    assert(err == Q_ERR_ARENA_OOM);
    
    free_test_tokenizer(&tok);
    printf("  ✓ PASSED\n\n");
}

// Test 5: Empty text
// Specification: Should return empty tokens (or BOS/EOS only)
static void test_empty_text(void) {
    printf("Test 5: Empty text\n");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 0);
    assert(err == Q_OK);
    
    const char* text = "";
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    // Test without BOS/EOS
    err = q_tokenizer_encode(&tok, text, tokens, &num_tokens, 256, false, false);
    assert(err == Q_OK);
    assert(num_tokens == 0);
    
    // Test with BOS/EOS
    num_tokens = 0;
    err = q_tokenizer_encode(&tok, text, tokens, &num_tokens, 256, true, true);
    assert(err == Q_OK);
    assert(num_tokens == 2);
    assert(tokens[0] == tok.bos_token_id);
    assert(tokens[1] == tok.eos_token_id);
    
    free_test_tokenizer(&tok);
    printf("  ✓ PASSED\n\n");
}

// Test 6: Multiple merges (priority order)
// Specification: Merges should be applied in priority order (index order)
static void test_multiple_merges(void) {
    printf("Test 6: Multiple merges (priority order)\n");
    
    q_tokenizer tok;
    q_error_code err = create_test_tokenizer(&tok, 256, 2);
    assert(err == Q_OK);
    
    // Setup merges:
    // Merge 0 (high priority): (108, 108) -> 500 ('ll')
    // Merge 1 (low priority): (101, 108) -> 501 ('el')
    tok.merges[0].token_id1 = 108;
    tok.merges[0].token_id2 = 108;
    tok.merges[0].merged_id = 500;
    
    tok.merges[1].token_id1 = 101;
    tok.merges[1].token_id2 = 108;
    tok.merges[1].merged_id = 501;
    
    // Text "hello" should apply merge 0 first (higher priority)
    const char* text = "hello";
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    err = q_tokenizer_encode(&tok, text, tokens, &num_tokens, 256, false, false);
    assert(err == Q_OK);
    assert(num_tokens == 4);
    assert(tokens[0] == 104);  // 'h'
    assert(tokens[1] == 101);  // 'e'
    assert(tokens[2] == 500);  // merged 'll' (high priority merge applied)
    assert(tokens[3] == 111);  // 'o'
    
    free_test_tokenizer(&tok);
    printf("  ✓ PASSED\n\n");
}

int main(void) {
    printf("========================================\n");
    printf("BPE Tokenizer Specification Tests (TDD)\n");
    printf("========================================\n\n");
    
    test_basic_encode();
    test_encode_with_merge();
    test_encode_with_special_tokens();
    test_buffer_overflow();
    test_empty_text();
    test_multiple_merges();
    
    printf("========================================\n");
    printf("✓ All specification tests PASSED\n");
    printf("========================================\n");
    
    return 0;
}

