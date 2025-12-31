#include "qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <tokenizer.bin>\n", argv[0]);
        return 1;
    }
    
    const char* tokenizer_path = argv[1];
    
    printf("Test: Tokenizer Load & Encode/Decode\n");
    printf("=====================================\n\n");
    
    // Initialize tokenizer
    q_tokenizer tok;
    q_error_code err = q_tokenizer_load(&tok, tokenizer_path);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: Failed to load tokenizer: %s\n", q_strerror(err));
        return 1;
    }
    
    printf("✓ Tokenizer loaded successfully\n");
    printf("  Vocab size: %u\n", tok.vocab_size);
    printf("  BOS token ID: %u\n", tok.bos_token_id);
    printf("  EOS token ID: %u\n", tok.eos_token_id);
    printf("  PAD token ID: %u\n", tok.pad_token_id);
    printf("\n");
    
    // Test encode: "Hello World"
    const char* test_text = "Hello World";
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    printf("Test: Encode \"%s\"\n", test_text);
    err = q_tokenizer_encode(&tok, test_text, tokens, &num_tokens, 256, false, false);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: Failed to encode: %s\n", q_strerror(err));
        q_tokenizer_free(&tok);
        return 1;
    }
    
    printf("✓ Encoded successfully\n");
    printf("  Number of tokens: %u\n", num_tokens);
    printf("  Token IDs: ");
    for (uint32_t i = 0; i < num_tokens; i++) {
        printf("%u ", tokens[i]);
    }
    printf("\n\n");
    
    // Test decode
    char decoded_text[1024];
    printf("Test: Decode tokens back to text\n");
    err = q_tokenizer_decode(&tok, tokens, num_tokens, decoded_text, sizeof(decoded_text));
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: Failed to decode: %s\n", q_strerror(err));
        q_tokenizer_free(&tok);
        return 1;
    }
    
    printf("✓ Decoded successfully\n");
    printf("  Decoded text: \"%s\"\n", decoded_text);
    printf("\n");
    
    // Test with BOS/EOS
    printf("Test: Encode with BOS/EOS\n");
    uint32_t tokens_with_special[256];
    uint32_t num_tokens_special = 0;
    
    err = q_tokenizer_encode(&tok, test_text, tokens_with_special, &num_tokens_special, 
                             256, true, true);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: Failed to encode with BOS/EOS: %s\n", q_strerror(err));
        q_tokenizer_free(&tok);
        return 1;
    }
    
    printf("✓ Encoded with BOS/EOS successfully\n");
    printf("  Number of tokens: %u\n", num_tokens_special);
    printf("  First token (should be BOS): %u\n", tokens_with_special[0]);
    printf("  Last token (should be EOS): %u\n", tokens_with_special[num_tokens_special - 1]);
    printf("\n");
    
    // Cleanup
    q_tokenizer_free(&tok);
    
    printf("✓ All tests passed!\n");
    return 0;
}

