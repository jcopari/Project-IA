#include "qorus.h"
#include <stdio.h>
#include <stdlib.h>

// Hello World Example: Tokenizer Encode/Decode
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <tokenizer.bin>\n", argv[0]);
        fprintf(stderr, "Example: %s tokenizer.bin\n", argv[0]);
        return 1;
    }
    
    const char* tokenizer_path = argv[1];
    
    printf("Qorus-IA: Hello World Example\n");
    printf("============================\n\n");
    
    // Load tokenizer
    q_tokenizer tok;
    q_error_code err = q_tokenizer_load(&tok, tokenizer_path);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: Failed to load tokenizer: %s\n", q_strerror(err));
        return 1;
    }
    
    printf("✓ Tokenizer loaded\n\n");
    
    // Encode "Hello World"
    const char* text = "Hello World";
    uint32_t tokens[256];
    uint32_t num_tokens = 0;
    
    printf("Encoding: \"%s\"\n", text);
    err = q_tokenizer_encode(&tok, text, tokens, &num_tokens, 256, true, true);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: Failed to encode: %s\n", q_strerror(err));
        q_tokenizer_free(&tok);
        return 1;
    }
    
    printf("Tokens: ");
    for (uint32_t i = 0; i < num_tokens; i++) {
        printf("%u ", tokens[i]);
    }
    printf("\n\n");
    
    // Decode back to text
    char decoded[1024];
    printf("Decoding tokens back to text...\n");
    err = q_tokenizer_decode(&tok, tokens, num_tokens, decoded, sizeof(decoded));
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: Failed to decode: %s\n", q_strerror(err));
        q_tokenizer_free(&tok);
        return 1;
    }
    
    printf("Decoded: \"%s\"\n\n", decoded);
    
    // Cleanup
    q_tokenizer_free(&tok);
    
    printf("✓ Hello World complete!\n");
    return 0;
}

