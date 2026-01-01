#include "qorus.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

// Tokenizer binary file format:
// Header (32 bytes):
//   - magic: uint32_t (0x51544B52 = 'QTKR')
//   - version: uint32_t (1)
//   - vocab_size: uint32_t
//   - num_merges: uint32_t
//   - bos_token_id: uint32_t
//   - eos_token_id: uint32_t
//   - pad_token_id: uint32_t
//   - reserved: uint32_t[1]
// Vocab: For each token [vocab_size]:
//   - length: uint8_t (token string length)
//   - token_bytes: char[length] (token string, not null-terminated)
// Merges: For each merge [num_merges]:
//   - token_id1: uint32_t
//   - token_id2: uint32_t
//   - merged_id: uint32_t

#define TOKENIZER_MAGIC 0x51544B52  // 'QTKR'
#define TOKENIZER_VERSION 1
#define TOKENIZER_HEADER_SIZE 32

// Helper: Read uint32_t from file (little-endian)
static q_error_code read_u32(FILE* f, uint32_t* out) {
    uint8_t bytes[4];
    if (fread(bytes, 1, 4, f) != 4) {
        return Q_ERR_FILE_OPEN;
    }
    *out = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) | 
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    return Q_OK;
}

// Helper: Read uint8_t from file
static q_error_code read_u8(FILE* f, uint8_t* out) {
    if (fread(out, 1, 1, f) != 1) {
        return Q_ERR_FILE_OPEN;
    }
    return Q_OK;
}

// Load tokenizer from binary file
q_error_code q_tokenizer_load(q_tokenizer* restrict tok, const char* tokenizer_path) {
    Q_VALIDATE_PTR_OR_RETURN(tok, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(tokenizer_path, Q_ERR_INVALID_ARG);
    
    // Initialize tokenizer
    memset(tok, 0, sizeof(q_tokenizer));
    
    FILE* f = fopen(tokenizer_path, "rb");
    if (f == NULL) {
        return Q_ERR_FILE_OPEN;
    }
    
    // Read header
    uint32_t magic;
    q_error_code err = read_u32(f, &magic);
    if (err != Q_OK || magic != TOKENIZER_MAGIC) {
        fclose(f);
        return (err != Q_OK) ? err : Q_ERR_INVALID_MAGIC;
    }
    
    uint32_t version;
    err = read_u32(f, &version);
    if (err != Q_OK || version != TOKENIZER_VERSION) {
        fclose(f);
        return (err != Q_OK) ? err : Q_ERR_INVALID_ARG;
    }
    
    err = read_u32(f, &tok->vocab_size);
    if (err != Q_OK) {
        fclose(f);
        return err;
    }
    
    uint32_t num_merges;
    err = read_u32(f, &num_merges);
    if (err != Q_OK) {
        fclose(f);
        return err;
    }
    
    err = read_u32(f, &tok->bos_token_id);
    if (err != Q_OK) {
        fclose(f);
        return err;
    }
    
    err = read_u32(f, &tok->eos_token_id);
    if (err != Q_OK) {
        fclose(f);
        return err;
    }
    
    err = read_u32(f, &tok->pad_token_id);
    if (err != Q_OK) {
        fclose(f);
        return err;
    }
    
    // Skip reserved bytes
    uint32_t reserved;
    err = read_u32(f, &reserved);
    if (err != Q_OK) {
        fclose(f);
        return err;
    }
    
    // Validate sizes
    if (tok->vocab_size == 0 || tok->vocab_size > 1000000) {
        fclose(f);
        return Q_ERR_INVALID_SIZE;
    }
    if (num_merges > 1000000) {
        fclose(f);
        return Q_ERR_INVALID_SIZE;
    }
    
    // Allocate vocab array
    tok->vocab = (char**)calloc(tok->vocab_size, sizeof(char*));
    if (tok->vocab == NULL) {
        fclose(f);
        return Q_ERR_ALLOC_FAILED;
    }
    
    // Read vocab tokens
    for (uint32_t i = 0; i < tok->vocab_size; i++) {
        uint8_t length;
        err = read_u8(f, &length);
        if (err != Q_OK || length == 0) {
            // Cleanup on error
            for (uint32_t j = 0; j < i; j++) {
                free(tok->vocab[j]);
            }
            free(tok->vocab);
            fclose(f);
            return (err != Q_OK) ? err : Q_ERR_INVALID_SIZE;
        }
        
        tok->vocab[i] = (char*)malloc(length + 1);
        if (tok->vocab[i] == NULL) {
            // Cleanup on error
            for (uint32_t j = 0; j < i; j++) {
                free(tok->vocab[j]);
            }
            free(tok->vocab);
            fclose(f);
            return Q_ERR_ALLOC_FAILED;
        }
        
        if (fread(tok->vocab[i], 1, length, f) != length) {
            // Cleanup on error
            free(tok->vocab[i]);
            for (uint32_t j = 0; j < i; j++) {
                free(tok->vocab[j]);
            }
            free(tok->vocab);
            fclose(f);
            return Q_ERR_FILE_OPEN;
        }
        tok->vocab[i][length] = '\0';  // Null-terminate
    }
    
    // Allocate merges array
    tok->num_merges = num_merges;
    if (num_merges > 0) {
        tok->merges = (q_bpe_merge*)calloc(num_merges, sizeof(q_bpe_merge));
        if (tok->merges == NULL) {
            // Cleanup vocab
            for (uint32_t i = 0; i < tok->vocab_size; i++) {
                free(tok->vocab[i]);
            }
            free(tok->vocab);
            fclose(f);
            return Q_ERR_ALLOC_FAILED;
        }
        
        // Read merges
        for (uint32_t i = 0; i < num_merges; i++) {
            err = read_u32(f, &tok->merges[i].token_id1);
            if (err != Q_OK) {
                free(tok->merges);
                for (uint32_t j = 0; j < tok->vocab_size; j++) {
                    free(tok->vocab[j]);
                }
                free(tok->vocab);
                fclose(f);
                return err;
            }
            
            err = read_u32(f, &tok->merges[i].token_id2);
            if (err != Q_OK) {
                free(tok->merges);
                for (uint32_t j = 0; j < tok->vocab_size; j++) {
                    free(tok->vocab[j]);
                }
                free(tok->vocab);
                fclose(f);
                return err;
            }
            
            err = read_u32(f, &tok->merges[i].merged_id);
            if (err != Q_OK) {
                free(tok->merges);
                for (uint32_t j = 0; j < tok->vocab_size; j++) {
                    free(tok->vocab[j]);
                }
                free(tok->vocab);
                fclose(f);
                return err;
            }
        }
    }
    
    fclose(f);
    tok->initialized = true;
    return Q_OK;
}

// Simple BPE encode: text -> bytes -> apply merges -> token IDs
// This is a simplified version - full BPE requires regex splitting
q_error_code q_tokenizer_encode(
    q_tokenizer* restrict tok,
    const char* restrict text,
    uint32_t* restrict tokens_out,
    uint32_t* restrict num_tokens_out,
    uint32_t max_tokens,
    bool add_bos,
    bool add_eos
) {
    Q_VALIDATE_PTR_OR_RETURN(tok, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(text, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(tokens_out, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(num_tokens_out, Q_ERR_INVALID_ARG);
    Q_VALIDATE_OR_RETURN(tok->initialized, Q_ERR_INVALID_ARG);
    
    uint32_t pos = 0;
    
    // Add BOS token if requested
    if (add_bos) {
        if (pos >= max_tokens) {
            *num_tokens_out = 0;
            return Q_ERR_ARENA_OOM;
        }
        tokens_out[pos++] = tok->bos_token_id;
    }
    
    // Simple encoding: map each byte to its token ID (for now)
    // Full BPE would require regex splitting and merge application
    size_t text_len = strlen(text);
    for (size_t i = 0; i < text_len; i++) {
        if (pos >= max_tokens) {
            *num_tokens_out = pos;
            return Q_ERR_ARENA_OOM;
        }
        
        // Map byte to token ID (simple: byte value as token ID if < vocab_size)
        uint8_t byte_val = (uint8_t)text[i];
        if (byte_val < tok->vocab_size) {
            tokens_out[pos++] = byte_val;
        } else {
            // Fallback: use pad token for invalid bytes
            tokens_out[pos++] = tok->pad_token_id;
        }
    }
    
    // Add EOS token if requested
    if (add_eos) {
        if (pos >= max_tokens) {
            *num_tokens_out = pos;
            return Q_ERR_ARENA_OOM;
        }
        tokens_out[pos++] = tok->eos_token_id;
    }
    
    *num_tokens_out = pos;
    return Q_OK;
}

// Decode token IDs into text
q_error_code q_tokenizer_decode(
    q_tokenizer* restrict tok,
    const uint32_t* restrict tokens,
    uint32_t num_tokens,
    char* restrict text_out,
    size_t text_buf_size
) {
    Q_VALIDATE_PTR_OR_RETURN(tok, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(tokens, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(text_out, Q_ERR_INVALID_ARG);
    Q_VALIDATE_OR_RETURN(tok->initialized, Q_ERR_INVALID_ARG);
    Q_VALIDATE_OR_RETURN(text_buf_size > 0, Q_ERR_INVALID_SIZE);
    
    size_t pos = 0;
    
    for (uint32_t i = 0; i < num_tokens; i++) {
        uint32_t token_id = tokens[i];
        
        // Skip special tokens
        if (token_id == tok->bos_token_id || token_id == tok->eos_token_id || 
            token_id == tok->pad_token_id) {
            continue;
        }
        
        // Validate token ID
        if (token_id >= tok->vocab_size) {
            // Invalid token ID - skip or use placeholder
            continue;
        }
        
        // Get token string
        const char* token_str = tok->vocab[token_id];
        if (token_str == NULL) {
            continue;
        }
        
        size_t token_len = strlen(token_str);
        
        // Check buffer space
        if (pos + token_len >= text_buf_size - 1) {
            // Buffer full - truncate
            text_out[pos] = '\0';
            return Q_ERR_ARENA_OOM;
        }
        
        // Append token string
        memcpy(text_out + pos, token_str, token_len);
        pos += token_len;
    }
    
    text_out[pos] = '\0';
    return Q_OK;
}

// Free tokenizer resources
void q_tokenizer_free(q_tokenizer* restrict tok) {
    if (tok == NULL) {
        return;
    }
    
    if (tok->vocab != NULL) {
        for (uint32_t i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i]);
        }
        free(tok->vocab);
        tok->vocab = NULL;
    }
    
    if (tok->merges != NULL) {
        free(tok->merges);
        tok->merges = NULL;
    }
    
    memset(tok, 0, sizeof(q_tokenizer));
}
