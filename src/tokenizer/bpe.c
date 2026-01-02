// ============================================================================
// BPE TOKENIZER (Real Implementation)
// ============================================================================
// This file implements the complete BPE (Byte Pair Encoding) algorithm
// following the planning document: docs/BPE_TOKENIZER_PLAN.md
//
// Algorithm:
// 1. Split text into bytes (UTF-8 encoding)
// 2. Convert bytes to base token IDs
// 3. Apply BPE merges greedily (highest priority first)
// 4. Add special tokens (BOS/EOS) if requested
//
// This file replaces dummy_tokenizer.c with a complete BPE implementation.
//
// ============================================================================

#include "qorus.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

// Tokenizer binary file format constants
#define TOKENIZER_MAGIC 0x51544B52  // 'QTKR'
#define TOKENIZER_VERSION 1
#define TOKENIZER_HEADER_SIZE 32

// ============================================================================
// STEP 0: CHAIN OF THOUGHT - Problem Analysis
// ============================================================================
//
// UNDERSTAND:
// - Input: Text string (UTF-8 encoded)
// - Output: Array of token IDs (uint32_t)
// - Process: Apply BPE merges greedily to reduce token count
//
// BREAK DOWN:
// 1. Split text to bytes (UTF-8 handling)
// 2. Convert bytes to base token IDs
// 3. Apply merges iteratively (greedy algorithm)
// 4. Add special tokens (BOS/EOS)
//
// REASON:
// - Greedy algorithm: Apply highest priority merges first
// - Iterative: Continue until no more merges can be applied
// - In-place: Modify token list directly (efficient)
//
// EDGE CASES:
// - Empty text → return empty tokens (or BOS/EOS only)
// - Buffer overflow → return Q_ERR_ARENA_OOM
// - Invalid token IDs → validate during load, not in hot path
// - No merges → fallback to byte-level tokenization
//
// ============================================================================

// Maximum text length for internal buffers (safety limit)
#define MAX_TEXT_BYTES (1024 * 1024)  // 1MB max text

// ============================================================================
// Hash Table for BPE Merge Lookup (Optimization)
// ============================================================================

// Hash table entry (chaining for collisions)
typedef struct bpe_hash_entry {
    uint64_t key;                    // (token_id1 << 16) | token_id2
    uint32_t merged_id;              // Resulting merged token ID
    struct bpe_hash_entry* next;      // Next entry in chain (for collisions)
} bpe_hash_entry;

// Hash table structure
typedef struct {
    bpe_hash_entry** buckets;        // Array of bucket pointers
    size_t num_buckets;              // Number of buckets (power of 2)
    size_t num_entries;              // Number of entries
} bpe_hash_table;

// Helper: Find next power of 2 >= n
static inline size_t next_power_of_2(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    #if SIZE_MAX > UINT32_MAX
    n |= n >> 32;
    #endif
    return n + 1;
}

// Hash function: Multiplicative hash (Knuth)
static inline uint64_t hash_pair(uint32_t id1, uint32_t id2) {
    uint64_t key = ((uint64_t)id1 << 16) | id2;
    // Golden ratio multiplier for good distribution
    return key * 2654435761ULL;
}

// Build hash table from merge rules
// Called during q_tokenizer_load after merges are loaded
static q_error_code build_merge_hash_table(q_tokenizer* restrict tok) {
    Q_VALIDATE_PTR_OR_RETURN(tok, Q_ERR_INVALID_ARG);
    
    // Early return if no merges
    if (tok->num_merges == 0) {
        // Store hash table pointer in a way that doesn't require modifying q_tokenizer struct
        // For now, we'll use a static variable (not thread-safe, but acceptable for v1.0)
        // TODO: Add merge_hash_table field to q_tokenizer struct in future version
        return Q_OK;
    }
    
    // Calculate number of buckets (load factor 0.5)
    size_t num_buckets = next_power_of_2((size_t)tok->num_merges * 2);
    if (num_buckets == 0 || num_buckets > SIZE_MAX / sizeof(bpe_hash_entry*)) {
        return Q_ERR_OVERFLOW;
    }
    
    // Allocate hash table
    bpe_hash_table* ht = (bpe_hash_table*)calloc(1, sizeof(bpe_hash_table));
    if (ht == NULL) {
        return Q_ERR_ALLOC_FAILED;
    }
    
    ht->buckets = (bpe_hash_entry**)calloc(num_buckets, sizeof(bpe_hash_entry*));
    if (ht->buckets == NULL) {
        free(ht);
        return Q_ERR_ALLOC_FAILED;
    }
    
    ht->num_buckets = num_buckets;
    ht->num_entries = 0;
    
    // Insert all merge rules into hash table
    for (uint32_t i = 0; i < tok->num_merges; i++) {
        uint32_t id1 = tok->merges[i].token_id1;
        uint32_t id2 = tok->merges[i].token_id2;
        uint32_t merged = tok->merges[i].merged_id;
        
        // Calculate hash
        uint64_t hash = hash_pair(id1, id2);
        size_t bucket = hash % num_buckets;
        uint64_t key = ((uint64_t)id1 << 16) | id2;
        
        // Create entry
        bpe_hash_entry* entry = (bpe_hash_entry*)malloc(sizeof(bpe_hash_entry));
        if (entry == NULL) {
            // Cleanup on error
            for (size_t b = 0; b < num_buckets; b++) {
                bpe_hash_entry* e = ht->buckets[b];
                while (e != NULL) {
                    bpe_hash_entry* next = e->next;
                    free(e);
                    e = next;
                }
            }
            free(ht->buckets);
            free(ht);
            return Q_ERR_ALLOC_FAILED;
        }
        
        entry->key = key;
        entry->merged_id = merged;
        entry->next = ht->buckets[bucket];
        ht->buckets[bucket] = entry;
        ht->num_entries++;
    }
    
    // Store hash table pointer in tokenizer struct
    tok->merge_hash_table = (struct bpe_hash_table*)ht;
    
    return Q_OK;
}

// Lookup merge rule in hash table (O(1) amortized)
// Returns merged_id or UINT32_MAX if not found
static uint32_t lookup_merge_hash(
    const bpe_hash_table* restrict ht,
    uint32_t token_id1,
    uint32_t token_id2
) {
    if (ht == NULL) {
        return UINT32_MAX;
    }
    
    uint64_t hash = hash_pair(token_id1, token_id2);
    size_t bucket = hash % ht->num_buckets;
    uint64_t search_key = ((uint64_t)token_id1 << 16) | token_id2;
    
    // Traverse chain
    for (const bpe_hash_entry* entry = ht->buckets[bucket]; entry != NULL; entry = entry->next) {
        if (entry->key == search_key) {
            return entry->merged_id;
        }
    }
    
    return UINT32_MAX;  // Not found
}

// Wrapper to lookup using tokenizer's hash table
static inline uint32_t lookup_merge_in_tokenizer(
    const q_tokenizer* restrict tok,
    uint32_t token_id1,
    uint32_t token_id2
) {
    if (tok->merge_hash_table == NULL) {
        return UINT32_MAX;
    }
    return lookup_merge_hash((const bpe_hash_table*)tok->merge_hash_table, token_id1, token_id2);
}

// Free hash table
static void free_hash_table(bpe_hash_table* restrict ht) {
    if (ht == NULL) {
        return;
    }
    
    if (ht->buckets != NULL) {
        for (size_t i = 0; i < ht->num_buckets; i++) {
            bpe_hash_entry* entry = ht->buckets[i];
            while (entry != NULL) {
                bpe_hash_entry* next = entry->next;
                free(entry);
                entry = next;
            }
        }
        free(ht->buckets);
    }
    
    free(ht);
}

// ============================================================================
// STEP 1: MODEL CONSTRUCTION (MFR Phase 1)
// ============================================================================
//
// ENTITIES:
// - q_tokenizer: Already defined in qorus_types.h
// - q_bpe_merge: Already defined in qorus_types.h
// - Internal buffers: Stack-allocated arrays for efficiency
//
// MEMORY LAYOUT:
// - Vocab: Array of pointers → strings (heap-allocated)
// - Merges: Contiguous array of q_bpe_merge (12 bytes each)
// - Token processing: Stack buffers (MAX_TEXT_BYTES)
//
// CONSTRAINTS:
// - Thread-safe: No global mutable state
// - Memory safety: All allocations checked, cleanup on error
// - Validation: All inputs validated before processing
//
// ============================================================================

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
// This function is kept from dummy_tokenizer.c (unchanged)
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
            free((void*)tok->vocab);
            fclose(f);
            return (err != Q_OK) ? err : Q_ERR_INVALID_SIZE;
        }
        
        tok->vocab[i] = (char*)malloc(length + 1);
        if (tok->vocab[i] == NULL) {
            // Cleanup on error
            for (uint32_t j = 0; j < i; j++) {
                free(tok->vocab[j]);
            }
            free((void*)tok->vocab);
            fclose(f);
            return Q_ERR_ALLOC_FAILED;
        }
        
        if (fread(tok->vocab[i], 1, length, f) != length) {
            // Cleanup on error
            free(tok->vocab[i]);
            for (uint32_t j = 0; j < i; j++) {
                free(tok->vocab[j]);
            }
            free((void*)tok->vocab);
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
            free((void*)tok->vocab);
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
                free((void*)tok->vocab);
                fclose(f);
                return err;
            }
            
            err = read_u32(f, &tok->merges[i].token_id2);
            if (err != Q_OK) {
                free(tok->merges);
                for (uint32_t j = 0; j < tok->vocab_size; j++) {
                    free(tok->vocab[j]);
                }
                free((void*)tok->vocab);
                fclose(f);
                return err;
            }
            
            err = read_u32(f, &tok->merges[i].merged_id);
            if (err != Q_OK) {
                free(tok->merges);
                for (uint32_t j = 0; j < tok->vocab_size; j++) {
                    free(tok->vocab[j]);
                }
                free((void*)tok->vocab);
                fclose(f);
                return err;
            }
            
            // CRITICAL: Validate merge rule token IDs are valid
            if (tok->merges[i].token_id1 >= tok->vocab_size ||
                tok->merges[i].token_id2 >= tok->vocab_size ||
                tok->merges[i].merged_id >= tok->vocab_size) {
                free(tok->merges);
                for (uint32_t j = 0; j < tok->vocab_size; j++) {
                    free(tok->vocab[j]);
                }
                free((void*)tok->vocab);
                fclose(f);
                return Q_ERR_INVALID_ARG;
            }
        }
        
        // Build hash table for fast merge lookup (optimization)
        q_error_code err_hash = build_merge_hash_table(tok);
        if (err_hash != Q_OK) {
            // Cleanup on error
            free(tok->merges);
            for (uint32_t j = 0; j < tok->vocab_size; j++) {
                free(tok->vocab[j]);
            }
            free((void*)tok->vocab);
            fclose(f);
            return err_hash;
        }
    } else {
        // No merges, hash table is NULL
        tok->merge_hash_table = NULL;
    }
    
    if (fclose(f) != 0) {
        #ifdef DEBUG
        fprintf(stderr, "WARNING: q_tokenizer_load: fclose() failed, but data already loaded\n");
        #endif
    }
    tok->initialized = true;
    return Q_OK;
}

// Helper: Split text into bytes (UTF-8 aware, but simplified for v1.0)
// For v1.0, we treat each byte as a separate unit
// Future: Implement proper UTF-8 decoding and regex splitting
static q_error_code split_text_to_bytes(
    const char* restrict text,
    uint8_t* restrict bytes_out,
    size_t* restrict num_bytes_out,
    size_t max_bytes
) {
    Q_VALIDATE_PTR_OR_RETURN(text, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(bytes_out, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(num_bytes_out, Q_ERR_INVALID_ARG);
    
    size_t text_len = strlen(text);
    if (text_len > max_bytes) {
        return Q_ERR_ARENA_OOM;
    }
    
    // Simple byte-by-byte copy (UTF-8 handling deferred to future version)
    for (size_t i = 0; i < text_len; i++) {
        bytes_out[i] = (uint8_t)text[i];
    }
    
    *num_bytes_out = text_len;
    return Q_OK;
}

// Helper: Convert bytes to base token IDs
// Each byte value becomes a token ID (if < vocab_size)
static q_error_code bytes_to_token_ids(
    const q_tokenizer* restrict tok,
    const uint8_t* restrict bytes,
    size_t num_bytes,
    uint32_t* restrict token_ids_out,
    size_t* restrict num_tokens_out,
    size_t max_tokens
) {
    Q_VALIDATE_PTR_OR_RETURN(tok, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(bytes, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(token_ids_out, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(num_tokens_out, Q_ERR_INVALID_ARG);
    
    if (num_bytes > max_tokens) {
        return Q_ERR_ARENA_OOM;
    }
    
    for (size_t i = 0; i < num_bytes; i++) {
        uint8_t byte_val = bytes[i];
        // Map byte to token ID (byte value = token ID if < vocab_size)
        if (byte_val < tok->vocab_size) {
            token_ids_out[i] = byte_val;
        } else {
            // Fallback: use pad token for invalid bytes
            token_ids_out[i] = tok->pad_token_id;
        }
    }
    
    *num_tokens_out = num_bytes;
    return Q_OK;
}

// Helper: Apply BPE merges greedily
// Algorithm: Iterate through merges in priority order, apply all applicable merges
// Continue until no more merges can be applied
static q_error_code apply_bpe_merges(
    const q_tokenizer* restrict tok,
    uint32_t* restrict token_ids,
    size_t* restrict num_tokens,
    size_t max_tokens __attribute__((unused))
) {
    Q_VALIDATE_PTR_OR_RETURN(tok, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(token_ids, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(num_tokens, Q_ERR_INVALID_ARG);
    
    // Early return if no merges
    if (tok->num_merges == 0 || *num_tokens < 2) {
        return Q_OK;
    }
    
    bool changed = true;
    
    // Iterate while merges are being applied
    while (changed) {
        changed = false;
        
        // For each merge rule in priority order (index = priority)
        for (uint32_t i = 0; i < tok->num_merges; i++) {
            uint32_t id1 = tok->merges[i].token_id1;
            uint32_t id2 = tok->merges[i].token_id2;
            uint32_t merged;
            
            // OPTIMIZATION: Use hash table for O(1) lookup if available
            // Fallback to direct access if hash table not built (for tests)
            if (tok->merge_hash_table != NULL) {
                merged = lookup_merge_in_tokenizer(tok, id1, id2);
                if (merged == UINT32_MAX) {
                    // Hash table lookup failed, use direct access (defensive)
                    merged = tok->merges[i].merged_id;
                }
            } else {
                // Fallback: Direct access (for compatibility with tests)
                merged = tok->merges[i].merged_id;
            }
            
            // Find all adjacent pairs (id1, id2) in token list
            // Process from left to right, but re-check after each merge
            for (size_t j = 0; j < *num_tokens - 1; j++) {
                if (token_ids[j] == id1 && token_ids[j + 1] == id2) {
                    // Apply merge: replace pair with merged_id
                    token_ids[j] = merged;
                    
                    // Remove token_ids[j+1] by shifting left
                    if (j + 2 < *num_tokens) {
                        memmove(&token_ids[j + 1], &token_ids[j + 2],
                                (*num_tokens - j - 2) * sizeof(uint32_t));
                    }
                    
                    (*num_tokens)--;
                    changed = true;
                    
                    // Re-check this position (may have another merge)
                    // j-- will be incremented by loop, so we check j again
                    if (j > 0) {
                        j--;  // Re-check previous position too
                    }
                }
            }
        }
    }
    
    return Q_OK;
}

// Helper: Add special tokens (BOS/EOS) to token list
static q_error_code add_special_tokens(
    const q_tokenizer* restrict tok,
    uint32_t* restrict tokens,
    size_t* restrict num_tokens,
    size_t max_tokens,
    bool add_bos,
    bool add_eos
) {
    Q_VALIDATE_PTR_OR_RETURN(tok, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(tokens, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(num_tokens, Q_ERR_INVALID_ARG);
    
    size_t needed = *num_tokens;
    if (add_bos) needed++;
    if (add_eos) needed++;
    
    if (needed > max_tokens) {
        return Q_ERR_ARENA_OOM;
    }
    
    // Shift tokens right to make room for BOS
    if (add_bos) {
        if (*num_tokens > 0) {
            memmove(&tokens[1], &tokens[0], *num_tokens * sizeof(uint32_t));
        }
        tokens[0] = tok->bos_token_id;
        (*num_tokens)++;
    }
    
    // Append EOS
    if (add_eos) {
        tokens[*num_tokens] = tok->eos_token_id;
        (*num_tokens)++;
    }
    
    return Q_OK;
}

// Main BPE encoding function
// This replaces the dummy implementation in dummy_tokenizer.c
q_error_code q_tokenizer_encode(
    q_tokenizer* restrict tok,
    const char* restrict text,
    uint32_t* restrict tokens_out,
    uint32_t* restrict num_tokens_out,
    uint32_t max_tokens,
    bool add_bos,
    bool add_eos
) {
    // STEP 0.5: VALIDATION (Preconditions)
    Q_VALIDATE_PTR_OR_RETURN(tok, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(text, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(tokens_out, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(num_tokens_out, Q_ERR_INVALID_ARG);
    Q_VALIDATE_OR_RETURN(tok->initialized, Q_ERR_INVALID_ARG);
    Q_VALIDATE_OR_RETURN(max_tokens > 0, Q_ERR_INVALID_SIZE);
    
    // Handle empty text
    if (text[0] == '\0') {
        *num_tokens_out = 0;
        // Still add BOS/EOS if requested
        if (add_bos || add_eos) {
            size_t num_tokens_temp = 0;
            q_error_code err = add_special_tokens(tok, tokens_out, &num_tokens_temp, max_tokens, add_bos, add_eos);
            if (err == Q_OK) {
                *num_tokens_out = (uint32_t)num_tokens_temp;
            }
            return err;
        }
        return Q_OK;
    }
    
    // Calculate text length (bounded check)
    size_t text_len = strlen(text);
    if (text_len > MAX_TEXT_BYTES) {
        return Q_ERR_ARENA_OOM;
    }
    
    // Allocate buffers dynamically (avoid stack overflow)
    // Worst case: one token per byte (before merges)
    size_t buffer_size = (text_len > (size_t)max_tokens) ? text_len : (size_t)max_tokens;
    uint8_t* bytes = (uint8_t*)malloc(buffer_size);
    if (bytes == NULL) {
        return Q_ERR_ALLOC_FAILED;
    }
    
    uint32_t* token_ids = (uint32_t*)malloc(buffer_size * sizeof(uint32_t));
    if (token_ids == NULL) {
        free(bytes);
        return Q_ERR_ALLOC_FAILED;
    }
    
    size_t num_bytes = 0;
    size_t num_tokens = 0;
    
    // Step 1: Split text to bytes
    q_error_code err = split_text_to_bytes(text, bytes, &num_bytes, buffer_size);
    if (err != Q_OK) {
        free(bytes);
        free(token_ids);
        return err;
    }
    
    // Step 2: Convert bytes to base token IDs
    err = bytes_to_token_ids(tok, bytes, num_bytes, token_ids, &num_tokens, buffer_size);
    if (err != Q_OK) {
        free(bytes);
        free(token_ids);
        return err;
    }
    
    // Step 3: Apply BPE merges (greedy)
    err = apply_bpe_merges(tok, token_ids, &num_tokens, buffer_size);
    if (err != Q_OK) {
        free(bytes);
        free(token_ids);
        return err;
    }
    
    // Step 4: Add special tokens (BOS/EOS)
    err = add_special_tokens(tok, token_ids, &num_tokens, buffer_size, add_bos, add_eos);
    if (err != Q_OK) {
        free(bytes);
        free(token_ids);
        return err;
    }
    
    // Step 5: Validate final count
    if (num_tokens > (size_t)max_tokens) {
        free(bytes);
        free(token_ids);
        return Q_ERR_ARENA_OOM;
    }
    
    // Step 6: Copy to output
    memcpy(tokens_out, token_ids, num_tokens * sizeof(uint32_t));
    *num_tokens_out = (uint32_t)num_tokens;
    
    // Cleanup
    free(bytes);
    free(token_ids);
    
    return Q_OK;
}


// Decode token IDs into text
// This function is kept from dummy_tokenizer.c (unchanged)
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
// This function is kept from dummy_tokenizer.c (unchanged)
void q_tokenizer_free(q_tokenizer* restrict tok) {
    if (tok == NULL) {
        return;
    }
    
    if (tok->vocab != NULL) {
        for (uint32_t i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i]);
        }
        // CRITICAL FIX: Explicit cast improves clarity and suppresses warning
        // Proof: free() accepts void*, char** -> void* is valid conversion
        // Safety: No change in behavior, only clarity improvement
        free((void*)tok->vocab);
        tok->vocab = NULL;
    }
    
    if (tok->merges != NULL) {
        free(tok->merges);
        tok->merges = NULL;
    }
    
    // Free hash table if it exists
    if (tok->merge_hash_table != NULL) {
        free_hash_table((bpe_hash_table*)tok->merge_hash_table);
        tok->merge_hash_table = NULL;
    }
    
    memset(tok, 0, sizeof(q_tokenizer));
}
