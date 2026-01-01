#ifndef QORUS_TYPES_H
#define QORUS_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>

// Alinhamento obrigatório para AVX2/AVX-512
#define Q_ALIGN 64
#define Q_MAGIC 0x514F5231  // 'QOR1'
#define Q_HEADER_SIZE 64
#define Q_FIRST_TENSOR_OFFSET Q_HEADER_SIZE

// Macro de alinhamento (branchless)
#define Q_ALIGN_SIZE(size) (((size) + Q_ALIGN - 1) & ~(Q_ALIGN - 1))

// Precision tolerance constants (from PRECISION_STANDARDS.md)
#define Q_EPSILON_ABS_F32   1e-5f   // Absolute tolerance for FP32
#define Q_EPSILON_REL_F32   1e-4f   // Relative tolerance for FP32
#define Q_EPSILON_ABS_APPROX 2.5e-1f  // Absolute tolerance for approximations (polynomial exp)
#define Q_EPSILON_REL_APPROX 5e-1f    // Relative tolerance for approximations (polynomial exp)
#define Q_EPSILON_ABS_Q4_VAL 1e-2f  // Absolute tolerance for Q4_0 quantization
#define Q_EPSILON_REL_Q4_VAL 5e-2f  // Relative tolerance for Q4_0 quantization

// ============================================================================
// Error Codes (Standardized)
// ============================================================================

typedef enum {
    Q_OK = 0,
    Q_ERR_NULL_PTR = -1,
    Q_ERR_FILE_OPEN = -2,
    Q_ERR_FILE_STAT = -3,
    Q_ERR_FILE_TOO_SMALL = -4,
    Q_ERR_MMAP_FAILED = -5,
    Q_ERR_INVALID_MAGIC = -6,
    Q_ERR_ALLOC_FAILED = -7,
    Q_ERR_ARENA_OOM = -8,
    Q_ERR_INVALID_CONFIG = -9,
    // Security validation errors (added for Release mode safety)
    Q_ERR_INVALID_ARG = -10,      // Invalid argument (null pointer, wrong type, etc.)
    Q_ERR_ALIASING = -11,         // Input/output aliasing detected
    Q_ERR_OVERFLOW = -12,         // Integer overflow detected
    Q_ERR_MISALIGNED = -13,       // Pointer not properly aligned
    Q_ERR_INVALID_DTYPE = -14,    // Wrong data type
    Q_ERR_INVALID_SIZE = -15      // Invalid size (zero, not multiple of N, etc.)
} q_error_code;

// ============================================================================
// Memory Mapping Strategy
// ============================================================================

typedef enum {
    Q_MMAP_LAZY = 0,      // Lazy loading (fast startup, page faults on first access)
    Q_MMAP_EAGER = 1      // Eager loading (slow startup, fast first inference via MAP_POPULATE)
} q_mmap_strategy;

#ifdef DEBUG
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define Q_ASSERT_ALIGNED(ptr) \
    do { \
        if (((uintptr_t)(ptr) % Q_ALIGN) != 0) { \
            fprintf(stderr, "ERROR: %s not aligned to %d bytes at %p\n", \
                    #ptr, Q_ALIGN, (void*)(ptr)); \
            abort(); \
        } \
    } while(0)
#else
#define Q_ASSERT_ALIGNED(ptr) ((void)0)
#endif

// ============================================================================
// Validation Macros (Security: Always Active, Optimized for Release)
// ============================================================================

// Critical validation: Returns error code in Release, aborts in DEBUG
// Uses __builtin_expect for branch prediction optimization
// For functions returning q_error_code
#ifdef DEBUG
#define Q_VALIDATE_OR_RETURN(condition, error_code) \
    do { \
        if (__builtin_expect(!(condition), 0)) { \
            fprintf(stderr, "ERROR: Validation failed at %s:%d\n", \
                    __FILE__, __LINE__); \
            abort(); \
        } \
    } while (0)
#else
#define Q_VALIDATE_OR_RETURN(condition, error_code) \
    do { \
        if (__builtin_expect(!(condition), 0)) { \
            return (error_code); \
        } \
    } while (0)
#endif

// Pointer validation (null check) - returns error code
#define Q_VALIDATE_PTR_OR_RETURN(ptr, error_code) \
    Q_VALIDATE_OR_RETURN((ptr) != NULL, (error_code))

// Alignment validation - returns error code
// NOTE: Q_ALIGN is 64 bytes, but AVX2 needs 32-byte alignment
// This macro checks 32-byte alignment (not Q_ALIGN) for AVX2 compatibility
#define Q_VALIDATE_ALIGNED_OR_RETURN(ptr, error_code) \
    do { \
        uintptr_t ptr_addr = (uintptr_t)(ptr); \
        uintptr_t misalignment = ptr_addr % 32; \
        if (misalignment != 0) { \
            fprintf(stderr, "ERROR: Q_VALIDATE_ALIGNED_OR_RETURN failed at %s:%d\n", __FILE__, __LINE__); \
            fprintf(stderr, "  ptr=%p, addr=%zu, %% 32 = %zu\n", (ptr), ptr_addr, misalignment); \
            return (error_code); \
        } \
    } while (0)

// Size validation (multiple of N) - returns error code
#define Q_VALIDATE_MULTIPLE_OR_RETURN(value, multiple, error_code) \
    Q_VALIDATE_OR_RETURN((value) % (multiple) == 0, (error_code))

// Non-zero validation - returns error code
#define Q_VALIDATE_NONZERO_OR_RETURN(value, error_code) \
    Q_VALIDATE_OR_RETURN((value) != 0, (error_code))

// Overflow validation - returns error code
#define Q_VALIDATE_NO_OVERFLOW_OR_RETURN(a, b) \
    Q_VALIDATE_OR_RETURN((a) <= UINT32_MAX / (b), Q_ERR_OVERFLOW)

// ============================================================================
// Data Types
// ============================================================================

typedef enum {
    Q_F32  = 0,
    Q_Q8_0 = 1, // Weights (Embeddings/Output)
    Q_Q4_0 = 2  // Weights (Dense Layers)
} q_dtype;

// ============================================================================
// Tokenizer Types (BPE)
// ============================================================================

// BPE Merge Rule: (token_id1, token_id2) -> merged_token_id
typedef struct {
    uint32_t token_id1;   // First token ID
    uint32_t token_id2;   // Second token ID
    uint32_t merged_id;   // Resulting merged token ID
} q_bpe_merge;

// Tokenizer Structure (BPE - Byte Pair Encoding)
typedef struct {
    // Vocabulary: Array of token strings (variable length)
    char** vocab;              // Array of token strings [vocab_size]
    uint32_t vocab_size;       // Total vocabulary size
    
    // BPE Merges: Array of merge rules
    q_bpe_merge* merges;      // Array of BPE merge rules [num_merges]
    uint32_t num_merges;       // Number of BPE merges
    
    // Special Tokens
    uint32_t bos_token_id;     // Beginning of sequence token ID
    uint32_t eos_token_id;     // End of sequence token ID
    uint32_t pad_token_id;     // Padding token ID
    
    // Initialization flag
    bool initialized;          // True if tokenizer loaded successfully
} q_tokenizer;

// ============================================================================
// Tensor Types
// ============================================================================

// Q4_0 Quantization Block (20 bytes: 16 bytes qs + 4 bytes scale)
// Dequantization: value = (quantized - 8) * scale
typedef struct {
    uint8_t  qs[16];  // 16 bytes: 32 quantized values (4 bits each)
    float    scale;   // 4 bytes: scale factor for the block
} __attribute__((packed)) q_block_q4_0;

// Header compacto (64 bytes = 1 cache line, alinhado)
typedef struct {
    uint32_t magic;          // 4 bytes
    uint32_t version;        // 4 bytes
    uint32_t vocab_size;     // 4 bytes
    uint32_t dim;            // 4 bytes
    uint32_t hidden_dim;     // 4 bytes
    uint32_t n_layers;       // 4 bytes
    uint32_t n_heads;        // 4 bytes
    uint32_t n_kv_heads;     // 4 bytes
    uint32_t max_seq_len;    // 4 bytes
    float    rope_freq_base; // 4 bytes
    float    rms_norm_eps;   // 4 bytes: RMSNorm epsilon for numerical stability
    uint32_t reserved[5];    // 20 bytes reservados
    // Total: 64 bytes (9*4 + 4 + 4 + 5*4 = 36 + 4 + 4 + 20 = 64)
} __attribute__((packed, aligned(64))) q_model_header;

// Tensor View (alinhado para SIMD)
typedef struct {
    void*     data;          // Ponteiro para dados (Mmap ou Arena)
    float*    scales;        // Ponteiro para escalas (se quantizado)
    uint32_t  ne[4];         // Dimensões: [Batch, Head, Seq, Dim]
    size_t    nb[4];         // Strides em bytes
    q_dtype   type;          // Tipo de dado
    char      name[32];      // Debugging
} __attribute__((aligned(Q_ALIGN))) q_tensor;

// Contexto Global de Memória
typedef struct {
    // Tier 1: Static (Mmap)
    void*           weights_mmap;
    size_t          weights_size;
    q_model_header* header;

    // Tier 2: Persistent (KV Cache)
    void*           kv_buffer;
    size_t          kv_size;

    // Tier 3: Transient (Arena)
    void*           scratch_buffer;
    size_t          scratch_size;
    size_t          scratch_head;
    size_t          scratch_base_offset;  // Watermark: onde o scratchpad começa (modelo antes disso)
} q_context;

// ============================================================================
// Llama-3 Model Structures (Cache-Optimized)
// ============================================================================

// Llama-3 Configuration (exact model parameters)
typedef struct {
    uint32_t vocab_size;
    uint32_t dim;
    uint32_t hidden_dim;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;      // GQA: Grouped Query Attention
    uint32_t max_seq_len;
    float    rope_freq_base;
    float    rms_norm_eps;
} q_llama_config;

// Llama-3 Layer Structure (per-layer tensor views)
typedef struct {
    uint32_t layer_idx;       // Layer index (0..n_layers-1)
    q_tensor* attn_norm;      // [dim] (FP32)
    q_tensor* wq;             // [dim, dim] (Q4_0)
    q_tensor* wk;             // [dim, n_kv_heads*head_dim] (Q4_0)
    q_tensor* wv;             // [dim, n_kv_heads*head_dim] (Q4_0)
    q_tensor* wo;             // [dim, dim] (Q4_0)
    q_tensor* ffn_norm;       // [dim] (FP32)
    q_tensor* w_gate;         // [dim, hidden_dim] (Q4_0)
    q_tensor* w_up;           // [dim, hidden_dim] (Q4_0)
    q_tensor* w_down;        // [hidden_dim, dim] (Q4_0)
} q_llama_layer;

// Llama-3 Model Graph (tensor views pointing to mmap)
typedef struct {
    q_llama_config config;
    
    // Embeddings
    q_tensor* token_embd;     // [vocab_size, dim] (FP32)
    
    // Output Layer
    q_tensor* output_norm;     // [dim] (FP32)
    q_tensor* output;          // [vocab_size, dim] (FP32)
    
    // Layers (array of layer structures)
    q_llama_layer* layers;    // [n_layers] array of layer structures
    
    // RoPE cache (optional, for performance)
    float* rope_freqs;        // Pre-computed RoPE frequencies [head_dim/2]
    bool rope_cache_enabled;  // Flag: if true, use cached cos/sin
    float* rope_cos_cache;   // Cached cos values [max_seq_len, head_dim/2]
    float* rope_sin_cache;   // Cached sin values [max_seq_len, head_dim/2]
    
    // Context pointer (for arena access)
    q_context* ctx;           // Memory context (for arena allocations)
} q_llama_model;

#endif // QORUS_TYPES_H
