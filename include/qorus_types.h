#ifndef QORUS_TYPES_H
#define QORUS_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

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
#define Q_VALIDATE_ALIGNED_OR_RETURN(ptr, error_code) \
    Q_VALIDATE_OR_RETURN(((uintptr_t)(ptr) % Q_ALIGN) == 0, (error_code))

// Size validation (multiple of N) - returns error code
#define Q_VALIDATE_MULTIPLE_OR_RETURN(value, multiple, error_code) \
    Q_VALIDATE_OR_RETURN((value) % (multiple) == 0, (error_code))

// Non-zero validation - returns error code
#define Q_VALIDATE_NONZERO_OR_RETURN(value, error_code) \
    Q_VALIDATE_OR_RETURN((value) > 0, (error_code))

// Overflow validation (for multiplication) - returns error code
#define Q_VALIDATE_NO_OVERFLOW_OR_RETURN(a, b, error_code) \
    Q_VALIDATE_OR_RETURN((b) == 0 || (a) <= UINT32_MAX / (b), (error_code))

// Validation macros for functions returning void* (like q_arena_alloc)
// Returns NULL on failure in Release, aborts in DEBUG
#ifdef DEBUG
#define Q_VALIDATE_OR_RETURN_NULL(condition) \
    do { \
        if (__builtin_expect(!(condition), 0)) { \
            fprintf(stderr, "ERROR: Validation failed at %s:%d\n", \
                    __FILE__, __LINE__); \
            abort(); \
        } \
    } while (0)
#else
#define Q_VALIDATE_OR_RETURN_NULL(condition) \
    do { \
        if (__builtin_expect(!(condition), 0)) { \
            return NULL; \
        } \
    } while (0)
#endif

#define Q_VALIDATE_PTR_OR_RETURN_NULL(ptr) \
    Q_VALIDATE_OR_RETURN_NULL((ptr) != NULL)

// ============================================================================
// Debug Print Macros (Only Active in DEBUG Mode)
// ============================================================================

// Debug print macro: Only prints in DEBUG mode, zero overhead in Release
#ifdef DEBUG
#define Q_DEBUG_PRINT(...) do { \
    fprintf(stderr, __VA_ARGS__); \
} while(0)
#else
#define Q_DEBUG_PRINT(...) ((void)0)
#endif

// Debug write to stderr (for direct write() calls)
#ifdef DEBUG
#define Q_DEBUG_WRITE(msg, len) do { \
    write(2, msg, len); \
} while(0)
#else
#define Q_DEBUG_WRITE(msg, len) ((void)0)
#endif

typedef enum {
    Q_TYPE_INVALID = 0,  // Zero DEVE ser erro explícito (Fail-Fast)
    Q_F32  = 1,
    Q_Q8_0 = 2, // Pesos (Embeddings/Output)
    Q_Q4_0 = 3  // Pesos (Dense Layers)
} q_dtype;

// Q4_0 block structure (20 bytes: 16 bytes quantized data + 4 bytes scale)
// Each byte contains 2 quantized values (4 bits each, range 0-15)
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
    uint32_t n_kv_heads;      // GQA support
    uint32_t max_seq_len;
    float    rope_theta;      // RoPE base frequency (Llama 3 uses theta)
    float    rms_norm_eps;    // RMSNorm epsilon for numerical stability
} llama_config;

// Llama-3 Layer Structure (128 bytes = 2 cache lines, aligned to 64 bytes)
// Contains only VIEWS (lightweight pointers), not heavy data
typedef struct {
    // Attention weights (4 pointers = 32 bytes)
    q_tensor* wq;              // Query weights
    q_tensor* wk;              // Key weights (GQA: n_kv_heads < n_heads)
    q_tensor* wv;              // Value weights (GQA: n_kv_heads < n_heads)
    q_tensor* wo;              // Output projection
    
    // MLP weights (3 pointers = 24 bytes)
    q_tensor* w_gate;          // Gate projection (SiLU gate)
    q_tensor* w_up;            // Up projection
    q_tensor* w_down;          // Down projection
    
    // Normalization weights (2 pointers = 16 bytes)
    q_tensor* attn_norm;       // Pre-attention RMSNorm
    q_tensor* ffn_norm;        // Pre-MLP RMSNorm
    
    // Metadata (8 bytes)
    uint32_t layer_idx;        // Layer index (for debugging/performance)
    uint32_t _reserved;        // Reserved for future use
    
    // Padding explícito para 128 bytes (2 cache lines)
    // 128 - (32 + 24 + 16 + 8) = 48 bytes
    uint8_t _padding[48];
} __attribute__((aligned(64))) llama_layer;

// Compile-time assertions: Verificar tamanho e alinhamento
_Static_assert(sizeof(llama_layer) == 128, 
               "llama_layer must be exactly 128 bytes for cache alignment");
_Static_assert(_Alignof(llama_layer) == 64,
               "llama_layer must be 64-byte aligned");

// Llama-3 Model Structure (complete model)
typedef struct {
    llama_config config;       // Model configuration
    q_tensor*    token_embd;   // Token embeddings [vocab_size, dim]
    q_tensor*    output_norm;  // Output layer normalization [dim]
    q_tensor*    output;       // Output projection [vocab_size, dim]
    llama_layer* layers;       // Array contíguo de layers [n_layers]
    q_context*   ctx;          // Memory context (Tier 1/2/3)
    
    // RoPE pre-computation (Correção 2: Otimização crítica)
    float*       rope_freqs;        // [head_dim/2] frequências base pré-calculadas
    float*       rope_cos_cache;     // Opcional: [max_seq_len, head_dim] cache completo
    float*       rope_sin_cache;     // Opcional: [max_seq_len, head_dim] cache completo
    bool         rope_cache_enabled; // Flag se cache completo está disponível
} llama_model;

#endif // QORUS_TYPES_H

