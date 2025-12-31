#ifndef QORUS_H
#define QORUS_H

#include "qorus_types.h"

// API Pública do Qorus-IA v2.0
// Este é o único header que deve ser incluído por código externo

// ============================================================================
// Memory Management API (Tier 1: Mmap, Tier 2: KV Cache, Tier 3: Arena)
// ============================================================================

// Inicializar memória (Tier 1: Mmap do arquivo modelo)
// Returns: Q_OK on success, negative q_error_code on error
q_error_code q_init_memory(q_context* restrict ctx, const char* model_path);

// Alocar KV Cache (Tier 2: Buffer persistente)
// Returns: Q_OK on success, negative q_error_code on error
q_error_code q_alloc_kv_cache(q_context* restrict ctx, size_t kv_size);

// Alocar Arena (Tier 3: Buffer temporário)
// Returns: Q_OK on success, negative q_error_code on error
q_error_code q_alloc_arena(q_context* restrict ctx, size_t arena_size);

// Alocar memória na Arena (alinhada a 64 bytes)
// Retorna: ponteiro alinhado ou NULL em OOM
void* q_arena_alloc(q_context* restrict ctx, size_t size);

// Reset da Arena (zera head, poison em DEBUG)
void q_arena_reset(q_context* restrict ctx);

// Liberar toda a memória do contexto
void q_free_memory(q_context* restrict ctx);

// ============================================================================
// Error Handling API
// ============================================================================

// Convert error code to human-readable string
// Returns: Pointer to static string (do not free)
const char* q_strerror(q_error_code err);

// ============================================================================
// Mathematical Operations API (AVX2 Optimized)
// ============================================================================

// Dequantize a Q4_0 block to 32 floats using AVX2
// Input: block - pointer to q_block_q4_0 (20 bytes)
// Output: output - pointer to 32 floats (128 bytes, must be 32-byte aligned)
// Note: Internal implementation is static inline for MatMul integration
// This public wrapper is for testing purposes
void q_dequantize_q4_0_block_avx2_public(
    const q_block_q4_0* restrict block,
    float* restrict output
);

// GEMV Q4_F32: Matrix Q4_0 * Vector F32 -> Vector F32
// Critical operation for Llama-3 inference
// Preconditions:
// - weights: Q4_0 matrix [M, N], N must be multiple of 32
// - input: F32 vector [N], 32-byte aligned
// - output: F32 vector [M], 32-byte aligned
// Returns: Q_OK on success, negative q_error_code on validation failure
q_error_code q_gemv_q4_f32_avx2(
    const q_tensor* restrict weights,
    const float* restrict input,
    float* restrict output
);

// MatMul FP32: Matrix F32 * Matrix F32 -> Matrix F32
// Critical operation for attention (Q @ K^T, probs @ V) and LM Head projection
// Preconditions:
// - A: FP32 matrix [M, K], 32-byte aligned
// - B: FP32 matrix [K, N], 32-byte aligned
// - C: FP32 matrix [M, N], 32-byte aligned (output, may alias A or B)
// - ctx: Memory context with arena allocated (for temporary transposed B buffer)
// Returns: Q_OK on success, negative q_error_code on validation failure
q_error_code q_matmul_f32_avx2(
    const q_tensor* restrict A,
    const q_tensor* restrict B,
    q_tensor* C,
    q_context* restrict ctx
);

// Causal Masking FP32: Set upper triangular elements to mask_value
// Critical operation for attention (prevent future tokens from attending to past tokens)
// Preconditions:
// - scores: FP32 matrix [seq_len, seq_len], 32-byte aligned (modified in-place)
// - mask_value: float (typically -1e9f), value to set masked positions
// Returns: Q_OK on success, negative q_error_code on validation failure
q_error_code q_causal_mask_f32_avx2(
    q_tensor* scores,           // [seq_len, seq_len] (modified in-place)
    float mask_value            // Value to set masked positions
);

// Tensor Add FP32: output = a + b
// Critical operation for residual connections in Transformer blocks
// Preconditions:
// - a, b: FP32 vectors [N], 32-byte aligned, same shape, contiguous (nb[0] == N * sizeof(float))
// - output: FP32 vector [N], 32-byte aligned (may alias a or b for in-place operation)
// Note: NO restrict qualifiers to allow safe aliasing (output == a or output == b)
// Returns: Q_OK on success, negative q_error_code on validation failure
q_error_code q_add_f32_avx2(
    const q_tensor* a,     // [N] (NO restrict: output may alias)
    const q_tensor* b,     // [N] (NO restrict: output may alias)
    q_tensor* output        // [N] (may alias a or b)
);

// Element-wise Mul FP32: output = a * b
// Critical operation for SwiGLU activation (gate * up in MLP)
// Preconditions:
// - a, b: FP32 vectors [N], 32-byte aligned, same shape, contiguous (nb[0] == N * sizeof(float))
// - output: FP32 vector [N], 32-byte aligned (may alias a or b for in-place operation)
// Note: NO restrict qualifiers to allow safe aliasing (output == a or output == b)
// Returns: Q_OK on success, negative q_error_code on validation failure
q_error_code q_mul_f32_avx2(
    const q_tensor* a,     // [N] (NO restrict: output may alias)
    const q_tensor* b,     // [N] (NO restrict: output may alias)
    q_tensor* output        // [N] (may alias a or b)
);

// ============================================================================
// Normalization & Positional Encoding Operations (AVX2 Optimized)
// ============================================================================

// RMSNorm: Root Mean Square Normalization
// y = x * rsqrt(mean(x^2) + eps) * weight
// Preconditions:
// - x, weight, output: F32 vectors [N], 32-byte aligned
// - N must be multiple of 8
// Returns: Q_OK on success, negative q_error_code on validation failure
q_error_code q_rmsnorm_f32_avx2(
    const float* restrict x,
    const float* restrict weight,
    float* restrict output,
    uint32_t N,
    float eps
);

// RoPE: Rotary Positional Embedding
// Rotates pairs (x, y) by angle theta using pre-computed cos/sin tables
// Preconditions:
// - x, output: F32 vectors [N], 32-byte aligned, N must be even
// - cos, sin: F32 vectors [N/2], 32-byte aligned
// - N must be multiple of 8
// Returns: Q_OK on success, negative q_error_code on validation failure
q_error_code q_rope_f32_avx2(
    const float* restrict x,
    const float* restrict cos,
    const float* restrict sin,
    float* restrict output,
    uint32_t N
);

// SiLU: Sigmoid Linear Unit (Swish activation)
// f(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Preconditions:
// - x, output: F32 vectors [N], 32-byte aligned
// - N must be multiple of 8
// Returns: Q_OK on success, negative q_error_code on validation failure
q_error_code q_silu_f32_avx2(
    const float* restrict x,
    float* restrict output,
    uint32_t N
);

// Softmax: Stable probability distribution
// output[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
// Preconditions:
// - x, output: F32 vectors [N], 32-byte aligned
// - N must be multiple of 8
// Returns: Q_OK on success, negative q_error_code on validation failure
q_error_code q_softmax_f32_avx2(
    const float* restrict x,
    float* restrict output,
    uint32_t N
);

// ============================================================================
// Llama-3 Model API
// ============================================================================

// Build model graph from mmap'd .qorus file
// Creates tensor views pointing to data in mmap (zero-copy)
// Allocates llama_model and llama_layer structures in arena
// Returns: Q_OK on success, negative q_error_code on error
q_error_code llama_build_graph(
    q_context* restrict ctx,
    llama_model* restrict model
);

// Free model graph (frees arena allocations, but NOT mmap)
// Note: Does NOT free mmap'd weights (call q_free_memory for that)
void llama_free_graph(llama_model* restrict model);

// Forward pass through Llama-3 model
// Executes inference: tokens -> embeddings -> layers -> logits
// Preconditions:
// - model: Valid model structure (from llama_build_graph)
// - ctx: Memory context with KV cache and arena allocated
// - tokens: Input token IDs [seq_len], valid token IDs (0 <= tokens[i] < vocab_size)
// - seq_len: Current sequence length (0 < seq_len <= max_seq_len)
// - pos: Current position in sequence (0 <= pos < max_seq_len)
// - logits: Output buffer [vocab_size], 32-byte aligned
// Returns: Q_OK on success, negative q_error_code on validation failure
q_error_code llama_forward(
    llama_model* restrict model,
    q_context* restrict ctx,
    const uint32_t* restrict tokens,
    uint32_t seq_len,
    uint32_t pos,
    float* restrict logits
);

#endif // QORUS_H

