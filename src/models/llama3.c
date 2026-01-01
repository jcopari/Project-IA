#include "qorus.h"
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

// Helper macro to suppress unused result warning for write()
#define Q_WRITE_STDERR(msg, len) do { \
    ssize_t _q_write_result = write(2, msg, len); \
    (void)_q_write_result; \
} while (0)

// ============================================================================
// Correção 1: Estrutura de Scratchpad Reutilizável
// ============================================================================

// Estrutura de scratchpad reutilizável para uma camada
// Elimina alocação linear O(L) na arena
typedef struct {
    // Buffers principais
    float* attn_out;
    float* mlp_out;
    float* x_norm;
    float* x_norm_mlp;
    
    // Buffers de atenção
    float* q_buf;
    float* k_buf;
    float* v_buf;
    float* q_rope_buf;
    float* k_rope_buf;
    float* cos_buf;
    float* sin_buf;
    float* scores_buf;
    size_t scores_stride_floats;  // Stride alinhado para scores_buf (em floats, não bytes)
    float* q_heads;
    float* k_heads;
    float* v_heads;
    float* attn_head_buf;
    float* k_t_buf;
    
    // Buffers MLP
    float* gate_buf;
    float* up_buf;
    float* mul_buf;
    float* gate_silu;
} layer_scratchpad;

// FASE 3.2: Build model graph from mmap'd .qorus file
// Creates tensor views pointing to data in mmap (zero-copy)

// Helper: Check for overflow in size_t multiplication
// Returns true if multiplication would overflow, false otherwise
// Time Complexity: O(1)
static inline bool check_size_t_mult_overflow(size_t a, size_t b) {
    // Early return for zero operands (optimization)
    if (a == 0 || b == 0) {
        return false;  // 0 * x = 0, no overflow
    }
    // Check: a * b > SIZE_MAX  <=>  a > SIZE_MAX / b
    // This is safe because b > 0 (checked above)
    return a > SIZE_MAX / b;
}

// Helper: Calculate size of FP32 tensor with overflow check
// Returns 0 on overflow (invalid)
static size_t calculate_f32_size(uint32_t ne0, uint32_t ne1, uint32_t ne2, uint32_t ne3) {
    const size_t element_size = sizeof(float);
    
    // Check overflow step by step
    if (check_size_t_mult_overflow((size_t)ne0, (size_t)ne1)) return 0;
    size_t step1 = (size_t)ne0 * (size_t)ne1;
    
    if (check_size_t_mult_overflow(step1, (size_t)ne2)) return 0;
    size_t step2 = step1 * (size_t)ne2;
    
    if (check_size_t_mult_overflow(step2, (size_t)ne3)) return 0;
    size_t step3 = step2 * (size_t)ne3;
    
    if (check_size_t_mult_overflow(step3, element_size)) return 0;
    
    return step3 * element_size;
}

// Helper: Calculate size of Q4_0 tensor (must be block-aligned) with overflow check
// Returns 0 on overflow or invalid dimensions
static size_t calculate_q4_0_size(uint32_t ne0, uint32_t ne1) {
    // Q4_0: 32 values per block
    if (ne1 % 32 != 0) return 0;
    
    uint32_t blocks_per_row = ne1 / 32;
    const size_t block_size = sizeof(q_block_q4_0);
    
    if (check_size_t_mult_overflow((size_t)ne0, (size_t)blocks_per_row)) return 0;
    size_t step1 = (size_t)ne0 * (size_t)blocks_per_row;
    
    if (check_size_t_mult_overflow(step1, block_size)) return 0;
    
    return step1 * block_size;
}

// Helper: Create tensor view pointing to mmap data
// Implements strict bounds checking and overflow protection.
// NOTE: Strides follow Row-Major convention where nb[0] is largest stride (outermost dim)
//       and nb[3] is smallest stride (innermost dim, varies fastest)
static q_tensor* create_tensor_view(
    q_context* restrict ctx,
    void* data_ptr,
    uint32_t ne0, uint32_t ne1, uint32_t ne2, uint32_t ne3,
    q_dtype type,
    const char* name
) {
    // Security: Validate context pointer
    if (__builtin_expect(ctx == NULL, 0)) return NULL;
    
    // Security: Validate mmap is initialized
    if (__builtin_expect(ctx->weights_mmap == NULL, 0)) {
        return NULL;  // Mmap not initialized
    }
    
    // Security: Validate data pointer is within mmap range
    uintptr_t mmap_start = (uintptr_t)ctx->weights_mmap;
    uintptr_t mmap_end = mmap_start + ctx->weights_size;
    uintptr_t data_addr = (uintptr_t)data_ptr;
    
    if (data_addr < mmap_start || data_addr >= mmap_end) {
        return NULL;
    }
    
    // Security: Calculate tensor size and validate it fits within mmap
    size_t tensor_size = 0;
    if (type == Q_F32) {
        tensor_size = calculate_f32_size(ne0, ne1, ne2, ne3);
    } else if (type == Q_Q4_0) {
        tensor_size = calculate_q4_0_size(ne0, ne1);
    } else {
        return NULL;
    }
    
    if (tensor_size == 0) {
        return NULL; // Overflow or invalid dimensions
    }
    
    // Security: Validate that entire tensor fits within mmap bounds
    // Check wraparound: tensor_end < data_addr indicates wraparound
    uintptr_t tensor_end = data_addr + tensor_size;
    if (tensor_end > mmap_end || tensor_end < data_addr) {
        return NULL; // Tensor extends beyond mmap or wraparound occurred
    }
    
    // Allocate tensor view in arena
    q_tensor* tensor = (q_tensor*)q_arena_alloc(ctx, sizeof(q_tensor));
    if (tensor == NULL) {
        return NULL;
    }
    
    // CRITICAL FIX (Correção 4): Inicializar explicitamente, NÃO usar memset
    // Zero não pode ser valor válido (Q_TYPE_INVALID = 0)
    // Time Complexity: O(1) - inicialização explícita
    // Space Complexity: O(1) - no additional memory
    tensor->data = data_ptr;
    tensor->scales = NULL;
    tensor->ne[0] = ne0;
    tensor->ne[1] = ne1;
    tensor->ne[2] = ne2;
    tensor->ne[3] = ne3;
    tensor->type = type;  // CRÍTICO: definir ANTES de calcular strides
    
    // Calculate strides (Row-Major Convention)
    if (type == Q_F32) {
        size_t element_size = sizeof(float);
        
        // nb[3] = element_size (innermost dimension)
        tensor->nb[3] = element_size;
        
        // nb[2] = ne3 * element_size
        if (check_size_t_mult_overflow((size_t)ne3, element_size)) {
            return NULL;
        }
        tensor->nb[2] = (size_t)ne3 * element_size;
        
        // nb[1] = ne2 * nb[2]
        if (check_size_t_mult_overflow(tensor->nb[2], (size_t)ne2)) {
            return NULL;
        }
        tensor->nb[1] = tensor->nb[2] * (size_t)ne2;
        
        // nb[0] = ne1 * nb[1] (outermost dimension)
        if (check_size_t_mult_overflow(tensor->nb[1], (size_t)ne1)) {
            return NULL;
        }
        tensor->nb[0] = tensor->nb[1] * (size_t)ne1;
        
    } else if (type == Q_Q4_0) {
        // Q4_0 Layout: Blocked format
        uint32_t blocks_per_row = ne1 / 32;
        size_t block_size = sizeof(q_block_q4_0);
        
        // nb[0]: Stride for rows
        if (check_size_t_mult_overflow((size_t)blocks_per_row, block_size)) {
            return NULL;
        }
        tensor->nb[0] = (size_t)blocks_per_row * block_size;
        
        // nb[1]: Symbolic stride for blocks
        tensor->nb[1] = block_size;
        tensor->nb[2] = block_size;
        tensor->nb[3] = block_size;
    } else {
        return NULL;
    }
    
    strncpy(tensor->name, name, sizeof(tensor->name) - 1);
    tensor->name[sizeof(tensor->name) - 1] = '\0';
    
    // CRITICAL VALIDATION (Correção 4): Tipo deve corresponder ao esperado
    if (tensor->type != type) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: create_tensor_view: invalid type!\n");
        fprintf(stderr, "  Expected type: %d, Got type: %d\n", type, tensor->type);
        fprintf(stderr, "  Tensor pointer: %p\n", (void*)tensor);
        abort();
        #endif
        return NULL; // Fail-fast
    }
    
    return tensor;
}

q_error_code llama_build_graph(q_context* restrict ctx, q_llama_model* restrict model) {
    // Validate inputs
    if (ctx == NULL || model == NULL) {
        return Q_ERR_NULL_PTR;
    }
    
    if (ctx->weights_mmap == NULL || ctx->header == NULL) {
        return Q_ERR_NULL_PTR;
    }
    
    // Validate magic number
    if (ctx->header->magic != Q_MAGIC) {
        return Q_ERR_INVALID_MAGIC;
    }
    
    // Validate configuration
    if (ctx->header->n_layers == 0 || ctx->header->dim == 0 ||
        ctx->header->vocab_size == 0 || ctx->header->n_heads == 0 ||
        ctx->header->n_kv_heads == 0) {
        return Q_ERR_INVALID_CONFIG;
    }
    
    // Validate dim is multiple of 32 (required for Q4_0)
    if (ctx->header->dim % 32 != 0) {
        return Q_ERR_INVALID_CONFIG;
    }
    
    // Validate hidden_dim is multiple of 32 (required for Q4_0)
    if (ctx->header->hidden_dim % 32 != 0) {
        return Q_ERR_INVALID_CONFIG;
    }
    
    // Copy configuration from header
    model->config.vocab_size = ctx->header->vocab_size;
    model->config.dim = ctx->header->dim;
    model->config.hidden_dim = ctx->header->hidden_dim;
    model->config.n_layers = ctx->header->n_layers;
    model->config.n_heads = ctx->header->n_heads;
    model->config.n_kv_heads = ctx->header->n_kv_heads;
    model->config.max_seq_len = ctx->header->max_seq_len;
    model->config.rope_freq_base = ctx->header->rope_freq_base;
    
    // Read rms_norm_eps from header if available (version >= 2), otherwise use default
    // Llama 2 uses 1e-6, Llama 3 uses 1e-5
    // Default to 1e-5 (Llama-3) for better compatibility
    if (ctx->header->version >= 2 && ctx->header->rms_norm_eps > 0.0f) {
        model->config.rms_norm_eps = ctx->header->rms_norm_eps;
    } else {
        model->config.rms_norm_eps = 1e-5f;  // Default Llama-3 epsilon (more conservative)
    }
    
    // Set context pointer
    model->ctx = ctx;
    
    // Calculate head dimensions (n_heads already validated to be > 0)
    uint32_t head_dim = model->config.dim / model->config.n_heads;
    uint32_t kv_dim = model->config.n_kv_heads * head_dim;
    
    // Start offset after header (64 bytes)
    size_t offset = Q_HEADER_SIZE;
    
    // 1. Token embeddings [vocab_size, dim] (FP32)
    size_t token_embd_size = calculate_f32_size(model->config.vocab_size, model->config.dim, 1, 1);
    token_embd_size = Q_ALIGN_SIZE(token_embd_size);  // Align to 64 bytes
    
    if (offset + token_embd_size > ctx->weights_size) {
        return Q_ERR_INVALID_CONFIG;
    }
    
    model->token_embd = create_tensor_view(
        ctx,
        (uint8_t*)ctx->weights_mmap + offset,
        model->config.vocab_size, model->config.dim, 1, 1,
        Q_F32,
        "token_embd.weight"
    );
    if (model->token_embd == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    
    // CRITICAL VALIDATION: Verify type was set correctly immediately after creation
    // This catches any initialization bugs before they cause runtime errors
    if (model->token_embd->type != Q_F32) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: llama_build_graph: token_embd->type = %d, expected Q_F32 (%d)\n"
            "  token_embd pointer: %p\n",
            (int)model->token_embd->type, (int)Q_F32,
            (void*)model->token_embd);
        abort();
        #endif
        return Q_ERR_INVALID_DTYPE;
    }
    offset += token_embd_size;
    
    // 2. Output normalization [dim] (FP32)
    size_t output_norm_size = calculate_f32_size(model->config.dim, 1, 1, 1);
    output_norm_size = Q_ALIGN_SIZE(output_norm_size);
    
    if (offset + output_norm_size > ctx->weights_size) {
        return Q_ERR_INVALID_CONFIG;
    }
    
    model->output_norm = create_tensor_view(
        ctx,
        (uint8_t*)ctx->weights_mmap + offset,
        model->config.dim, 1, 1, 1,
        Q_F32,
        "output_norm.weight"
    );
    if (model->output_norm == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    offset += output_norm_size;
    
    // 3. Output projection [vocab_size, dim] (FP32)
    size_t output_size = calculate_f32_size(model->config.vocab_size, model->config.dim, 1, 1);
    output_size = Q_ALIGN_SIZE(output_size);
    
    if (offset + output_size > ctx->weights_size) {
        return Q_ERR_INVALID_CONFIG;
    }
    
    model->output = create_tensor_view(
        ctx,
        (uint8_t*)ctx->weights_mmap + offset,
        model->config.vocab_size, model->config.dim, 1, 1,
        Q_F32,
        "output.weight"
    );
    if (model->output == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    offset += output_size;
    
    // 4. Allocate layers array
    model->layers = (q_llama_layer*)q_arena_alloc(ctx, sizeof(q_llama_layer) * model->config.n_layers);
    if (model->layers == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    
    // 5. Build each layer
    for (uint32_t i = 0; i < model->config.n_layers; i++) {
        q_llama_layer* layer = &model->layers[i];
        memset(layer, 0, sizeof(q_llama_layer));
        layer->layer_idx = i;
        
        // Attention norm [dim] (FP32)
        size_t attn_norm_size = calculate_f32_size(model->config.dim, 1, 1, 1);
        attn_norm_size = Q_ALIGN_SIZE(attn_norm_size);
        
        if (offset + attn_norm_size > ctx->weights_size) {
            return Q_ERR_INVALID_CONFIG;
        }
        
        layer->attn_norm = create_tensor_view(
            ctx,
            (uint8_t*)ctx->weights_mmap + offset,
            model->config.dim, 1, 1, 1,
            Q_F32,
            "attn_norm.weight"
        );
        if (layer->attn_norm == NULL) {
            return Q_ERR_ARENA_OOM;
        }
        offset += attn_norm_size;
        
        // Q projection [dim, dim] (Q4_0)
        size_t wq_size = calculate_q4_0_size(model->config.dim, model->config.dim);
        wq_size = Q_ALIGN_SIZE(wq_size);
        
        if (wq_size == 0 || offset + wq_size > ctx->weights_size) {
            return Q_ERR_INVALID_CONFIG;
        }
        
        layer->wq = create_tensor_view(
            ctx,
            (uint8_t*)ctx->weights_mmap + offset,
            model->config.dim, model->config.dim, 1, 1,
            Q_Q4_0,
            "wq.weight"
        );
        if (layer->wq == NULL) {
            return Q_ERR_ARENA_OOM;
        }
        offset += wq_size;
        
        // K projection [dim, kv_dim] (Q4_0)
        size_t wk_size = calculate_q4_0_size(model->config.dim, kv_dim);
        wk_size = Q_ALIGN_SIZE(wk_size);
        
        if (wk_size == 0 || offset + wk_size > ctx->weights_size) {
            return Q_ERR_INVALID_CONFIG;
        }
        
        layer->wk = create_tensor_view(
            ctx,
            (uint8_t*)ctx->weights_mmap + offset,
            model->config.dim, kv_dim, 1, 1,
            Q_Q4_0,
            "wk.weight"
        );
        if (layer->wk == NULL) {
            return Q_ERR_ARENA_OOM;
        }
        offset += wk_size;
        
        // V projection [dim, kv_dim] (Q4_0)
        size_t wv_size = calculate_q4_0_size(model->config.dim, kv_dim);
        wv_size = Q_ALIGN_SIZE(wv_size);
        
        if (wv_size == 0 || offset + wv_size > ctx->weights_size) {
            return Q_ERR_INVALID_CONFIG;
        }
        
        layer->wv = create_tensor_view(
            ctx,
            (uint8_t*)ctx->weights_mmap + offset,
            model->config.dim, kv_dim, 1, 1,
            Q_Q4_0,
            "wv.weight"
        );
        if (layer->wv == NULL) {
            return Q_ERR_ARENA_OOM;
        }
        offset += wv_size;
        
        // Output projection [dim, dim] (Q4_0)
        size_t wo_size = calculate_q4_0_size(model->config.dim, model->config.dim);
        wo_size = Q_ALIGN_SIZE(wo_size);
        
        if (wo_size == 0 || offset + wo_size > ctx->weights_size) {
            return Q_ERR_INVALID_CONFIG;
        }
        
        layer->wo = create_tensor_view(
            ctx,
            (uint8_t*)ctx->weights_mmap + offset,
            model->config.dim, model->config.dim, 1, 1,
            Q_Q4_0,
            "wo.weight"
        );
        if (layer->wo == NULL) {
            return Q_ERR_ARENA_OOM;
        }
        offset += wo_size;
        
        // FFN norm [dim] (FP32)
        size_t ffn_norm_size = calculate_f32_size(model->config.dim, 1, 1, 1);
        ffn_norm_size = Q_ALIGN_SIZE(ffn_norm_size);
        
        if (offset + ffn_norm_size > ctx->weights_size) {
            return Q_ERR_INVALID_CONFIG;
        }
        
        layer->ffn_norm = create_tensor_view(
            ctx,
            (uint8_t*)ctx->weights_mmap + offset,
            model->config.dim, 1, 1, 1,
            Q_F32,
            "ffn_norm.weight"
        );
        if (layer->ffn_norm == NULL) {
            return Q_ERR_ARENA_OOM;
        }
        offset += ffn_norm_size;
        
        // Gate projection [dim, hidden_dim] (Q4_0)
        size_t w_gate_size = calculate_q4_0_size(model->config.dim, model->config.hidden_dim);
        w_gate_size = Q_ALIGN_SIZE(w_gate_size);
        
        if (w_gate_size == 0 || offset + w_gate_size > ctx->weights_size) {
            return Q_ERR_INVALID_CONFIG;
        }
        
        layer->w_gate = create_tensor_view(
            ctx,
            (uint8_t*)ctx->weights_mmap + offset,
            model->config.dim, model->config.hidden_dim, 1, 1,
            Q_Q4_0,
            "w_gate.weight"
        );
        if (layer->w_gate == NULL) {
            return Q_ERR_ARENA_OOM;
        }
        offset += w_gate_size;
        
        // Up projection [dim, hidden_dim] (Q4_0)
        size_t w_up_size = calculate_q4_0_size(model->config.dim, model->config.hidden_dim);
        w_up_size = Q_ALIGN_SIZE(w_up_size);
        
        if (w_up_size == 0 || offset + w_up_size > ctx->weights_size) {
            return Q_ERR_INVALID_CONFIG;
        }
        
        layer->w_up = create_tensor_view(
            ctx,
            (uint8_t*)ctx->weights_mmap + offset,
            model->config.dim, model->config.hidden_dim, 1, 1,
            Q_Q4_0,
            "w_up.weight"
        );
        if (layer->w_up == NULL) {
            return Q_ERR_ARENA_OOM;
        }
        offset += w_up_size;
        
        // Down projection [hidden_dim, dim] (Q4_0)
        size_t w_down_size = calculate_q4_0_size(model->config.hidden_dim, model->config.dim);
        w_down_size = Q_ALIGN_SIZE(w_down_size);
        
        if (w_down_size == 0 || offset + w_down_size > ctx->weights_size) {
            return Q_ERR_INVALID_CONFIG;
        }
        
        layer->w_down = create_tensor_view(
            ctx,
            (uint8_t*)ctx->weights_mmap + offset,
            model->config.hidden_dim, model->config.dim, 1, 1,
            Q_Q4_0,
            "w_down.weight"
        );
        if (layer->w_down == NULL) {
            return Q_ERR_ARENA_OOM;
        }
        offset += w_down_size;
    }
    
    // ============================================================================
    // Correção 2: Pré-calcular frequências RoPE (elimina powf do hot path)
    // ============================================================================
    // head_dim já foi calculado acima, reutilizar
    uint32_t num_pairs = head_dim / 2;
    
    // Pré-calcular frequências base (invariantes do modelo)
    model->rope_freqs = (float*)q_arena_alloc(ctx, num_pairs * sizeof(float));
    if (model->rope_freqs == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    
    for (uint32_t i = 0; i < num_pairs; i++) {
        float freq_exp = -2.0f * (float)i / (float)head_dim;
        model->rope_freqs[i] = powf(model->config.rope_freq_base, freq_exp);
    }
    
    // Opcional: Pré-calcular cache completo (trade-off memória vs velocidade)
    model->rope_cache_enabled = false;
    model->rope_cos_cache = NULL;
    model->rope_sin_cache = NULL;
    
    if (model->config.max_seq_len <= 8192) { // Limite razoável para cache
        size_t cache_size = (size_t)model->config.max_seq_len * head_dim * sizeof(float);
        model->rope_cos_cache = (float*)q_arena_alloc(ctx, cache_size);
        model->rope_sin_cache = (float*)q_arena_alloc(ctx, cache_size);
        
        if (model->rope_cos_cache != NULL && model->rope_sin_cache != NULL) {
            for (uint32_t pos = 0; pos < model->config.max_seq_len; pos++) {
                for (uint32_t i = 0; i < num_pairs; i++) {
                    float theta = model->rope_freqs[i] * (float)pos; // SEM powf!
                    float c = cosf(theta);
                    float s = sinf(theta);
                    model->rope_cos_cache[pos * head_dim + i * 2] = c;
                    model->rope_cos_cache[pos * head_dim + i * 2 + 1] = c;
                    model->rope_sin_cache[pos * head_dim + i * 2] = s;
                    model->rope_sin_cache[pos * head_dim + i * 2 + 1] = s;
                }
            }
            model->rope_cache_enabled = true;
        }
    }
    
    // CORREÇÃO 5: Definir watermark após todas as alocações do modelo
    // Tudo antes deste ponto é estrutura do modelo (persistente)
    // Tudo depois será scratchpad (transiente, resetado a cada inferência)
    ctx->scratch_base_offset = ctx->scratch_head;
    
    return Q_OK;
}

void llama_free_graph(q_llama_model* restrict model) {
    if (model == NULL) {
        return;
    }
    
    // Reset all pointers
    // Note: We don't free the memory because it's in the arena
    // The arena will be reset/cleared when q_arena_reset() is called
    model->token_embd = NULL;
    model->output_norm = NULL;
    model->output = NULL;
    model->layers = NULL;
    model->ctx = NULL;
    
    // Clear configuration
    memset(&model->config, 0, sizeof(q_llama_config));
}

// ============================================================================
// FASE 3.3: Forward Pass Implementation
// ============================================================================

// Helper: Get KV cache pointer for a specific layer/head/position
// Layout: [n_layers, n_kv_heads, max_seq_len, head_dim]
// Returns NULL if invalid parameters
static float* get_kv_cache_ptr(
    q_context* restrict ctx,
    const q_llama_config* restrict config,
    uint32_t layer_idx,
    uint32_t kv_head_idx,
    uint32_t pos,
    bool is_key  // true for K, false for V
) {
    if (ctx->kv_buffer == NULL) {
        return NULL;
    }
    
    if (layer_idx >= config->n_layers ||
        kv_head_idx >= config->n_kv_heads ||
        pos >= config->max_seq_len) {
        return NULL;
    }
    
    uint32_t head_dim = config->dim / config->n_heads;
    
    // Calculate offset: layer_offset + head_offset + pos_offset + key/value_offset
    size_t layer_stride = (size_t)config->n_kv_heads * 
                          (size_t)config->max_seq_len * 
                          (size_t)head_dim * 
                          sizeof(float) * 2; // *2 for K and V
    
    size_t head_stride = (size_t)config->max_seq_len * 
                         (size_t)head_dim * 
                         sizeof(float) * 2; // *2 for K and V
    
    size_t pos_stride = (size_t)head_dim * sizeof(float) * 2; // *2 for K and V
    
    size_t kv_offset = is_key ? 0 : ((size_t)head_dim * sizeof(float));
    
    size_t offset = (size_t)layer_idx * layer_stride +
                    (size_t)kv_head_idx * head_stride +
                    (size_t)pos * pos_stride +
                    kv_offset;
    
    return (float*)((uint8_t*)ctx->kv_buffer + offset);
}

// ============================================================================
// Correção 1: Funções auxiliares do Scratchpad
// ============================================================================

// Calcular tamanho máximo necessário para scratchpad
static size_t calculate_layer_scratchpad_size(const q_llama_config* config, uint32_t seq_len) {
    uint32_t dim = config->dim;
    uint32_t hidden_dim = config->hidden_dim;
    uint32_t head_dim = dim / config->n_heads;
    uint32_t n_heads = config->n_heads;
    uint32_t n_kv_heads = config->n_kv_heads;
    
    size_t buf_size = Q_ALIGN_SIZE((size_t)seq_len * (size_t)dim * sizeof(float));
    size_t hidden_size = Q_ALIGN_SIZE((size_t)seq_len * (size_t)hidden_dim * sizeof(float));
    size_t head_dim_size = Q_ALIGN_SIZE((size_t)head_dim * sizeof(float));
    // CORREÇÃO 1: Stride alinhado para scores_buf - cada linha está alinhada a 32 bytes
    size_t row_stride_floats = Q_ALIGN_SIZE(seq_len * sizeof(float)) / sizeof(float);
    size_t scores_size = row_stride_floats * seq_len * sizeof(float);
    size_t q_per_head_size = Q_ALIGN_SIZE((size_t)seq_len * head_dim * sizeof(float));
    size_t kv_dim = (size_t)n_kv_heads * head_dim;
    size_t kv_buf_size = Q_ALIGN_SIZE((size_t)seq_len * kv_dim * sizeof(float));
    
    return buf_size * 4 +                    // attn_out, mlp_out, x_norm, x_norm_mlp
           buf_size +                         // q_buf
           kv_buf_size * 2 +                  // k_buf, v_buf
           buf_size * 2 +                     // q_rope_buf, k_rope_buf
           head_dim_size * 2 +                // cos_buf, sin_buf
           scores_size +                      // scores_buf (com stride alinhado)
           q_per_head_size * n_heads +        // q_heads
           q_per_head_size * n_kv_heads * 2 + // k_heads, v_heads
           q_per_head_size +                  // attn_head_buf
           q_per_head_size +                  // k_t_buf
           hidden_size * 4;                   // gate_buf, up_buf, mul_buf, gate_silu
}

// Inicializar scratchpad a partir de bloco de memória contíguo
static void init_layer_scratchpad(
    layer_scratchpad* scratch,
    uint8_t* mem_base,
    const q_llama_config* config,
    uint32_t seq_len
) {
    uint32_t dim = config->dim;
    uint32_t hidden_dim = config->hidden_dim;
    uint32_t head_dim = dim / config->n_heads;
    uint32_t n_heads = config->n_heads;
    uint32_t n_kv_heads = config->n_kv_heads;
    
    size_t offset = 0;
    size_t buf_size = Q_ALIGN_SIZE((size_t)seq_len * (size_t)dim * sizeof(float));
    size_t hidden_size = Q_ALIGN_SIZE((size_t)seq_len * (size_t)hidden_dim * sizeof(float));
    size_t head_dim_size = Q_ALIGN_SIZE((size_t)head_dim * sizeof(float));
    // CORREÇÃO 1: Stride alinhado para scores_buf
    size_t row_stride_floats = Q_ALIGN_SIZE(seq_len * sizeof(float)) / sizeof(float);
    size_t scores_size = row_stride_floats * seq_len * sizeof(float);
    size_t q_per_head_size = Q_ALIGN_SIZE((size_t)seq_len * head_dim * sizeof(float));
    size_t kv_dim = (size_t)n_kv_heads * head_dim;
    size_t kv_buf_size = Q_ALIGN_SIZE((size_t)seq_len * kv_dim * sizeof(float));
    
    scratch->attn_out = (float*)(mem_base + offset);
    offset += buf_size;
    
    scratch->mlp_out = (float*)(mem_base + offset);
    offset += buf_size;
    
    scratch->x_norm = (float*)(mem_base + offset);
    offset += buf_size;
    
    scratch->x_norm_mlp = (float*)(mem_base + offset);
    offset += buf_size;
    
    scratch->q_buf = (float*)(mem_base + offset);
    offset += buf_size;
    
    scratch->k_buf = (float*)(mem_base + offset);
    offset += kv_buf_size;
    
    scratch->v_buf = (float*)(mem_base + offset);
    offset += kv_buf_size;
    
    scratch->q_rope_buf = (float*)(mem_base + offset);
    offset += buf_size;
    
    scratch->k_rope_buf = (float*)(mem_base + offset);
    offset += buf_size;
    
    scratch->cos_buf = (float*)(mem_base + offset);
    offset += head_dim_size;
    
    scratch->sin_buf = (float*)(mem_base + offset);
    offset += head_dim_size;
    
    scratch->scores_buf = (float*)(mem_base + offset);
    scratch->scores_stride_floats = row_stride_floats;  // Armazenar stride
    offset += scores_size;
    
    scratch->q_heads = (float*)(mem_base + offset);
    offset += q_per_head_size * n_heads;
    
    scratch->k_heads = (float*)(mem_base + offset);
    offset += q_per_head_size * n_kv_heads;
    
    scratch->v_heads = (float*)(mem_base + offset);
    offset += q_per_head_size * n_kv_heads;
    
    scratch->attn_head_buf = (float*)(mem_base + offset);
    offset += q_per_head_size;
    
    scratch->k_t_buf = (float*)(mem_base + offset);
    offset += q_per_head_size;
    
    scratch->gate_buf = (float*)(mem_base + offset);
    offset += hidden_size;
    
    scratch->up_buf = (float*)(mem_base + offset);
    offset += hidden_size;
    
    scratch->mul_buf = (float*)(mem_base + offset);
    offset += hidden_size;
    
    scratch->gate_silu = (float*)(mem_base + offset);
    // offset += hidden_size; // Não necessário, último buffer
}

// ============================================================================
// Correção 2: Refatorar generate_rope_cos_sin para usar pré-cálculo
// ============================================================================

// Helper: Generate RoPE cos/sin tables for a specific position
// CORRIGIDO: Usa frequências pré-calculadas (elimina powf do hot path)
// Output: cos_buf[head_dim], sin_buf[head_dim] (duplicated layout: [c0,c0,c1,c1,...])
static q_error_code generate_rope_cos_sin(
    const q_llama_model* restrict model,  // NOVO: recebe model completo
    uint32_t head_dim,
    uint32_t pos,
    float* restrict cos_buf,  // Output [head_dim], 32-byte aligned
    float* restrict sin_buf   // Output [head_dim], 32-byte aligned
) {
    Q_VALIDATE_PTR_OR_RETURN(model, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(cos_buf, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(sin_buf, Q_ERR_INVALID_ARG);
    Q_VALIDATE_NONZERO_OR_RETURN(head_dim, Q_ERR_INVALID_SIZE);
    Q_VALIDATE_MULTIPLE_OR_RETURN(head_dim, 2, Q_ERR_INVALID_SIZE); // Must be even
    
    // Se cache completo disponível, apenas lookup (O(1))
    if (model->rope_cache_enabled && pos < model->config.max_seq_len) {
        memcpy(cos_buf, model->rope_cos_cache + pos * head_dim, head_dim * sizeof(float));
        memcpy(sin_buf, model->rope_sin_cache + pos * head_dim, head_dim * sizeof(float));
        return Q_OK;
    }
    
    // Caso contrário, calcular apenas multiplicação escalar (SEM powf!)
    const uint32_t num_pairs = head_dim / 2;
    
    for (uint32_t i = 0; i < num_pairs; i++) {
        float theta = model->rope_freqs[i] * (float)pos; // SEM powf!
        float c = cosf(theta);
        float s = sinf(theta);
        
        // Duplicate for AVX2 layout: [c0, c0, c1, c1, ...]
        cos_buf[i * 2] = c;
        cos_buf[i * 2 + 1] = c;
        sin_buf[i * 2] = s;
        sin_buf[i * 2 + 1] = s;
    }
    
    return Q_OK;
}

// Helper: Token embedding lookup
// Copies embeddings for tokens into output buffer
static q_error_code token_embedding_lookup(
    const q_tensor* restrict token_embd,
    const uint32_t* restrict tokens,
    uint32_t seq_len,
    float* restrict output  // [seq_len, dim]
) {
    Q_VALIDATE_PTR_OR_RETURN(token_embd, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(tokens, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(output, Q_ERR_INVALID_ARG);
    Q_VALIDATE_NONZERO_OR_RETURN(seq_len, Q_ERR_INVALID_SIZE);
    
    // CRITICAL VALIDATION: Verify type is Q_F32
    // This validation is necessary because arena might be reset externally
    // or memory corruption might occur
    if (token_embd->type != Q_F32) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: token_embedding_lookup: token_embd->type = %d, expected Q_F32 (%d)\n",
                (int)token_embd->type, (int)Q_F32);
        abort();
        #endif
        return Q_ERR_INVALID_DTYPE;
    }
    
    uint32_t vocab_size = token_embd->ne[0];
    uint32_t dim = token_embd->ne[1];
    
    const float* embd_data = (const float*)token_embd->data;
    
    // Copy embeddings for each token
    for (uint32_t i = 0; i < seq_len; i++) {
        if (tokens[i] >= vocab_size) {
            return Q_ERR_INVALID_ARG;  // Invalid token ID
        }
        
        // Copy embedding: embd_data[tokens[i] * dim : (tokens[i] + 1) * dim]
        const float* src = embd_data + (size_t)tokens[i] * dim;
        float* dst = output + (size_t)i * dim;
        memcpy(dst, src, dim * sizeof(float));
    }
    
    return Q_OK;
}

// Helper: Single layer forward pass
// Implements: Attention block + MLP block with residuals
// CORRIGIDO: Usa scratchpad reutilizável (Correção 1)
static q_error_code llama_layer_forward(
    q_llama_layer* restrict layer,
    q_context* restrict ctx,
    const q_llama_model* restrict model,  // NOVO: necessário para RoPE pré-calculado
    const q_llama_config* restrict config,
    const float* restrict x,           // Input [seq_len, dim]
    float* restrict output,            // Output [seq_len, dim]
    uint32_t layer_idx,
    uint32_t seq_len,
    uint32_t pos,
    layer_scratchpad* restrict scratch  // NOVO: scratchpad reutilizável
);

// Helper: Attention forward pass with GQA support
// CORRIGIDO: Usa scratchpad reutilizável (Correção 1)
static q_error_code llama_attention_forward(
    q_llama_layer* restrict layer,
    q_context* restrict ctx,
    const q_llama_model* restrict model,  // NOVO: necessário para RoPE pré-calculado
    const q_llama_config* restrict config,
    const float* restrict x,           // Input [seq_len, dim]
    float* restrict output,            // Output [seq_len, dim]
    uint32_t layer_idx,
    uint32_t seq_len,
    uint32_t pos,
    layer_scratchpad* restrict scratch  // NOVO: scratchpad reutilizável
) {
    uint32_t dim = config->dim;
    uint32_t n_heads = config->n_heads;
    uint32_t n_kv_heads = config->n_kv_heads;
    uint32_t head_dim = dim / n_heads;
    
    // CORREÇÃO 1: Usar scratchpad em vez de q_arena_alloc
    // REMOVIDO: Todas as alocações q_arena_alloc
    // USAR: scratch->x_norm, scratch->q_buf, etc.
    
    // Pre-attention RMSNorm: x -> x_norm
    q_error_code ret = q_rmsnorm_f32_avx2(x, (const float*)layer->attn_norm->data, scratch->x_norm, dim, config->rms_norm_eps);
    if (ret != Q_OK) return ret;
    
    // Q/K/V projections using GEMV (Q4_0 weights)
    // CRITICAL FIX: Use q_gemv_q4_f32_avx2 for Q4_0 weights (not q_matmul_f32_avx2)
    // For each row of x_norm, compute GEMV: output_row = x_norm_row @ weights^T
    // This is equivalent to MatMul but optimized for Q4_0 weights
    
    // Q projection: x_norm @ wq -> q_buf [seq_len, dim]
    
    for (uint32_t i = 0; i < seq_len; i++) {
        const float* x_row = scratch->x_norm + (size_t)i * dim;
        float* q_row = scratch->q_buf + (size_t)i * dim;
        ret = q_gemv_q4_f32_avx2(layer->wq, x_row, q_row);
        if (ret != Q_OK) {
            #ifdef DEBUG
            fprintf(stderr, "ERROR: Q projection failed at row %u: ret=%d\n", i, ret);
            abort();
            #endif
            return ret;
        }
    }
    
    // K projection: x_norm @ wk -> k_buf [seq_len, n_kv_heads * head_dim]
    uint32_t kv_dim = n_kv_heads * head_dim;
    // CRITICAL: q_gemv_q4_f32_avx2 requires input size (dim) to be multiple of 32
    // dim is validated to be multiple of 32 in llama_build_graph, so this should be OK
    
    for (uint32_t i = 0; i < seq_len; i++) {
        const float* x_row = scratch->x_norm + (size_t)i * dim;
        float* k_row = scratch->k_buf + (size_t)i * kv_dim;
        ret = q_gemv_q4_f32_avx2(layer->wk, x_row, k_row);
        if (ret != Q_OK) {
            #ifdef DEBUG
            fprintf(stderr, "ERROR: K projection failed at row %u: ret=%d, dim=%u, kv_dim=%u\n",
                i, ret, dim, kv_dim);
            abort();
            #endif
            return ret;
        }
    }
    
    // V projection: x_norm @ wv -> v_buf [seq_len, n_kv_heads * head_dim]
    for (uint32_t i = 0; i < seq_len; i++) {
        const float* x_row = scratch->x_norm + (size_t)i * dim;
        float* v_row = scratch->v_buf + (size_t)i * kv_dim;
        ret = q_gemv_q4_f32_avx2(layer->wv, x_row, v_row);
        if (ret != Q_OK) {
            #ifdef DEBUG
            fprintf(stderr, "ERROR: V projection failed at row %u: ret=%d, dim=%u, kv_dim=%u\n",
                i, ret, dim, kv_dim);
            abort();
            #endif
            return ret;
        }
    }
    
    // CORREÇÃO 1: Buffers já alocados no scratchpad
    // REMOVIDO: Todas as alocações q_arena_alloc
    // USAR: scratch->cos_buf, scratch->sin_buf, scratch->q_rope_buf, etc.
    
    // Apply RoPE to Q and K (per head, per token)
    // Q: [seq_len, n_heads, head_dim] -> reshape to [seq_len * n_heads, head_dim]
    // K: [seq_len, n_kv_heads, head_dim] -> reshape to [seq_len * n_kv_heads, head_dim]
    for (uint32_t t = 0; t < seq_len; t++) {
        uint32_t token_pos = pos + t;  // Absolute position in sequence
        
        // CORREÇÃO 2: Generate RoPE cos/sin usando pré-cálculo
        ret = generate_rope_cos_sin(model, head_dim, token_pos, scratch->cos_buf, scratch->sin_buf);
        if (ret != Q_OK) return ret;
        
        // Apply RoPE to each Q head
        for (uint32_t h = 0; h < n_heads; h++) {
            const float* q_head = scratch->q_buf + (size_t)t * dim + (size_t)h * head_dim;
            float* q_head_out = scratch->q_rope_buf + (size_t)t * dim + (size_t)h * head_dim;
            
            ret = q_rope_f32_avx2(q_head, scratch->cos_buf, scratch->sin_buf, q_head_out, head_dim);
            if (ret != Q_OK) return ret;
        }
        
        // Apply RoPE to each K head (KV heads only)
        for (uint32_t h = 0; h < n_kv_heads; h++) {
            const float* k_head = scratch->k_buf + (size_t)t * (n_kv_heads * head_dim) + (size_t)h * head_dim;
            float* k_head_out = scratch->k_rope_buf + (size_t)t * (n_kv_heads * head_dim) + (size_t)h * head_dim;
            
            ret = q_rope_f32_avx2(k_head, scratch->cos_buf, scratch->sin_buf, k_head_out, head_dim);
            if (ret != Q_OK) return ret;
        }
    }
    
    // Update KV cache at position pos (store K and V for all tokens in sequence)
    for (uint32_t t = 0; t < seq_len; t++) {
        uint32_t cache_pos = pos + t;
        if (cache_pos >= config->max_seq_len) {
            return Q_ERR_INVALID_ARG;  // Position out of bounds
        }
        
        for (uint32_t h = 0; h < n_kv_heads; h++) {
            // Store K
            float* k_cache = get_kv_cache_ptr(ctx, config, layer_idx, h, cache_pos, true);
            if (k_cache == NULL) return Q_ERR_INVALID_ARG;
            
            const float* k_src = scratch->k_rope_buf + (size_t)t * (n_kv_heads * head_dim) + (size_t)h * head_dim;
            memcpy(k_cache, k_src, head_dim * sizeof(float));
            
            // Store V
            float* v_cache = get_kv_cache_ptr(ctx, config, layer_idx, h, cache_pos, false);
            if (v_cache == NULL) return Q_ERR_INVALID_ARG;
            
            const float* v_src = scratch->v_buf + (size_t)t * (n_kv_heads * head_dim) + (size_t)h * head_dim;
            memcpy(v_cache, v_src, head_dim * sizeof(float));
        }
    }
    
    // CORREÇÃO 1: Buffers já alocados no scratchpad
    // Reshape Q: [seq_len, dim] -> [seq_len, n_heads, head_dim]
    for (uint32_t t = 0; t < seq_len; t++) {
        for (uint32_t h = 0; h < n_heads; h++) {
            const float* q_src = scratch->q_rope_buf + (size_t)t * dim + (size_t)h * head_dim;
            float* q_dst = scratch->q_heads + (size_t)h * (seq_len * head_dim) + (size_t)t * head_dim;
            memcpy(q_dst, q_src, head_dim * sizeof(float));
        }
    }
    
    // Reshape K/V: [seq_len, n_kv_heads * head_dim] -> [seq_len, n_kv_heads, head_dim]
    for (uint32_t t = 0; t < seq_len; t++) {
        for (uint32_t h = 0; h < n_kv_heads; h++) {
            const float* k_src = scratch->k_rope_buf + (size_t)t * (n_kv_heads * head_dim) + (size_t)h * head_dim;
            float* k_dst = scratch->k_heads + (size_t)h * (seq_len * head_dim) + (size_t)t * head_dim;
            memcpy(k_dst, k_src, head_dim * sizeof(float));
            
            const float* v_src = scratch->v_buf + (size_t)t * (n_kv_heads * head_dim) + (size_t)h * head_dim;
            float* v_dst = scratch->v_heads + (size_t)h * (seq_len * head_dim) + (size_t)t * head_dim;
            memcpy(v_dst, v_src, head_dim * sizeof(float));
        }
    }
    
    // CRITICAL: Ensure k_t_buf is properly aligned for AVX2 (32-byte alignment)
    // Q_ALIGN_SIZE already aligns to 64 bytes, but verify explicitly
    if (((uintptr_t)scratch->k_t_buf % 32) != 0) {
        // Always print debug info (not just in DEBUG mode) to diagnose alignment issues
        fprintf(stderr, "ERROR: k_t_buf not 32-byte aligned: %p (offset: %zu)\n", 
                (void*)scratch->k_t_buf, (uintptr_t)scratch->k_t_buf % 32);
        fprintf(stderr, "  k_t_buf addr=%zu, k_t_buf %% 32 = %zu\n", 
                (uintptr_t)scratch->k_t_buf, (uintptr_t)scratch->k_t_buf % 32);
        #ifdef DEBUG
        abort();
        #endif
        return Q_ERR_MISALIGNED;
    }
    
    // OPTIMIZATION: Track which KV heads have been transposed
    // Since multiple query heads share the same KV head (GQA), we transpose
    // each KV head once when first used, then reuse for subsequent query heads
    uint32_t last_transposed_kv_head = UINT32_MAX;  // Invalid index
    
    // Process each query head
    float scale = 1.0f / sqrtf((float)head_dim);
    
    for (uint32_t qh = 0; qh < n_heads; qh++) {
        // Determine which KV head to use (GQA: multiple Q heads share KV head)
        uint32_t kv_head_idx = qh / (n_heads / n_kv_heads);
        
        // OPTIMIZATION: Transpose K only if this KV head hasn't been transposed yet
        // This avoids redundant transposition for query heads that share the same KV head
        if (kv_head_idx != last_transposed_kv_head) {
            const float* k_head_data = scratch->k_heads + (size_t)kv_head_idx * (seq_len * head_dim);
            // CORREÇÃO 3: Transposição tiled para melhor cache locality
            // Transpose K: [seq_len, head_dim] -> [head_dim, seq_len]
            #define TRANSPOSE_TILE_SIZE 32
            for (uint32_t ii = 0; ii < seq_len; ii += TRANSPOSE_TILE_SIZE) {
                uint32_t i_end = (ii + TRANSPOSE_TILE_SIZE < seq_len) ? ii + TRANSPOSE_TILE_SIZE : seq_len;
                for (uint32_t jj = 0; jj < head_dim; jj += TRANSPOSE_TILE_SIZE) {
                    uint32_t j_end = (jj + TRANSPOSE_TILE_SIZE < head_dim) ? jj + TRANSPOSE_TILE_SIZE : head_dim;
                    for (uint32_t i = ii; i < i_end; i++) {
                        for (uint32_t j = jj; j < j_end; j++) {
                            scratch->k_t_buf[j * seq_len + i] = k_head_data[i * head_dim + j];
                        }
                    }
                }
            }
            last_transposed_kv_head = kv_head_idx;
        }
        
        // Extract Q head: [seq_len, head_dim]
        q_tensor q_head_tensor = {
            .data = (void*)(scratch->q_heads + (size_t)qh * (seq_len * head_dim)),
            .ne = {seq_len, head_dim, 1, 1},
            .nb = {head_dim * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
            .type = Q_F32
        };
        
        // Compute scores: Q @ K^T -> [seq_len, seq_len]
        // OPTIMIZATION: Use pre-transposed K from k_t_buf (transposed once per KV head)
        q_tensor k_t_tensor = {
            .data = (void*)scratch->k_t_buf,
            .ne = {head_dim, seq_len, 1, 1},
            .nb = {seq_len * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
            .type = Q_F32
        };
        
        q_tensor scores_tensor = {
            .data = (void*)scratch->scores_buf,
            .ne = {seq_len, seq_len, 1, 1},
            .nb = {scratch->scores_stride_floats * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
            .type = Q_F32
        };
        
        // CRITICAL VALIDATION: Verify dimensions match before MatMul
        // A[seq_len, head_dim] @ B[head_dim, seq_len] -> C[seq_len, seq_len]
        if (q_head_tensor.ne[0] != seq_len || q_head_tensor.ne[1] != head_dim) {
            #ifdef DEBUG
            fprintf(stderr, "ERROR: q_head_tensor dimensions mismatch: expected [%u,%u], got [%u,%u]\n",
                seq_len, head_dim, q_head_tensor.ne[0], q_head_tensor.ne[1]);
            abort();
            #endif
            return Q_ERR_INVALID_SIZE;
        }
        if (k_t_tensor.ne[0] != head_dim || k_t_tensor.ne[1] != seq_len) {
            #ifdef DEBUG
            fprintf(stderr, "ERROR: k_t_tensor dimensions mismatch: expected [%u,%u], got [%u,%u]\n",
                head_dim, seq_len, k_t_tensor.ne[0], k_t_tensor.ne[1]);
            abort();
            #endif
            return Q_ERR_INVALID_SIZE;
        }
        if (scores_tensor.ne[0] != seq_len || scores_tensor.ne[1] != seq_len) {
            #ifdef DEBUG
            fprintf(stderr, "ERROR: scores_tensor dimensions mismatch: expected [%u,%u], got [%u,%u]\n",
                seq_len, seq_len, scores_tensor.ne[0], scores_tensor.ne[1]);
            abort();
            #endif
            return Q_ERR_INVALID_SIZE;
        }
        
        ret = q_matmul_f32_avx2(&q_head_tensor, &k_t_tensor, &scores_tensor, ctx);
        if (ret != Q_OK) {
            #ifdef DEBUG
            fprintf(stderr, "ERROR: Attention scores MatMul failed: ret=%d\n", ret);
            abort();
            #endif
            return ret;
        }
        
        // Scale scores: scores *= 1/sqrt(head_dim)
        for (uint32_t i = 0; i < seq_len * seq_len; i++) {
            scratch->scores_buf[i] *= scale;
        }
        
        // Apply causal mask
        ret = q_causal_mask_f32_avx2(&scores_tensor, -1e9f);
        if (ret != Q_OK) {
            #ifdef DEBUG
            fprintf(stderr, "ERROR: Causal mask failed: ret=%d\n", ret);
            abort();
            #endif
            return ret;
        }
        
        // Softmax: probs = softmax(scores) per row
        // CORREÇÃO 1: scores_buf já tem stride alinhado, cada linha está alinhada a 32 bytes
        // Não precisamos mais de memcpy - podemos chamar softmax diretamente in-place
        float* probs_buf = scratch->scores_buf;  // Reuse scores buffer
        
        for (uint32_t i = 0; i < seq_len; i++) {
            // Cada linha está alinhada devido ao stride alinhado
            float* row_ptr = &scratch->scores_buf[i * scratch->scores_stride_floats];
            
            // Call softmax in-place (input e output são o mesmo buffer)
            // Usar cast para evitar warning de aliasing (é seguro aqui pois é in-place)
            ret = q_softmax_f32_avx2(row_ptr, (float* __restrict__)row_ptr, seq_len);
            if (ret != Q_OK) {
                #ifdef DEBUG
                fprintf(stderr, "ERROR: Softmax failed at row %u: ret=%d\n", i, ret);
                abort();
                #endif
                return ret;
            }
        }
        
        // Attention output: probs @ V -> [seq_len, head_dim]
        // CORREÇÃO 1: Usar stride alinhado para probs_tensor
        q_tensor probs_tensor = {
            .data = (void*)probs_buf,
            .ne = {seq_len, seq_len, 1, 1},
            .nb = {scratch->scores_stride_floats * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
            .type = Q_F32
        };
        
        q_tensor v_head_tensor = {
            .data = (void*)(scratch->v_heads + (size_t)kv_head_idx * (seq_len * head_dim)),
            .ne = {seq_len, head_dim, 1, 1},
            .nb = {head_dim * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
            .type = Q_F32
        };
        
        q_tensor attn_head_tensor = {
            .data = (void*)scratch->attn_head_buf,
            .ne = {seq_len, head_dim, 1, 1},
            .nb = {head_dim * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
            .type = Q_F32
        };
        
        ret = q_matmul_f32_avx2(&probs_tensor, &v_head_tensor, &attn_head_tensor, ctx);
        if (ret != Q_OK) {
            #ifdef DEBUG
            fprintf(stderr, "ERROR: Attention output MatMul failed: ret=%d\n", ret);
            abort();
            #endif
            return ret;
        }
        
        // Concatenate attention outputs from all heads
        // CORREÇÃO: Usar buffer temporário (scratch->q_rope_buf) para evitar aliasing
        // Não escrever diretamente em output porque vamos usar output como entrada depois
        for (uint32_t t = 0; t < seq_len; t++) {
            float* out_head = scratch->q_rope_buf + (size_t)t * dim + (size_t)qh * head_dim;
            const float* attn_head = scratch->attn_head_buf + (size_t)t * head_dim;
            memcpy(out_head, attn_head, head_dim * sizeof(float));
        }
    }
    
    // Output projection: attn_out @ wo -> [seq_len, dim]
    // CRITICAL FIX: Use q_gemv_q4_f32_avx2 for Q4_0 weights
    // CORREÇÃO: Usar scratch->q_rope_buf como entrada (dados concatenados das heads)
    // e output como saída (sem aliasing)
    
    for (uint32_t i = 0; i < seq_len; i++) {
        const float* attn_row = scratch->q_rope_buf + (size_t)i * dim;  // Input: dados concatenados das heads
        float* out_row = output + (size_t)i * dim;  // Output: escrever diretamente em output
        
        ret = q_gemv_q4_f32_avx2(layer->wo, attn_row, out_row);
        if (ret != Q_OK) {
            #ifdef DEBUG
            fprintf(stderr, "ERROR: Output projection failed at row %u: ret=%d\n", i, ret);
            abort();
            #endif
            return ret;
        }
    }
    
    return Q_OK;
}

// Helper: MLP forward pass (SwiGLU)
// CORRIGIDO: Usa scratchpad reutilizável (Correção 1)
static q_error_code llama_mlp_forward(
    q_llama_layer* restrict layer,
    q_context* restrict ctx __attribute__((unused)),  // Pode não ser usado em todas as implementações
    const q_llama_config* restrict config,
    const float* restrict x,           // Input [seq_len, dim]
    float* restrict output,             // Output [seq_len, dim]
    uint32_t seq_len,
    layer_scratchpad* restrict scratch  // NOVO: scratchpad reutilizável
) {
    uint32_t dim = config->dim;
    uint32_t hidden_dim = config->hidden_dim;
    
    // CORREÇÃO 1: Usar scratchpad em vez de q_arena_alloc
    // REMOVIDO: Todas as alocações q_arena_alloc
    // USAR: scratch->gate_buf, scratch->up_buf, etc.
    
    // Gate projection: x_norm @ w_gate -> gate_buf [seq_len, hidden_dim]
    // CRITICAL FIX: Use q_gemv_q4_f32_avx2 for Q4_0 weights
    
    q_error_code ret;
    for (uint32_t i = 0; i < seq_len; i++) {
        const float* x_row = x + (size_t)i * dim;
        float* gate_row = scratch->gate_buf + (size_t)i * hidden_dim;
        ret = q_gemv_q4_f32_avx2(layer->w_gate, x_row, gate_row);
        if (ret != Q_OK) {
            #ifdef DEBUG
            fprintf(stderr, "ERROR: Gate projection failed at row %u: ret=%d\n", i, ret);
            abort();
            #endif
            return ret;
        }
    }
    
    // Up projection: x_norm @ w_up -> up_buf [seq_len, hidden_dim]
    // CRITICAL FIX: Use q_gemv_q4_f32_avx2 for Q4_0 weights
    
    for (uint32_t i = 0; i < seq_len; i++) {
        const float* x_row = x + (size_t)i * dim;
        float* up_row = scratch->up_buf + (size_t)i * hidden_dim;
        ret = q_gemv_q4_f32_avx2(layer->w_up, x_row, up_row);
        if (ret != Q_OK) {
            #ifdef DEBUG
            fprintf(stderr, "ERROR: Up projection failed at row %u: ret=%d\n", i, ret);
            abort();
            #endif
            return ret;
        }
    }
    
    // CORREÇÃO 1: SiLU activation usando scratchpad
    // REMOVIDO: Alocação q_arena_alloc
    ret = q_silu_f32_avx2(scratch->gate_buf, scratch->gate_silu, seq_len * hidden_dim);
    
    if (ret != Q_OK) return ret;
    
    // Element-wise multiply: gate * up
    uint32_t mul_size = seq_len * hidden_dim;
    
    q_tensor mul_tensor = {
        .data = (void*)scratch->mul_buf,
        .ne = {mul_size, 1, 1, 1},
        .nb = {sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    q_tensor gate_silu_flat = {
        .data = (void*)scratch->gate_silu,
        .ne = {mul_size, 1, 1, 1},
        .nb = {sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    q_tensor up_flat = {
        .data = (void*)scratch->up_buf,
        .ne = {mul_size, 1, 1, 1},
        .nb = {sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    // CRITICAL FIX: nb[0] must equal mul_size * sizeof(float) for contiguous 1D tensor
    mul_tensor.nb[0] = mul_size * sizeof(float);
    gate_silu_flat.nb[0] = mul_size * sizeof(float);
    up_flat.nb[0] = mul_size * sizeof(float);
    
    ret = q_mul_f32_avx2(&gate_silu_flat, &up_flat, &mul_tensor);
    
    if (ret != Q_OK) return ret;
    
    // Down projection: mul_buf @ w_down -> output [seq_len, dim]
    // CRITICAL FIX: Use q_gemv_q4_f32_avx2 for Q4_0 weights
    for (uint32_t i = 0; i < seq_len; i++) {
        const float* mul_row = scratch->mul_buf + (size_t)i * hidden_dim;
        float* out_row = output + (size_t)i * dim;
        ret = q_gemv_q4_f32_avx2(layer->w_down, mul_row, out_row);
        
        if (ret != Q_OK) return ret;
    }
    
    return Q_OK;
}

// Helper: Single layer forward pass
// CORRIGIDO: Usa scratchpad reutilizável (Correção 1)
static q_error_code llama_layer_forward(
    q_llama_layer* restrict layer,
    q_context* restrict ctx,
    const q_llama_model* restrict model,  // NOVO: necessário para RoPE pré-calculado
    const q_llama_config* restrict config,
    const float* restrict x,           // Input [seq_len, dim]
    float* restrict output,            // Output [seq_len, dim]
    uint32_t layer_idx,
    uint32_t seq_len,
    uint32_t pos,
    layer_scratchpad* restrict scratch  // NOVO: scratchpad reutilizável
) {
    uint32_t dim = config->dim;
    
    // CORREÇÃO 1: Usar scratchpad em vez de q_arena_alloc
    // REMOVIDO: Todas as alocações q_arena_alloc
    // USAR: scratch->attn_out, scratch->mlp_out, scratch->x_norm, scratch->x_norm_mlp
    
    // Attention block
    q_error_code ret = llama_attention_forward(layer, ctx, model, config, x, scratch->attn_out, layer_idx, seq_len, pos, scratch);
    if (ret != Q_OK) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: llama_attention_forward failed: ret=%d\n", ret);
        abort();
        #endif
        return ret;
    }
    
    // Residual connection: x = x + attn_out
    uint32_t total_size = seq_len * dim;
    
    q_tensor x_tensor = {
        .data = (void*)x,
        .ne = {total_size, 1, 1, 1},
        .nb = {total_size * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    q_tensor attn_tensor = {
        .data = (void*)scratch->attn_out,
        .ne = {total_size, 1, 1, 1},
        .nb = {total_size * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    q_tensor x_residual = {
        .data = (void*)scratch->x_norm,
        .ne = {total_size, 1, 1, 1},
        .nb = {total_size * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    ret = q_add_f32_avx2(&x_tensor, &attn_tensor, &x_residual);
    if (ret != Q_OK) return ret;
    
    // Pre-MLP RMSNorm (usar scratch->x_norm_mlp)
    ret = q_rmsnorm_f32_avx2(scratch->x_norm, (const float*)layer->ffn_norm->data, scratch->x_norm_mlp, dim, config->rms_norm_eps);
    if (ret != Q_OK) return ret;
    
    // MLP block
    ret = llama_mlp_forward(layer, ctx, config, scratch->x_norm_mlp, scratch->mlp_out, seq_len, scratch);
    
    if (ret != Q_OK) return ret;
    
    // Residual connection: x = x + mlp_out
    q_tensor mlp_tensor = {
        .data = (void*)scratch->mlp_out,
        .ne = {total_size, 1, 1, 1},
        .nb = {total_size * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    q_tensor output_tensor = {
        .data = (void*)output,
        .ne = {total_size, 1, 1, 1},
        .nb = {total_size * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    ret = q_add_f32_avx2(&x_residual, &mlp_tensor, &output_tensor);
    
    if (ret != Q_OK) return ret;
    
    return Q_OK;
}

// Main forward pass function
q_error_code llama_forward(
    q_llama_model* restrict model,
    q_context* restrict ctx,
    const uint32_t* restrict tokens,
    uint32_t seq_len,
    uint32_t pos,
    float* restrict logits
) {
    // Validation
    Q_VALIDATE_PTR_OR_RETURN(model, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(ctx, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(tokens, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(logits, Q_ERR_INVALID_ARG);
    Q_VALIDATE_NONZERO_OR_RETURN(seq_len, Q_ERR_INVALID_SIZE);
    
    if (seq_len > model->config.max_seq_len) {
        return Q_ERR_INVALID_SIZE;
    }
    
    if (pos >= model->config.max_seq_len) {
        return Q_ERR_INVALID_ARG;
    }
    
    if (ctx->scratch_buffer == NULL) {
        return Q_ERR_INVALID_ARG;  // Arena not allocated
    }
    
    if (ctx->kv_buffer == NULL) {
        return Q_ERR_INVALID_ARG;  // KV cache not allocated
    }
    
    // NOTE: The arena contains persistent structures (q_tensor views) allocated
    // during llama_build_graph(). We cannot reset the entire arena because that
    // would corrupt these structures. Instead, we'll allocate temporary buffers
    // after the model structures. The model structures are at the beginning of
    // the arena, so we can safely allocate after them.
    
    // For now, we'll use a simple approach: allocate temporaries after current head
    // In a production system, we'd track model_arena_head separately
    
    uint32_t dim = model->config.dim;
    uint32_t vocab_size = model->config.vocab_size;
    
    // Allocate buffer for token embeddings [seq_len, dim]
    size_t embd_size = (size_t)seq_len * (size_t)dim * sizeof(float);
    embd_size = Q_ALIGN_SIZE(embd_size);
    float* x = (float*)q_arena_alloc(ctx, embd_size);
    if (x == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    
    // Step 1: Token embeddings
    // CRITICAL VALIDATION: Verify token_embd is still valid before use
    // This catches arena corruption or pointer invalidation bugs
    Q_VALIDATE_PTR_OR_RETURN(model->token_embd, Q_ERR_INVALID_ARG);
    
    // NOTE: token_embd->type is validated in llama_build_graph()
    // Arena is not reset between llama_build_graph() and llama_forward()
    // Therefore, type validation here is redundant unless arena is reset externally
    // AddressSanitizer will catch memory corruption in DEBUG mode
    
    q_error_code ret = token_embedding_lookup(model->token_embd, tokens, seq_len, x);
    if (ret != Q_OK) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: llama_forward: token_embedding_lookup returned %d\n", ret);
        abort();
        #endif
        return ret;
    }
    
    // ============================================================================
    // CORREÇÃO 1: Alocar scratchpad UMA VEZ antes do loop de camadas
    // ============================================================================
    size_t scratchpad_size = calculate_layer_scratchpad_size(&model->config, seq_len);
    layer_scratchpad scratch;
    uint8_t* scratch_mem = (uint8_t*)q_arena_alloc(ctx, scratchpad_size);
    if (scratch_mem == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    
    init_layer_scratchpad(&scratch, scratch_mem, &model->config, seq_len);
    
    // Buffers ping-pong para saída de camada (reutilização eficiente)
    size_t layer_buf_size = (size_t)seq_len * (size_t)dim * sizeof(float);
    layer_buf_size = Q_ALIGN_SIZE(layer_buf_size);
    float* layer_buf_A = (float*)q_arena_alloc(ctx, layer_buf_size);
    float* layer_buf_B = (float*)q_arena_alloc(ctx, layer_buf_size);
    if (layer_buf_A == NULL || layer_buf_B == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    
    // Step 2: Forward through layers
    // Process each layer - REUTILIZA scratchpad para todas as camadas
    for (uint32_t l = 0; l < model->config.n_layers; l++) {
        float* output = (l % 2 == 0) ? layer_buf_B : layer_buf_A;
        
        ret = llama_layer_forward(&model->layers[l], ctx, model, &model->config, 
                                 x, output, l, seq_len, pos, &scratch);
        if (ret != Q_OK) {
            #ifdef DEBUG
            fprintf(stderr, "ERROR: llama_forward: llama_layer_forward[%u] returned %d\n", l, ret);
            abort();
            #endif
            return ret;
        }
        
        x = output; // Swap para próxima camada
    }
    
    // Step 3: Final RMSNorm
    float* x_final = (float*)q_arena_alloc(ctx, layer_buf_size);
    if (x_final == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    
    ret = q_rmsnorm_f32_avx2(x, (const float*)model->output_norm->data, x_final, dim, model->config.rms_norm_eps);
    if (ret != Q_OK) {
        return ret;
    }
    
    // Step 4: LM Head projection
    // For last token only (incremental generation: seq_len == 1)
    // For prefill (seq_len > 1), we only need logits for last position
    
    // CRITICAL FIX: Ensure last_token is 32-byte aligned
    // Always allocate aligned buffer to guarantee alignment, regardless of theoretical calculation
    // This avoids issues with pointer arithmetic that may not reflect actual memory alignment
    const float* last_token_ptr = x_final + (size_t)(seq_len - 1) * dim;
    
    // Always allocate aligned buffer for last_token to guarantee 32-byte alignment
    // This is safer than relying on pointer arithmetic which may not reflect actual memory alignment
    size_t aligned_size = Q_ALIGN_SIZE(dim * sizeof(float));
    float* last_token_aligned = (float*)q_arena_alloc(ctx, aligned_size);
    if (last_token_aligned == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    
    // Copy last token to aligned buffer
    memcpy(last_token_aligned, last_token_ptr, dim * sizeof(float));
    const float* last_token = last_token_aligned;
    
    // Create tensor view for last token [1, dim]
    q_tensor last_token_tensor = {
        .data = (void*)last_token,
        .ne = {1, dim, 1, 1},
        .nb = {dim * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    // DEBUG: Verify alignment of last_token (always print, not just in DEBUG mode)
    uintptr_t last_token_addr = (uintptr_t)last_token;
    fprintf(stderr, "DEBUG: llama_forward: last_token alignment check:\n");
    fprintf(stderr, "  last_token=%p, addr=%zu\n", last_token, last_token_addr);
    fprintf(stderr, "  last_token %% 32 = %zu\n", last_token_addr % 32);
    fprintf(stderr, "  last_token %% 64 = %zu\n", last_token_addr % 64);
    fprintf(stderr, "  last_token_tensor.nb[0]=%zu, nb[0] %% 32 = %zu\n", 
            last_token_tensor.nb[0], last_token_tensor.nb[0] % 32);
    
    // CRITICAL FIX: For transposed tensors (B->nb[0] == sizeof(float)),
    // we cannot guarantee alignment of all elements.
    // q_matmul_f32_avx2 will detect this and use unaligned loads automatically.
    // No need to validate base pointer alignment here - let q_matmul_f32_avx2 handle it.
    
    // Create transposed view of output: [vocab_size, dim] -> [dim, vocab_size]
    // This allows us to compute: last_token [1, dim] @ output^T [dim, vocab_size] -> logits [1, vocab_size]
    q_tensor output_t_tensor = {
        .data = (void*)model->output->data,
        .ne = {dim, vocab_size, 1, 1},  // Transposed dimensions
        .nb = {sizeof(float), dim * sizeof(float), sizeof(float), sizeof(float)},  // Transposed strides
        .type = Q_F32
    };
    
    // DEBUG: Verify alignment of output tensor (always print)
    uintptr_t output_addr = (uintptr_t)model->output->data;
    fprintf(stderr, "DEBUG: llama_forward: output_t_tensor alignment check:\n");
    fprintf(stderr, "  output->data=%p, addr=%zu\n", model->output->data, output_addr);
    fprintf(stderr, "  output %% 32 = %zu\n", output_addr % 32);
    fprintf(stderr, "  output_t_tensor.nb[0]=%zu, nb[0] %% 32 = %zu\n", 
            output_t_tensor.nb[0], output_t_tensor.nb[0] % 32);
    
    // Create tensor view for logits [1, vocab_size]
    q_tensor logits_tensor = {
        .data = (void*)logits,
        .ne = {1, vocab_size, 1, 1},
        .nb = {vocab_size * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    // DEBUG: Verify alignment of logits tensor (always print)
    uintptr_t logits_addr = (uintptr_t)logits;
    fprintf(stderr, "DEBUG: llama_forward: logits_tensor alignment check:\n");
    fprintf(stderr, "  logits=%p, addr=%zu\n", logits, logits_addr);
    fprintf(stderr, "  logits %% 32 = %zu\n", logits_addr % 32);
    fprintf(stderr, "  logits_tensor.nb[0]=%zu, nb[0] %% 32 = %zu\n", 
            logits_tensor.nb[0], logits_tensor.nb[0] % 32);
    
    // Compute: last_token [1, dim] @ output^T [dim, vocab_size] -> logits [1, vocab_size]
    ret = q_matmul_f32_avx2(&last_token_tensor, &output_t_tensor, &logits_tensor, ctx);
    if (ret != Q_OK) {
        return ret;
    }
    
    return Q_OK;
}
