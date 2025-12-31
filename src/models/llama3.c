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
    
    // CRITICAL FIX: Zero-initialize entire tensor structure to prevent
    // uninitialized memory corruption. This ensures all fields (including
    // 'type') start with known values, preventing Q_ERR_INVALID_DTYPE errors.
    // Time Complexity: O(1) - single memset call
    // Space Complexity: O(1) - no additional memory
    memset(tensor, 0, sizeof(q_tensor));
    
    // Initialize tensor view
    tensor->data = data_ptr;
    tensor->scales = NULL;
    tensor->ne[0] = ne0;
    tensor->ne[1] = ne1;
    tensor->ne[2] = ne2;
    tensor->ne[3] = ne3;
    
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
    
    tensor->type = type;
    strncpy(tensor->name, name, sizeof(tensor->name) - 1);
    tensor->name[sizeof(tensor->name) - 1] = '\0';
    
    // CRITICAL VALIDATION: Verify type was set correctly
    // This catches any memory corruption or initialization bugs immediately
    if (tensor->type != type) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: create_tensor_view: type corruption detected!\n");
        fprintf(stderr, "  Expected type: %d, Got type: %d\n", type, tensor->type);
        fprintf(stderr, "  Tensor pointer: %p\n", (void*)tensor);
        abort();
        #endif
        // In Release, return NULL to indicate failure
        return NULL;
    }
    
    return tensor;
}

q_error_code llama_build_graph(q_context* restrict ctx, llama_model* restrict model) {
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
    model->config.rope_theta = ctx->header->rope_freq_base;  // rope_freq_base stored as rope_theta
    model->config.rms_norm_eps = 1e-6f;  // Default Llama-3 epsilon
    
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
    // Store pointer and type for later validation
    volatile q_tensor* token_embd_check = model->token_embd;
    volatile q_dtype type_check = token_embd_check->type;
    
    // DEBUG: Print type immediately after creation (for diagnosis)
    char create_msg[128];
    int create_len = snprintf(create_msg, sizeof(create_msg),
        "DEBUG: llama_build_graph: Created token_embd, type = %d (expected %d), ptr = %p\n",
        (int)type_check, (int)Q_F32, (void*)token_embd_check);
    Q_WRITE_STDERR( create_msg, (size_t)create_len);
    
    if (type_check != Q_F32) {
        char err_msg[256];
        int err_len = snprintf(err_msg, sizeof(err_msg),
            "ERROR: llama_build_graph: token_embd->type = %d, expected Q_F32 (%d)\n"
            "  token_embd pointer: %p\n",
            (int)type_check, (int)Q_F32,
            (void*)token_embd_check);
        Q_WRITE_STDERR( err_msg, (size_t)err_len);
        #ifdef DEBUG
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
    model->layers = (llama_layer*)q_arena_alloc(ctx, sizeof(llama_layer) * model->config.n_layers);
    if (model->layers == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    
    // 5. Build each layer
    for (uint32_t i = 0; i < model->config.n_layers; i++) {
        llama_layer* layer = &model->layers[i];
        memset(layer, 0, sizeof(llama_layer));
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
    
    return Q_OK;
}

void llama_free_graph(llama_model* restrict model) {
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
    memset(&model->config, 0, sizeof(llama_config));
}

// ============================================================================
// FASE 3.3: Forward Pass Implementation
// ============================================================================

// Helper: Get KV cache pointer for a specific layer/head/position
// Layout: [n_layers, n_kv_heads, max_seq_len, head_dim]
// Returns NULL if invalid parameters
static float* get_kv_cache_ptr(
    q_context* restrict ctx,
    const llama_config* restrict config,
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

// Helper: Generate RoPE cos/sin tables for a specific position
// Generates cos/sin values for head_dim/2 pairs, duplicated for AVX2
// Formula: theta_i = rope_theta^(-2i/head_dim) * pos
// Output: cos_buf[head_dim], sin_buf[head_dim] (duplicated layout: [c0,c0,c1,c1,...])
static q_error_code generate_rope_cos_sin(
    float rope_theta,
    uint32_t head_dim,
    uint32_t pos,
    float* restrict cos_buf,  // Output [head_dim], 32-byte aligned
    float* restrict sin_buf   // Output [head_dim], 32-byte aligned
) {
    Q_VALIDATE_PTR_OR_RETURN(cos_buf, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(sin_buf, Q_ERR_INVALID_ARG);
    Q_VALIDATE_NONZERO_OR_RETURN(head_dim, Q_ERR_INVALID_SIZE);
    Q_VALIDATE_MULTIPLE_OR_RETURN(head_dim, 2, Q_ERR_INVALID_SIZE); // Must be even
    
    const uint32_t num_pairs = head_dim / 2;
    
    // Generate cos/sin for each pair
    for (uint32_t i = 0; i < num_pairs; i++) {
        // Calculate frequency: theta_i = rope_theta^(-2i/head_dim)
        float freq_exp = -2.0f * (float)i / (float)head_dim;
        float theta = powf(rope_theta, freq_exp) * (float)pos;
        
        // Calculate cos and sin
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
    
    // CRITICAL DEBUG: Print type at function entry (for diagnosis)
    volatile q_dtype type_at_entry = token_embd->type;
    char entry_msg[256];
    int entry_len = snprintf(entry_msg, sizeof(entry_msg),
        "DEBUG: token_embedding_lookup: Entry, token_embd->type = %d (expected %d), ptr = %p\n",
        (int)type_at_entry, (int)Q_F32, (void*)token_embd);
    Q_WRITE_STDERR( entry_msg, (size_t)entry_len);
    
    // CRITICAL DEBUG: Always print error details (critical safety check)
    if (type_at_entry != Q_F32) {
        // Use direct write to stderr to ensure message is printed
        char msg[512];
        int len = snprintf(msg, sizeof(msg),
            "ERROR: token_embedding_lookup: token_embd->type = %d, expected Q_F32 (%d)\n"
            "  token_embd pointer: %p\n"
            "  token_embd->data: %p\n"
            "  token_embd->ne[0]=%u, ne[1]=%u\n"
            "  token_embd->name: %.32s\n",
            (int)type_at_entry, (int)Q_F32,
            (void*)token_embd,
            (void*)token_embd->data,
            token_embd->ne[0], token_embd->ne[1],
            token_embd->name);
        Q_WRITE_STDERR( msg, (size_t)len);  // Write directly to stderr (fd 2)
        #ifdef DEBUG
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
static q_error_code llama_layer_forward(
    llama_layer* restrict layer,
    q_context* restrict ctx,
    const llama_config* restrict config,
    const float* restrict x,           // Input [seq_len, dim]
    float* restrict output,            // Output [seq_len, dim]
    uint32_t layer_idx,
    uint32_t seq_len,
    uint32_t pos
);

// Helper: Attention forward pass with GQA support
static q_error_code llama_attention_forward(
    llama_layer* restrict layer,
    q_context* restrict ctx,
    const llama_config* restrict config,
    const float* restrict x,           // Input [seq_len, dim]
    float* restrict output,            // Output [seq_len, dim]
    uint32_t layer_idx,
    uint32_t seq_len,
    uint32_t pos
) {
    uint32_t dim = config->dim;
    uint32_t n_heads = config->n_heads;
    uint32_t n_kv_heads = config->n_kv_heads;
    uint32_t head_dim = dim / n_heads;
    // uint32_t q_per_kv = n_heads / n_kv_heads;  // GQA replication factor (TODO: use in attention computation)
    (void)layer_idx;  // TODO: use for KV cache access
    (void)pos;        // TODO: use for RoPE and KV cache update
    
    // Allocate temporary buffers from arena
    size_t qkv_size = (size_t)seq_len * (size_t)dim * sizeof(float);
    qkv_size = Q_ALIGN_SIZE(qkv_size);
    
    // CRITICAL FIX: Allocate separate buffer for x_norm to avoid aliasing
    float* x_norm = (float*)q_arena_alloc(ctx, qkv_size);
    float* q_buf = (float*)q_arena_alloc(ctx, qkv_size);
    float* k_buf = (float*)q_arena_alloc(ctx, qkv_size);
    float* v_buf = (float*)q_arena_alloc(ctx, qkv_size);
    float* attn_out = (float*)q_arena_alloc(ctx, qkv_size);
    
    if (x_norm == NULL || q_buf == NULL || k_buf == NULL || v_buf == NULL || attn_out == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    
    // Pre-attention RMSNorm: x -> x_norm
    q_error_code ret = q_rmsnorm_f32_avx2(x, (const float*)layer->attn_norm->data, x_norm, dim, config->rms_norm_eps);
    if (ret != Q_OK) return ret;
    
    // Q/K/V projections using GEMV (Q4_0 weights)
    // CRITICAL FIX: Use q_gemv_q4_f32_avx2 for Q4_0 weights (not q_matmul_f32_avx2)
    // For each row of x_norm, compute GEMV: output_row = x_norm_row @ weights^T
    // This is equivalent to MatMul but optimized for Q4_0 weights
    
    // Q projection: x_norm @ wq -> q_buf [seq_len, dim]
    // DEBUG: Print dimensions for diagnosis
    char q_debug[256];
    int q_debug_len = snprintf(q_debug, sizeof(q_debug),
        "DEBUG: Q projection: wq->ne[0]=%u, wq->ne[1]=%u, dim=%u, dim%%32=%u\n",
        layer->wq->ne[0], layer->wq->ne[1], dim, dim % 32);
    Q_WRITE_STDERR( q_debug, (size_t)q_debug_len);
    
    for (uint32_t i = 0; i < seq_len; i++) {
        const float* x_row = x_norm + (size_t)i * dim;
        float* q_row = q_buf + (size_t)i * dim;
        ret = q_gemv_q4_f32_avx2(layer->wq, x_row, q_row);
        if (ret != Q_OK) {
            char err_msg[256];
            int err_len = snprintf(err_msg, sizeof(err_msg),
                "ERROR: Q projection failed at row %u: ret=%d\n", i, ret);
            Q_WRITE_STDERR( err_msg, (size_t)err_len);
            return ret;
        }
    }
    
    // K projection: x_norm @ wk -> k_buf [seq_len, n_kv_heads * head_dim]
    uint32_t kv_dim = n_kv_heads * head_dim;
    // CRITICAL: q_gemv_q4_f32_avx2 requires input size (dim) to be multiple of 32
    // dim is validated to be multiple of 32 in llama_build_graph, so this should be OK
    
    for (uint32_t i = 0; i < seq_len; i++) {
        const float* x_row = x_norm + (size_t)i * dim;
        float* k_row = k_buf + (size_t)i * kv_dim;
        ret = q_gemv_q4_f32_avx2(layer->wk, x_row, k_row);
        if (ret != Q_OK) {
            char err_msg[256];
            int err_len = snprintf(err_msg, sizeof(err_msg),
                "ERROR: K projection failed at row %u: ret=%d, dim=%u, kv_dim=%u\n",
                i, ret, dim, kv_dim);
            Q_WRITE_STDERR( err_msg, (size_t)err_len);
            return ret;
        }
    }
    
    // V projection: x_norm @ wv -> v_buf [seq_len, n_kv_heads * head_dim]
    for (uint32_t i = 0; i < seq_len; i++) {
        const float* x_row = x_norm + (size_t)i * dim;
        float* v_row = v_buf + (size_t)i * kv_dim;
        ret = q_gemv_q4_f32_avx2(layer->wv, x_row, v_row);
        if (ret != Q_OK) {
            char err_msg[256];
            int err_len = snprintf(err_msg, sizeof(err_msg),
                "ERROR: V projection failed at row %u: ret=%d, dim=%u, kv_dim=%u\n",
                i, ret, dim, kv_dim);
            Q_WRITE_STDERR( err_msg, (size_t)err_len);
            return ret;
        }
    }
    
    // Allocate buffers for RoPE cos/sin and attention computation
    size_t head_dim_size = (size_t)head_dim * sizeof(float);
    head_dim_size = Q_ALIGN_SIZE(head_dim_size);
    
    float* cos_buf = (float*)q_arena_alloc(ctx, head_dim_size);
    float* sin_buf = (float*)q_arena_alloc(ctx, head_dim_size);
    float* q_rope_buf = (float*)q_arena_alloc(ctx, qkv_size);
    float* k_rope_buf = (float*)q_arena_alloc(ctx, qkv_size);
    
    // Allocate buffers for attention scores and probs
    size_t scores_size = (size_t)seq_len * (size_t)seq_len * sizeof(float);
    scores_size = Q_ALIGN_SIZE(scores_size);
    float* scores_buf = (float*)q_arena_alloc(ctx, scores_size);
    
    if (cos_buf == NULL || sin_buf == NULL || q_rope_buf == NULL || 
        k_rope_buf == NULL || scores_buf == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    
    // Apply RoPE to Q and K (per head, per token)
    // Q: [seq_len, n_heads, head_dim] -> reshape to [seq_len * n_heads, head_dim]
    // K: [seq_len, n_kv_heads, head_dim] -> reshape to [seq_len * n_kv_heads, head_dim]
    for (uint32_t t = 0; t < seq_len; t++) {
        uint32_t token_pos = pos + t;  // Absolute position in sequence
        
        // Generate RoPE cos/sin for this position
        ret = generate_rope_cos_sin(config->rope_theta, head_dim, token_pos, cos_buf, sin_buf);
        if (ret != Q_OK) return ret;
        
        // Apply RoPE to each Q head
        for (uint32_t h = 0; h < n_heads; h++) {
            const float* q_head = q_buf + (size_t)t * dim + (size_t)h * head_dim;
            float* q_head_out = q_rope_buf + (size_t)t * dim + (size_t)h * head_dim;
            
            ret = q_rope_f32_avx2(q_head, cos_buf, sin_buf, q_head_out, head_dim);
            if (ret != Q_OK) return ret;
        }
        
        // Apply RoPE to each K head (KV heads only)
        for (uint32_t h = 0; h < n_kv_heads; h++) {
            const float* k_head = k_buf + (size_t)t * (n_kv_heads * head_dim) + (size_t)h * head_dim;
            float* k_head_out = k_rope_buf + (size_t)t * (n_kv_heads * head_dim) + (size_t)h * head_dim;
            
            ret = q_rope_f32_avx2(k_head, cos_buf, sin_buf, k_head_out, head_dim);
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
            
            const float* k_src = k_rope_buf + (size_t)t * (n_kv_heads * head_dim) + (size_t)h * head_dim;
            memcpy(k_cache, k_src, head_dim * sizeof(float));
            
            // Store V
            float* v_cache = get_kv_cache_ptr(ctx, config, layer_idx, h, cache_pos, false);
            if (v_cache == NULL) return Q_ERR_INVALID_ARG;
            
            const float* v_src = v_buf + (size_t)t * (n_kv_heads * head_dim) + (size_t)h * head_dim;
            memcpy(v_cache, v_src, head_dim * sizeof(float));
        }
    }
    
    // Reorganize Q/K/V from [seq_len, dim] to [seq_len, n_heads, head_dim] layout
    // Allocate buffers for reshaped Q/K/V (per-head layout)
    size_t q_per_head_size = (size_t)seq_len * head_dim * sizeof(float);
    q_per_head_size = Q_ALIGN_SIZE(q_per_head_size);
    
    float* q_heads = (float*)q_arena_alloc(ctx, q_per_head_size * n_heads);
    float* k_heads = (float*)q_arena_alloc(ctx, q_per_head_size * n_kv_heads);
    float* v_heads = (float*)q_arena_alloc(ctx, q_per_head_size * n_kv_heads);
    
    if (q_heads == NULL || k_heads == NULL || v_heads == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    
    // Reshape Q: [seq_len, dim] -> [seq_len, n_heads, head_dim]
    for (uint32_t t = 0; t < seq_len; t++) {
        for (uint32_t h = 0; h < n_heads; h++) {
            const float* q_src = q_rope_buf + (size_t)t * dim + (size_t)h * head_dim;
            float* q_dst = q_heads + (size_t)h * (seq_len * head_dim) + (size_t)t * head_dim;
            memcpy(q_dst, q_src, head_dim * sizeof(float));
        }
    }
    
    // Reshape K/V: [seq_len, n_kv_heads * head_dim] -> [seq_len, n_kv_heads, head_dim]
    for (uint32_t t = 0; t < seq_len; t++) {
        for (uint32_t h = 0; h < n_kv_heads; h++) {
            const float* k_src = k_rope_buf + (size_t)t * (n_kv_heads * head_dim) + (size_t)h * head_dim;
            float* k_dst = k_heads + (size_t)h * (seq_len * head_dim) + (size_t)t * head_dim;
            memcpy(k_dst, k_src, head_dim * sizeof(float));
            
            const float* v_src = v_buf + (size_t)t * (n_kv_heads * head_dim) + (size_t)h * head_dim;
            float* v_dst = v_heads + (size_t)h * (seq_len * head_dim) + (size_t)t * head_dim;
            memcpy(v_dst, v_src, head_dim * sizeof(float));
        }
    }
    
    // Allocate buffers ONCE before the loop (optimization: reduce arena pressure)
    size_t attn_head_size = (size_t)seq_len * head_dim * sizeof(float);
    attn_head_size = Q_ALIGN_SIZE(attn_head_size);
    float* attn_head_buf = (float*)q_arena_alloc(ctx, attn_head_size);
    
    // Allocate buffer for transposed K (reused for all heads)
    size_t k_t_size = (size_t)head_dim * seq_len * sizeof(float);
    k_t_size = Q_ALIGN_SIZE(k_t_size);
    float* k_t_buf = (float*)q_arena_alloc(ctx, k_t_size);
    
    if (attn_head_buf == NULL || k_t_buf == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    
    // CRITICAL: Ensure k_t_buf is properly aligned for AVX2 (32-byte alignment)
    // Q_ALIGN_SIZE already aligns to 64 bytes, but verify explicitly
    if (((uintptr_t)k_t_buf % 32) != 0) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: k_t_buf not 32-byte aligned: %p\n", (void*)k_t_buf);
        abort();
        #endif
        return Q_ERR_MISALIGNED;
    }
    
    // Process each query head
    float scale = 1.0f / sqrtf((float)head_dim);
    
    for (uint32_t qh = 0; qh < n_heads; qh++) {
        // Determine which KV head to use (GQA: multiple Q heads share KV head)
        uint32_t kv_head_idx = qh / (n_heads / n_kv_heads);
        
        // Extract Q head: [seq_len, head_dim]
        q_tensor q_head_tensor = {
            .data = (void*)(q_heads + (size_t)qh * (seq_len * head_dim)),
            .ne = {seq_len, head_dim, 1, 1},
            .nb = {head_dim * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
            .type = Q_F32
        };
        
        // Compute scores: Q @ K^T -> [seq_len, seq_len]
        // Transpose K: [seq_len, head_dim] -> [head_dim, seq_len]
        // Reuse k_t_buf allocated before loop
        
        // Transpose K: [seq_len, head_dim] -> [head_dim, seq_len]
        const float* k_head_data = k_heads + (size_t)kv_head_idx * (seq_len * head_dim);
        for (uint32_t i = 0; i < seq_len; i++) {
            for (uint32_t j = 0; j < head_dim; j++) {
                k_t_buf[j * seq_len + i] = k_head_data[i * head_dim + j];
            }
        }
        
        q_tensor k_t_tensor = {
            .data = (void*)k_t_buf,
            .ne = {head_dim, seq_len, 1, 1},
            .nb = {seq_len * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
            .type = Q_F32
        };
        
        q_tensor scores_tensor = {
            .data = (void*)scores_buf,
            .ne = {seq_len, seq_len, 1, 1},
            .nb = {seq_len * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
            .type = Q_F32
        };
        
        // CRITICAL VALIDATION: Verify dimensions match before MatMul
        // A[seq_len, head_dim] @ B[head_dim, seq_len] -> C[seq_len, seq_len]
        if (q_head_tensor.ne[0] != seq_len || q_head_tensor.ne[1] != head_dim) {
            char err_msg[256];
            int err_len = snprintf(err_msg, sizeof(err_msg),
                "ERROR: q_head_tensor dimensions mismatch: expected [%u,%u], got [%u,%u]\n",
                seq_len, head_dim, q_head_tensor.ne[0], q_head_tensor.ne[1]);
            Q_WRITE_STDERR( err_msg, (size_t)err_len);
            return Q_ERR_INVALID_SIZE;
        }
        if (k_t_tensor.ne[0] != head_dim || k_t_tensor.ne[1] != seq_len) {
            char err_msg[256];
            int err_len = snprintf(err_msg, sizeof(err_msg),
                "ERROR: k_t_tensor dimensions mismatch: expected [%u,%u], got [%u,%u]\n",
                head_dim, seq_len, k_t_tensor.ne[0], k_t_tensor.ne[1]);
            Q_WRITE_STDERR( err_msg, (size_t)err_len);
            return Q_ERR_INVALID_SIZE;
        }
        if (scores_tensor.ne[0] != seq_len || scores_tensor.ne[1] != seq_len) {
            char err_msg[256];
            int err_len = snprintf(err_msg, sizeof(err_msg),
                "ERROR: scores_tensor dimensions mismatch: expected [%u,%u], got [%u,%u]\n",
                seq_len, seq_len, scores_tensor.ne[0], scores_tensor.ne[1]);
            Q_WRITE_STDERR( err_msg, (size_t)err_len);
            return Q_ERR_INVALID_SIZE;
        }
        
        // DEBUG: Print dimensions for attention scores
        char scores_debug[256];
        int scores_debug_len = snprintf(scores_debug, sizeof(scores_debug),
            "DEBUG: Attention scores: q_head [%u,%u] @ k_t [%u,%u] -> scores [%u,%u]\n",
            q_head_tensor.ne[0], q_head_tensor.ne[1],
            k_t_tensor.ne[0], k_t_tensor.ne[1],
            scores_tensor.ne[0], scores_tensor.ne[1]);
        Q_WRITE_STDERR( scores_debug, (size_t)scores_debug_len);
        
        ret = q_matmul_f32_avx2(&q_head_tensor, &k_t_tensor, &scores_tensor, ctx);
        if (ret != Q_OK) {
            char err_msg[256];
            int err_len = snprintf(err_msg, sizeof(err_msg),
                "ERROR: Attention scores MatMul failed: ret=%d\n", ret);
            Q_WRITE_STDERR( err_msg, (size_t)err_len);
            return ret;
        }
        
        // Scale scores: scores *= 1/sqrt(head_dim)
        for (uint32_t i = 0; i < seq_len * seq_len; i++) {
            scores_buf[i] *= scale;
        }
        
        // Apply causal mask
        char mask_debug[128];
        int mask_debug_len = snprintf(mask_debug, sizeof(mask_debug),
            "DEBUG: Applying causal mask: seq_len=%u\n", seq_len);
        Q_WRITE_STDERR( mask_debug, (size_t)mask_debug_len);
        
        ret = q_causal_mask_f32_avx2(&scores_tensor, -1e9f);
        if (ret != Q_OK) {
            char err_msg[256];
            int err_len = snprintf(err_msg, sizeof(err_msg),
                "ERROR: Causal mask failed: ret=%d\n", ret);
            Q_WRITE_STDERR( err_msg, (size_t)err_len);
            return ret;
        }
        
        // Softmax: probs = softmax(scores) per row
        float* probs_buf = scores_buf;  // Reuse scores buffer
        char softmax_debug[128];
        int softmax_debug_len = snprintf(softmax_debug, sizeof(softmax_debug),
            "DEBUG: Applying softmax: seq_len=%u\n", seq_len);
        Q_WRITE_STDERR( softmax_debug, (size_t)softmax_debug_len);
        
        for (uint32_t i = 0; i < seq_len; i++) {
            ret = q_softmax_f32_avx2(&scores_buf[i * seq_len], &probs_buf[i * seq_len], seq_len);
            if (ret != Q_OK) {
                char err_msg[256];
                int err_len = snprintf(err_msg, sizeof(err_msg),
                    "ERROR: Softmax failed at row %u: ret=%d\n", i, ret);
                Q_WRITE_STDERR( err_msg, (size_t)err_len);
                return ret;
            }
        }
        
        // Attention output: probs @ V -> [seq_len, head_dim]
        q_tensor probs_tensor = {
            .data = (void*)probs_buf,
            .ne = {seq_len, seq_len, 1, 1},
            .nb = {seq_len * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
            .type = Q_F32
        };
        
        q_tensor v_head_tensor = {
            .data = (void*)(v_heads + (size_t)kv_head_idx * (seq_len * head_dim)),
            .ne = {seq_len, head_dim, 1, 1},
            .nb = {head_dim * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
            .type = Q_F32
        };
        
        q_tensor attn_head_tensor = {
            .data = (void*)attn_head_buf,
            .ne = {seq_len, head_dim, 1, 1},
            .nb = {head_dim * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
            .type = Q_F32
        };
        
        // DEBUG: Print dimensions for attention output MatMul
        char attn_out_debug[256];
        int attn_out_debug_len = snprintf(attn_out_debug, sizeof(attn_out_debug),
            "DEBUG: Attention output: probs [%u,%u] @ v_head [%u,%u] -> attn_head [%u,%u]\n",
            probs_tensor.ne[0], probs_tensor.ne[1],
            v_head_tensor.ne[0], v_head_tensor.ne[1],
            attn_head_tensor.ne[0], attn_head_tensor.ne[1]);
        Q_WRITE_STDERR( attn_out_debug, (size_t)attn_out_debug_len);
        
        ret = q_matmul_f32_avx2(&probs_tensor, &v_head_tensor, &attn_head_tensor, ctx);
        if (ret != Q_OK) {
            char err_msg[256];
            int err_len = snprintf(err_msg, sizeof(err_msg),
                "ERROR: Attention output MatMul failed: ret=%d\n", ret);
            Q_WRITE_STDERR( err_msg, (size_t)err_len);
            return ret;
        }
        
        // Concatenate attention outputs from all heads
        // Copy attn_head_buf to output at position for this head
        for (uint32_t t = 0; t < seq_len; t++) {
            float* out_head = output + (size_t)t * dim + (size_t)qh * head_dim;
            const float* attn_head = attn_head_buf + (size_t)t * head_dim;
            memcpy(out_head, attn_head, head_dim * sizeof(float));
        }
    }
    
    // Output projection: attn_out @ wo -> [seq_len, dim]
    // CRITICAL FIX: Use q_gemv_q4_f32_avx2 for Q4_0 weights
    // DEBUG: Print dimensions
    char wo_debug[256];
    int wo_debug_len = snprintf(wo_debug, sizeof(wo_debug),
        "DEBUG: Output projection: wo->ne[0]=%u, wo->ne[1]=%u, dim=%u\n",
        layer->wo->ne[0], layer->wo->ne[1], dim);
    ssize_t write_result_wo = write(2, wo_debug, (size_t)wo_debug_len);
    (void)write_result_wo;
    
    for (uint32_t i = 0; i < seq_len; i++) {
        const float* attn_row = output + (size_t)i * dim;
        float* out_row = attn_out + (size_t)i * dim;
        
        // DEBUG: Check alignment before calling q_gemv_q4_f32_avx2
        char align_debug[256];
        int align_debug_len = snprintf(align_debug, sizeof(align_debug),
            "DEBUG: Output projection row %u: attn_row=%p (align=%zu), out_row=%p (align=%zu)\n",
            i, (void*)attn_row, ((uintptr_t)attn_row % 32), (void*)out_row, ((uintptr_t)out_row % 32));
        ssize_t write_result_align = write(2, align_debug, (size_t)align_debug_len);
        (void)write_result_align;
        
        ret = q_gemv_q4_f32_avx2(layer->wo, attn_row, out_row);
        if (ret != Q_OK) {
            char err_msg[256];
            int err_len = snprintf(err_msg, sizeof(err_msg),
                "ERROR: Output projection failed at row %u: ret=%d\n", i, ret);
            Q_WRITE_STDERR( err_msg, (size_t)err_len);
            return ret;
        }
    }
    
    // DEBUG: After output projection
    char after_wo_debug[128];
    int after_wo_debug_len = snprintf(after_wo_debug, sizeof(after_wo_debug),
        "DEBUG: After output projection, copying to output\n");
    ssize_t write_result_after_wo = write(2, after_wo_debug, (size_t)after_wo_debug_len);
    (void)write_result_after_wo;
    
    // Copy final output
    memcpy(output, attn_out, seq_len * dim * sizeof(float));
    
    char after_copy_debug[128];
    int after_copy_debug_len = snprintf(after_copy_debug, sizeof(after_copy_debug),
        "DEBUG: After copy, returning from llama_attention_forward\n");
    ssize_t write_result_after_copy = write(2, after_copy_debug, (size_t)after_copy_debug_len);
    (void)write_result_after_copy;
    
    return Q_OK;
}

// Helper: MLP forward pass (SwiGLU)
static q_error_code llama_mlp_forward(
    llama_layer* restrict layer,
    q_context* restrict ctx,
    const llama_config* restrict config,
    const float* restrict x,           // Input [seq_len, dim]
    float* restrict output,             // Output [seq_len, dim]
    uint32_t seq_len
) {
    uint32_t dim = config->dim;
    uint32_t hidden_dim = config->hidden_dim;
    
    // Allocate temporary buffers
    size_t hidden_size = (size_t)seq_len * (size_t)hidden_dim * sizeof(float);
    hidden_size = Q_ALIGN_SIZE(hidden_size);
    
    float* gate_buf = (float*)q_arena_alloc(ctx, hidden_size);
    float* up_buf = (float*)q_arena_alloc(ctx, hidden_size);
    float* mul_buf = (float*)q_arena_alloc(ctx, hidden_size);
    
    if (gate_buf == NULL || up_buf == NULL || mul_buf == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    
    // Create tensor views (x_tensor and gate_tensor not used, removed to avoid warnings)
    q_tensor up_tensor = {
        .data = (void*)up_buf,
        .ne = {seq_len, hidden_dim, 1, 1},
        .nb = {hidden_dim * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    // Gate projection: x_norm @ w_gate -> gate_buf [seq_len, hidden_dim]
    // CRITICAL FIX: Use q_gemv_q4_f32_avx2 for Q4_0 weights
    // DEBUG: Print dimensions
    char gate_debug[256];
    int gate_debug_len = snprintf(gate_debug, sizeof(gate_debug),
        "DEBUG: Gate projection: w_gate->ne[0]=%u, w_gate->ne[1]=%u, dim=%u, hidden_dim=%u\n",
        layer->w_gate->ne[0], layer->w_gate->ne[1], dim, hidden_dim);
    ssize_t write_result_gate = write(2, gate_debug, (size_t)gate_debug_len);
    (void)write_result_gate;
    
    q_error_code ret;
    for (uint32_t i = 0; i < seq_len; i++) {
        const float* x_row = x + (size_t)i * dim;
        float* gate_row = gate_buf + (size_t)i * hidden_dim;
        ret = q_gemv_q4_f32_avx2(layer->w_gate, x_row, gate_row);
        if (ret != Q_OK) {
            char err_msg[256];
            int err_len = snprintf(err_msg, sizeof(err_msg),
                "ERROR: Gate projection failed at row %u: ret=%d\n", i, ret);
            Q_WRITE_STDERR( err_msg, (size_t)err_len);
            return ret;
        }
    }
    
    // Up projection: x_norm @ w_up -> up_buf [seq_len, hidden_dim]
    // CRITICAL FIX: Use q_gemv_q4_f32_avx2 for Q4_0 weights
    char up_debug[256];
    int up_debug_len = snprintf(up_debug, sizeof(up_debug),
        "DEBUG: Up projection: w_up->ne[0]=%u, w_up->ne[1]=%u, dim=%u, hidden_dim=%u\n",
        layer->w_up->ne[0], layer->w_up->ne[1], dim, hidden_dim);
    ssize_t write_result_up = write(2, up_debug, (size_t)up_debug_len);
    (void)write_result_up;
    
    for (uint32_t i = 0; i < seq_len; i++) {
        const float* x_row = x + (size_t)i * dim;
        float* up_row = up_buf + (size_t)i * hidden_dim;
        ret = q_gemv_q4_f32_avx2(layer->w_up, x_row, up_row);
        if (ret != Q_OK) {
            char err_msg[256];
            int err_len = snprintf(err_msg, sizeof(err_msg),
                "ERROR: Up projection failed at row %u: ret=%d\n", i, ret);
            ssize_t write_result_up_err = write(2, err_msg, (size_t)err_len);
            (void)write_result_up_err;
            return ret;
        }
    }
    
    char after_up_debug[128];
    int after_up_debug_len = snprintf(after_up_debug, sizeof(after_up_debug),
        "DEBUG: After up projection, doing SiLU\n");
    Q_WRITE_STDERR(after_up_debug, (size_t)after_up_debug_len);
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            uint32_t silu_size = seq_len * hidden_dim;
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"llama3.c:%d\",\"message\":\"BEFORE gate_silu allocation\",\"data\":{\"seq_len\":%u,\"hidden_dim\":%u,\"silu_size\":%u,\"hidden_size\":%zu,\"gate_buf\":\"%p\",\"gate_buf_align\":%zu},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"A,B,C\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                    seq_len, hidden_dim, silu_size, hidden_size, (void*)gate_buf, ((uintptr_t)gate_buf % 32));
            fclose(log_file);
        }
    }
    // #endregion
    
    // SiLU activation on gate (in-place)
    float* gate_silu = (float*)q_arena_alloc(ctx, hidden_size);
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"llama3.c:%d\",\"message\":\"AFTER gate_silu allocation\",\"data\":{\"gate_silu\":\"%p\",\"gate_silu_align\":%zu,\"is_null\":%d,\"hidden_size\":%zu},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"A,D\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                    (void*)gate_silu, gate_silu ? ((uintptr_t)gate_silu % 32) : 0, gate_silu == NULL ? 1 : 0, hidden_size);
            fclose(log_file);
        }
    }
    // #endregion
    
    if (gate_silu == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            uint32_t silu_size = seq_len * hidden_dim;
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"llama3.c:%d\",\"message\":\"BEFORE q_silu_f32_avx2 call\",\"data\":{\"gate_buf\":\"%p\",\"gate_buf_align\":%zu,\"gate_silu\":\"%p\",\"gate_silu_align\":%zu,\"N\":%u,\"N_is_zero\":%d},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"A,B,C,E\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                    (void*)gate_buf, ((uintptr_t)gate_buf % 32), (void*)gate_silu, ((uintptr_t)gate_silu % 32), silu_size, (silu_size == 0 ? 1 : 0));
            fclose(log_file);
        }
    }
    // #endregion
    
    ret = q_silu_f32_avx2(gate_buf, gate_silu, seq_len * hidden_dim);
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"llama3.c:%d\",\"message\":\"AFTER q_silu_f32_avx2 call\",\"data\":{\"ret\":%d,\"ret_is_ok\":%d},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"E\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                    ret, (ret == Q_OK ? 1 : 0));
            fclose(log_file);
        }
    }
    // #endregion
    
    if (ret != Q_OK) return ret;
    
    // Element-wise multiply: gate * up
    uint32_t mul_size = seq_len * hidden_dim;
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"llama3.c:%d\",\"message\":\"BEFORE q_mul_f32_avx2 - tensor construction\",\"data\":{\"mul_size\":%u,\"gate_silu\":\"%p\",\"up_buf\":\"%p\",\"mul_buf\":\"%p\"},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"F\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                    mul_size, (void*)gate_silu, (void*)up_buf, (void*)mul_buf);
            fclose(log_file);
        }
    }
    // #endregion
    
    q_tensor mul_tensor = {
        .data = (void*)mul_buf,
        .ne = {mul_size, 1, 1, 1},
        .nb = {sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    q_tensor gate_silu_flat = {
        .data = (void*)gate_silu,
        .ne = {mul_size, 1, 1, 1},
        .nb = {sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    q_tensor up_flat = {
        .data = (void*)up_buf,
        .ne = {mul_size, 1, 1, 1},
        .nb = {sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    // CRITICAL FIX: nb[0] must equal mul_size * sizeof(float) for contiguous 1D tensor
    mul_tensor.nb[0] = mul_size * sizeof(float);
    gate_silu_flat.nb[0] = mul_size * sizeof(float);
    up_flat.nb[0] = mul_size * sizeof(float);
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"llama3.c:%d\",\"message\":\"BEFORE q_mul_f32_avx2 call\",\"data\":{\"mul_tensor_ne0\":%u,\"mul_tensor_nb0\":%zu,\"gate_silu_ne0\":%u,\"gate_silu_nb0\":%zu,\"up_ne0\":%u,\"up_nb0\":%zu,\"expected_nb0\":%u},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"F\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                    mul_tensor.ne[0], mul_tensor.nb[0], gate_silu_flat.ne[0], gate_silu_flat.nb[0],
                    up_flat.ne[0], up_flat.nb[0], mul_size * sizeof(float));
            fclose(log_file);
        }
    }
    // #endregion
    
    ret = q_mul_f32_avx2(&gate_silu_flat, &up_flat, &mul_tensor);
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"llama3.c:%d\",\"message\":\"AFTER q_mul_f32_avx2 call\",\"data\":{\"ret\":%d,\"ret_is_ok\":%d},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"F\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                    ret, (ret == Q_OK ? 1 : 0));
            fclose(log_file);
        }
    }
    // #endregion
    
    if (ret != Q_OK) return ret;
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"llama3.c:%d\",\"message\":\"BEFORE down projection\",\"data\":{\"seq_len\":%u,\"hidden_dim\":%u,\"dim\":%u,\"mul_buf\":\"%p\",\"output\":\"%p\"},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"G\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                    seq_len, hidden_dim, dim, (void*)mul_buf, (void*)output);
            fclose(log_file);
        }
    }
    // #endregion
    
    // Down projection: mul @ w_down
    q_tensor output_tensor = {
        .data = (void*)output,
        .ne = {seq_len, dim, 1, 1},
        .nb = {dim * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    q_tensor mul_2d = {
        .data = (void*)mul_buf,
        .ne = {seq_len, hidden_dim, 1, 1},
        .nb = {hidden_dim * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    // Down projection: mul_buf @ w_down -> output [seq_len, dim]
    // CRITICAL FIX: Use q_gemv_q4_f32_avx2 for Q4_0 weights
    for (uint32_t i = 0; i < seq_len; i++) {
        // #region agent log
        {
            FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
            if (log_file) {
                fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"llama3.c:%d\",\"message\":\"BEFORE q_gemv_q4_f32_avx2 (w_down) row %u\",\"data\":{\"i\":%u,\"mul_row\":\"%p\",\"out_row\":\"%p\",\"w_down_ne0\":%u,\"w_down_ne1\":%u,\"hidden_dim\":%u,\"dim\":%u},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"G\"}\n",
                        (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                        i, i, (void*)(mul_buf + (size_t)i * hidden_dim), (void*)(output + (size_t)i * dim),
                        layer->w_down ? layer->w_down->ne[0] : 0, layer->w_down ? layer->w_down->ne[1] : 0, hidden_dim, dim);
                fclose(log_file);
            }
        }
        // #endregion
        
        const float* mul_row = mul_buf + (size_t)i * hidden_dim;
        float* out_row = output + (size_t)i * dim;
        ret = q_gemv_q4_f32_avx2(layer->w_down, mul_row, out_row);
        
        // #region agent log
        {
            FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
            if (log_file) {
                fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"llama3.c:%d\",\"message\":\"AFTER q_gemv_q4_f32_avx2 (w_down) row %u\",\"data\":{\"i\":%u,\"ret\":%d,\"ret_is_ok\":%d},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"G\"}\n",
                        (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                        i, i, ret, (ret == Q_OK ? 1 : 0));
                fclose(log_file);
            }
        }
        // #endregion
        
        if (ret != Q_OK) return ret;
    }
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"llama3.c:%d\",\"message\":\"llama_mlp_forward EXIT SUCCESS\",\"data\":{\"ret\":\"Q_OK\"},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"G\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__);
            fclose(log_file);
        }
    }
    // #endregion
    
    return Q_OK;
}

// Helper: Single layer forward pass
static q_error_code llama_layer_forward(
    llama_layer* restrict layer,
    q_context* restrict ctx,
    const llama_config* restrict config,
    const float* restrict x,           // Input [seq_len, dim]
    float* restrict output,            // Output [seq_len, dim]
    uint32_t layer_idx,
    uint32_t seq_len,
    uint32_t pos
) {
    uint32_t dim = config->dim;
    
    // Allocate temporary buffers
    size_t buf_size = (size_t)seq_len * (size_t)dim * sizeof(float);
    buf_size = Q_ALIGN_SIZE(buf_size);
    
    float* attn_out = (float*)q_arena_alloc(ctx, buf_size);
    float* mlp_out = (float*)q_arena_alloc(ctx, buf_size);
    float* x_norm = (float*)q_arena_alloc(ctx, buf_size);
    
    if (attn_out == NULL || mlp_out == NULL || x_norm == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    
    // Attention block
    char before_attn_debug[128];
    int before_attn_debug_len = snprintf(before_attn_debug, sizeof(before_attn_debug),
        "DEBUG: llama_layer_forward: Calling llama_attention_forward\n");
    Q_WRITE_STDERR( before_attn_debug, (size_t)before_attn_debug_len);
    
    q_error_code ret = llama_attention_forward(layer, ctx, config, x, attn_out, layer_idx, seq_len, pos);
    if (ret != Q_OK) {
        char err_msg[256];
        int err_len = snprintf(err_msg, sizeof(err_msg),
            "ERROR: llama_attention_forward failed: ret=%d\n", ret);
        Q_WRITE_STDERR( err_msg, (size_t)err_len);
        return ret;
    }
    
    char after_attn_debug[128];
    int after_attn_debug_len = snprintf(after_attn_debug, sizeof(after_attn_debug),
        "DEBUG: llama_layer_forward: llama_attention_forward succeeded, doing residual\n");
    Q_WRITE_STDERR( after_attn_debug, (size_t)after_attn_debug_len);
    
    // Residual connection: x = x + attn_out
    uint32_t total_size = seq_len * dim;
    
    char residual_debug[256];
    int residual_debug_len = snprintf(residual_debug, sizeof(residual_debug),
        "DEBUG: Residual connection: seq_len=%u, dim=%u, total_size=%u\n",
        seq_len, dim, total_size);
    Q_WRITE_STDERR( residual_debug, (size_t)residual_debug_len);
    
    q_tensor x_tensor = {
        .data = (void*)x,
        .ne = {total_size, 1, 1, 1},
        .nb = {sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    q_tensor attn_tensor = {
        .data = (void*)attn_out,
        .ne = {total_size, 1, 1, 1},
        .nb = {sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    q_tensor x_residual = {
        .data = (void*)x_norm,
        .ne = {total_size, 1, 1, 1},
        .nb = {sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    // CRITICAL FIX: nb[0] must equal total_size * sizeof(float) for contiguous 1D tensor
    x_tensor.nb[0] = total_size * sizeof(float);
    attn_tensor.nb[0] = total_size * sizeof(float);
    x_residual.nb[0] = total_size * sizeof(float);
    
    ret = q_add_f32_avx2(&x_tensor, &attn_tensor, &x_residual);
    if (ret != Q_OK) return ret;
    
    // Pre-MLP RMSNorm (need separate buffer for in-place operation)
    float* x_norm_mlp = (float*)q_arena_alloc(ctx, buf_size);
    if (x_norm_mlp == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    ret = q_rmsnorm_f32_avx2(x_norm, (const float*)layer->ffn_norm->data, x_norm_mlp, dim, config->rms_norm_eps);
    if (ret != Q_OK) return ret;
    
    // MLP block
    ret = llama_mlp_forward(layer, ctx, config, x_norm_mlp, mlp_out, seq_len);
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"llama3.c:%d\",\"message\":\"AFTER llama_mlp_forward\",\"data\":{\"ret\":%d,\"ret_is_ok\":%d},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"H\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                    ret, (ret == Q_OK ? 1 : 0));
            fclose(log_file);
        }
    }
    // #endregion
    
    if (ret != Q_OK) return ret;
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"llama3.c:%d\",\"message\":\"BEFORE second residual (x + mlp_out)\",\"data\":{\"seq_len\":%u,\"dim\":%u,\"total_size\":%u,\"x_residual\":\"%p\",\"mlp_out\":\"%p\",\"output\":\"%p\"},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"H\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                    seq_len, dim, seq_len * dim, (void*)x_norm, (void*)mlp_out, (void*)output);
            fclose(log_file);
        }
    }
    // #endregion
    
    // Residual connection: x = x + mlp_out
    // Reuse total_size from first residual (already defined above)
    q_tensor mlp_tensor = {
        .data = (void*)mlp_out,
        .ne = {total_size, 1, 1, 1},
        .nb = {sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    q_tensor output_tensor = {
        .data = (void*)output,
        .ne = {total_size, 1, 1, 1},
        .nb = {sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    // CRITICAL FIX: nb[0] must equal total_size * sizeof(float) for contiguous 1D tensor
    mlp_tensor.nb[0] = total_size * sizeof(float);
    output_tensor.nb[0] = total_size * sizeof(float);
    x_residual.nb[0] = total_size * sizeof(float);
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"llama3.c:%d\",\"message\":\"BEFORE q_add_f32_avx2 (second residual)\",\"data\":{\"x_residual_ne0\":%u,\"x_residual_nb0\":%zu,\"mlp_ne0\":%u,\"mlp_nb0\":%zu,\"out_ne0\":%u,\"out_nb0\":%zu,\"expected_nb0\":%u},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"H\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                    x_residual.ne[0], x_residual.nb[0], mlp_tensor.ne[0], mlp_tensor.nb[0], output_tensor.ne[0], output_tensor.nb[0], (unsigned long)(total_size * sizeof(float)));
            fclose(log_file);
        }
    }
    // #endregion
    
    ret = q_add_f32_avx2(&x_residual, &mlp_tensor, &output_tensor);
    
    // #region agent log
    {
        FILE* log_file = fopen("/home/jcopari-/IA-study/.cursor/debug.log", "a");
        if (log_file) {
            fprintf(log_file, "{\"id\":\"log_%lu_%d\",\"timestamp\":%lu,\"location\":\"llama3.c:%d\",\"message\":\"AFTER q_add_f32_avx2 (second residual)\",\"data\":{\"ret\":%d,\"ret_is_ok\":%d},\"sessionId\":\"debug-session\",\"runId\":\"run1\",\"hypothesisId\":\"H\"}\n",
                    (unsigned long)time(NULL), __LINE__, (unsigned long)(time(NULL) * 1000), __LINE__,
                    ret, (ret == Q_OK ? 1 : 0));
            fclose(log_file);
        }
    }
    // #endregion
    
    if (ret != Q_OK) return ret;
    
    return Q_OK;
}

// Main forward pass function
q_error_code llama_forward(
    llama_model* restrict model,
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
    
    // CRITICAL DEBUG: Print type value before validation (cannot be optimized away)
    // Use volatile to prevent compiler optimization
    volatile q_dtype actual_type = model->token_embd->type;
    volatile q_dtype expected_type = Q_F32;
    volatile q_tensor* token_embd_ptr = model->token_embd;
    
    // DEBUG: Print type before validation (for diagnosis)
    char debug_msg[512];
    int msg_len = snprintf(debug_msg, sizeof(debug_msg),
        "DEBUG: llama_forward: token_embd->type = %d (expected %d)\n"
        "  token_embd pointer: %p (was %p at creation)\n"
        "  token_embd->data: %p\n"
        "  Arena head: %zu, Arena size: %zu\n"
        "  token_embd offset in arena: %zu\n"
        "  sizeof(q_tensor) = %zu\n",
        (int)actual_type, (int)expected_type,
        (void*)token_embd_ptr, (void*)0x7a136c3fe040UL,  // Hardcoded from debug output
        (void*)token_embd_ptr->data,
        ctx->scratch_head, ctx->scratch_size,
        (size_t)((uint8_t*)token_embd_ptr - (uint8_t*)ctx->scratch_buffer),
        sizeof(q_tensor));
    Q_WRITE_STDERR( debug_msg, (size_t)msg_len);
    
    // Verify type is still correct (catches memory corruption)
    // ALWAYS check this, even in Release mode (critical safety check)
    if (actual_type != expected_type) {
        char err_msg[512];
        int err_len = snprintf(err_msg, sizeof(err_msg),
            "ERROR: llama_forward: token_embd->type corrupted!\n"
            "  Expected: %d (Q_F32), Got: %d\n"
            "  token_embd pointer: %p\n"
            "  token_embd->data: %p\n"
            "  Arena head: %zu, Arena size: %zu\n"
            "  token_embd offset in arena: %zu\n",
            (int)Q_F32, (int)actual_type,
            (void*)token_embd_ptr,
            (void*)token_embd_ptr->data,
            ctx->scratch_head, ctx->scratch_size,
            (size_t)((uint8_t*)token_embd_ptr - (uint8_t*)ctx->scratch_buffer));
        Q_WRITE_STDERR( err_msg, (size_t)err_len);
        #ifdef DEBUG
        abort();
        #endif
        return Q_ERR_INVALID_DTYPE;
    }
    
    q_error_code ret = token_embedding_lookup(model->token_embd, tokens, seq_len, x);
    if (ret != Q_OK) {
        char err_msg[128];
        int err_len = snprintf(err_msg, sizeof(err_msg),
            "ERROR: llama_forward: token_embedding_lookup returned %d\n", ret);
        Q_WRITE_STDERR( err_msg, (size_t)err_len);
        return ret;
    }
    
    char success_msg[128];
    int success_len = snprintf(success_msg, sizeof(success_msg),
        "DEBUG: llama_forward: token_embedding_lookup succeeded\n");
    Q_WRITE_STDERR( success_msg, (size_t)success_len);
    
    // Step 2: Forward through layers
    // Allocate temporary buffer for layer output
    size_t layer_buf_size = (size_t)seq_len * (size_t)dim * sizeof(float);
    layer_buf_size = Q_ALIGN_SIZE(layer_buf_size);
    float* layer_out = (float*)q_arena_alloc(ctx, layer_buf_size);
    if (layer_out == NULL) {
        return Q_ERR_ARENA_OOM;
    }
    
    // Process each layer
    for (uint32_t l = 0; l < model->config.n_layers; l++) {
        char layer_msg[128];
        int layer_len = snprintf(layer_msg, sizeof(layer_msg),
            "DEBUG: llama_forward: Calling llama_layer_forward for layer %u\n", l);
        Q_WRITE_STDERR( layer_msg, (size_t)layer_len);
        
        ret = llama_layer_forward(&model->layers[l], ctx, &model->config, x, layer_out, l, seq_len, pos);
        if (ret != Q_OK) {
            char layer_err[128];
            int layer_err_len = snprintf(layer_err, sizeof(layer_err),
                "ERROR: llama_forward: llama_layer_forward[%u] returned %d\n", l, ret);
            Q_WRITE_STDERR( layer_err, (size_t)layer_err_len);
            return ret;
        }
        
        // Swap buffers for next layer (x becomes output of previous layer)
        float* tmp = x;
        x = layer_out;
        layer_out = tmp;
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
    // If x_final is 64-byte aligned and dim*sizeof(float) is multiple of 32,
    // then last_token will be 32-byte aligned. But we need to verify.
    const float* last_token_ptr = x_final + (size_t)(seq_len - 1) * dim;
    uintptr_t last_token_addr = (uintptr_t)last_token_ptr;
    uintptr_t misalignment = last_token_addr % 32;
    
    // If misaligned, allocate aligned buffer and copy
    const float* last_token;
    float* last_token_aligned = NULL;
    if (misalignment != 0) {
        // Allocate aligned buffer
        size_t aligned_size = Q_ALIGN_SIZE(dim * sizeof(float));
        last_token_aligned = (float*)q_arena_alloc(ctx, aligned_size);
        if (last_token_aligned == NULL) {
            return Q_ERR_ARENA_OOM;
        }
        // Copy last token to aligned buffer
        memcpy(last_token_aligned, last_token_ptr, dim * sizeof(float));
        last_token = last_token_aligned;
    } else {
        last_token = last_token_ptr;
    }
    
    // Create tensor view for last token [1, dim]
    q_tensor last_token_tensor = {
        .data = (void*)last_token,
        .ne = {1, dim, 1, 1},
        .nb = {dim * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
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
    
    // Create tensor view for logits [1, vocab_size]
    q_tensor logits_tensor = {
        .data = (void*)logits,
        .ne = {1, vocab_size, 1, 1},
        .nb = {vocab_size * sizeof(float), sizeof(float), sizeof(float), sizeof(float)},
        .type = Q_F32
    };
    
    // Compute: last_token [1, dim] @ output^T [dim, vocab_size] -> logits [1, vocab_size]
    ret = q_matmul_f32_avx2(&last_token_tensor, &output_t_tensor, &logits_tensor, ctx);
    if (ret != Q_OK) {
        return ret;
    }
    
    return Q_OK;
}
