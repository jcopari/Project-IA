#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>

// TDD Test for FASE 3.3: llama_forward()
// Tests forward pass execution through Llama-3 model
// Following MFR + CoT + Proof + TDD methodology

// Test configuration (will use actual model config)
// These are just for KV cache size calculation
#define TEST_N_LAYERS 2

// Helper: Compare two float arrays with tolerance
static bool float_array_eq(const float* a, const float* b, size_t n, float abs_tol, float rel_tol) {
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        float max_val = fmaxf(fabsf(a[i]), fabsf(b[i]));
        if (diff > abs_tol && diff > rel_tol * max_val) {
            printf("  Mismatch at index %zu: expected %.6f, got %.6f (diff: %.6f)\n",
                   i, b[i], a[i], diff);
            return false;
        }
    }
    return true;
}

// Helper: Initialize KV cache with zeros
static void init_kv_cache(q_context* ctx, const llama_config* config) {
    if (ctx->kv_buffer == NULL) {
        return;
    }
    
    uint32_t head_dim = config->dim / config->n_heads;
    size_t kv_size = (size_t)config->n_layers * 
                     (size_t)config->n_kv_heads * 
                     (size_t)config->max_seq_len * 
                     (size_t)head_dim * 
                     sizeof(float) * 2; // *2 for K and V
    
    memset(ctx->kv_buffer, 0, kv_size);
}

int main(void) {
    printf("=== TDD Test: llama_forward() ===\n\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    q_error_code ret;
    
    // Test 1: Load model file
    printf("Test 1: Loading model file...\n");
    ret = q_init_memory(&ctx, "model_dummy.qorus");
    if (ret != Q_OK) {
        printf("  ERROR: Failed to load model (code: %d, %s)\n", ret, q_strerror(ret));
        printf("  Execute first: python3 tools/convert_llama.py model_dummy.qorus %d\n", TEST_N_LAYERS);
        return 1;
    }
    printf("  ✓ Model loaded\n");
    
    // Test 2: Allocate KV cache
    printf("\nTest 2: Allocating KV cache...\n");
    // Use actual model config for KV cache size
    uint32_t n_kv_heads = ctx.header->n_kv_heads;
    uint32_t max_seq_len = ctx.header->max_seq_len;
    uint32_t head_dim = ctx.header->dim / ctx.header->n_heads;
    uint32_t n_layers = ctx.header->n_layers;
    
    size_t kv_size = (size_t)n_layers * 
                     (size_t)n_kv_heads * 
                     (size_t)max_seq_len * 
                     (size_t)head_dim * 
                     sizeof(float) * 2; // *2 for K and V
    
    ret = q_alloc_kv_cache(&ctx, kv_size);
    if (ret != Q_OK) {
        printf("  ERROR: Failed to allocate KV cache (code: %d, %s)\n", ret, q_strerror(ret));
        q_free_memory(&ctx);
        return 1;
    }
    printf("  ✓ KV cache allocated (%zu bytes)\n", kv_size);
    
    // Test 3: Allocate arena
    printf("\nTest 3: Allocating arena...\n");
    // CRITICAL: Arena size must be large enough for:
    // 1. Model structures (q_tensor views) - ~few KB
    // 2. Multiple forward pass temporary buffers - ~few MB per call
    // For this test with 3 forward passes, we need at least 32MB
    ret = q_alloc_arena(&ctx, 32 * 1024 * 1024);  // 32MB arena (increased from 16MB)
    if (ret != Q_OK) {
        printf("  ERROR: Failed to allocate arena (code: %d, %s)\n", ret, q_strerror(ret));
        q_free_memory(&ctx);
        return 1;
    }
    printf("  ✓ Arena allocated (32MB)\n");
    
    // Test 4: Build model graph
    printf("\nTest 4: Building model graph...\n");
    ret = llama_build_graph(&ctx, &model);
    if (ret != Q_OK) {
        printf("  ERROR: Failed to build graph (code: %d, %s)\n", ret, q_strerror(ret));
        q_free_memory(&ctx);
        return 1;
    }
    printf("  ✓ Model graph built\n");
    
    // Test 5: Validate model configuration matches test config
    printf("\nTest 5: Validating model configuration...\n");
    if (model.config.vocab_size != ctx.header->vocab_size ||
        model.config.dim != ctx.header->dim ||
        model.config.hidden_dim != ctx.header->hidden_dim ||
        model.config.n_layers != ctx.header->n_layers ||
        model.config.n_heads != ctx.header->n_heads ||
        model.config.n_kv_heads != ctx.header->n_kv_heads ||
        model.config.max_seq_len != ctx.header->max_seq_len) {
        printf("  ERROR: Model config mismatch\n");
        printf("    Expected: vocab=%u, dim=%u, hidden_dim=%u, n_layers=%u, n_heads=%u, n_kv_heads=%u, max_seq_len=%u\n",
               ctx.header->vocab_size, ctx.header->dim, ctx.header->hidden_dim,
               ctx.header->n_layers, ctx.header->n_heads, ctx.header->n_kv_heads,
               ctx.header->max_seq_len);
        printf("    Got: vocab=%u, dim=%u, hidden_dim=%u, n_layers=%u, n_heads=%u, n_kv_heads=%u, max_seq_len=%u\n",
               model.config.vocab_size, model.config.dim, model.config.hidden_dim,
               model.config.n_layers, model.config.n_heads, model.config.n_kv_heads,
               model.config.max_seq_len);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    
    // CRITICAL: Validate Q4_0 requirements
    if (model.config.dim % 32 != 0) {
        printf("  ERROR: dim (%u) must be multiple of 32 for Q4_0 weights\n", model.config.dim);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    if (model.config.hidden_dim % 32 != 0) {
        printf("  ERROR: hidden_dim (%u) must be multiple of 32 for Q4_0 weights\n", model.config.hidden_dim);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    printf("  ✓ Configuration validated\n");
    
    // Initialize KV cache
    init_kv_cache(&ctx, &model.config);
    
    // Test 6: Forward pass with single token (incremental generation)
    printf("\nTest 6: Forward pass - single token (incremental generation)...\n");
    
    // CRITICAL: Validate model structures are still valid before use
    if (model.token_embd == NULL) {
        printf("  ERROR: model.token_embd is NULL\n");
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    if (model.token_embd->type != Q_F32) {
        printf("  ERROR: token_embd->type is not Q_F32 (got %d)\n", model.token_embd->type);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    if (model.token_embd->ne[0] != model.config.vocab_size || 
        model.token_embd->ne[1] != model.config.dim) {
        printf("  ERROR: token_embd dimensions mismatch: expected [%u,%u], got [%u,%u]\n",
               model.config.vocab_size, model.config.dim,
               model.token_embd->ne[0], model.token_embd->ne[1]);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    
    uint32_t tokens[1] = {0};  // Token ID 0
    uint32_t vocab_size = model.config.vocab_size;
    uint32_t seq_len = 1;
    
    // CRITICAL: Validate arena has sufficient space before forward pass
    // Estimate required space: seq_len * dim * sizeof(float) * (multiple buffers)
    size_t estimated_space = (size_t)seq_len * (size_t)model.config.dim * sizeof(float) * 10; // Conservative estimate
    estimated_space = Q_ALIGN_SIZE(estimated_space);
    size_t available_space = ctx.scratch_size - ctx.scratch_head;
    if (available_space < estimated_space) {
        printf("  WARNING: Arena space may be tight: available=%zu, estimated=%zu\n", 
               available_space, estimated_space);
        printf("  NOTE: Arena cannot be reset because it contains model structures\n");
    }
    
    float* logits = (float*)aligned_alloc(32, vocab_size * sizeof(float));
    if (logits == NULL) {
        printf("  ERROR: Failed to allocate logits buffer\n");
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    // CRITICAL: Validate alignment (AVX2 requires 32-byte alignment)
    if (((uintptr_t)logits % 32) != 0) {
        printf("  ERROR: logits buffer not properly aligned (address: %p)\n", (void*)logits);
        free(logits);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    memset(logits, 0, vocab_size * sizeof(float));
    
    // NOTE: Do NOT reset arena here - it contains model structures (q_tensor views)
    // The forward pass will allocate temporary buffers after the model structures
    
    ret = llama_forward(&model, &ctx, tokens, seq_len, 0, logits);
    if (ret != Q_OK) {
        printf("  ERROR: Forward pass failed (code: %d, %s)\n", ret, q_strerror(ret));
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    printf("  ✓ Forward pass completed\n");
    
    // Validate logits (mathematical properties)
    bool has_finite = false;
    bool has_positive = false;
    bool has_negative = false;
    float max_logit = -FLT_MAX;
    float min_logit = FLT_MAX;
    float sum_logits = 0.0f;
    uint32_t finite_count = 0;
    
    for (uint32_t i = 0; i < model.config.vocab_size; i++) {
        if (isfinite(logits[i])) {
            has_finite = true;
            finite_count++;
            sum_logits += logits[i];
            
            if (logits[i] > max_logit) {
                max_logit = logits[i];
            }
            if (logits[i] < min_logit) {
                min_logit = logits[i];
            }
            if (logits[i] > 0.0f) {
                has_positive = true;
            }
            if (logits[i] < 0.0f) {
                has_negative = true;
            }
        }
    }
    
    if (!has_finite) {
        printf("  ERROR: All logits are non-finite (NaN or Inf)\n");
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    printf("  ✓ Logits are finite (%u/%u finite values)\n", finite_count, model.config.vocab_size);
    
    if (finite_count < model.config.vocab_size) {
        printf("  WARNING: Some logits are non-finite (%u non-finite)\n", 
               model.config.vocab_size - finite_count);
    }
    
    printf("  ✓ Logits statistics:\n");
    printf("    Range: [%.6f, %.6f]\n", min_logit, max_logit);
    printf("    Mean: %.6f\n", sum_logits / finite_count);
    printf("    Has positive: %s\n", has_positive ? "yes" : "no");
    printf("    Has negative: %s\n", has_negative ? "yes" : "no");
    
    // Validate that logits have reasonable distribution
    // (not all zeros, not all same value)
    float logit_variance = 0.0f;
    float logit_mean = sum_logits / finite_count;
    for (uint32_t i = 0; i < model.config.vocab_size; i++) {
        if (isfinite(logits[i])) {
            float diff = logits[i] - logit_mean;
            logit_variance += diff * diff;
        }
    }
    logit_variance /= finite_count;
    float logit_std = sqrtf(logit_variance);
    
    printf("    Std: %.6f\n", logit_std);
    
    if (logit_std < 1e-6f) {
        printf("  WARNING: Logits have very low variance (may indicate issue)\n");
    }
    
    // Test 7: Forward pass with multiple tokens (prefill)
    printf("\nTest 7: Forward pass - multiple tokens (prefill)...\n");
    
    // CRITICAL: Validate model structures are still valid before use
    if (model.token_embd == NULL || model.token_embd->type != Q_F32) {
        printf("  ERROR: Model structures corrupted\n");
        free(logits);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    
    uint32_t tokens_prefill[4] = {0, 1, 2, 3};
    uint32_t seq_len_prefill = 4;
    
    // CRITICAL: Validate arena has sufficient space before forward pass
    size_t estimated_space_prefill = (size_t)seq_len_prefill * (size_t)model.config.dim * sizeof(float) * 10;
    estimated_space_prefill = Q_ALIGN_SIZE(estimated_space_prefill);
    size_t available_space_prefill = ctx.scratch_size - ctx.scratch_head;
    if (available_space_prefill < estimated_space_prefill) {
        printf("  WARNING: Arena space may be tight: available=%zu, estimated=%zu\n", 
               available_space_prefill, estimated_space_prefill);
    }
    
    float* logits_prefill = (float*)aligned_alloc(32, vocab_size * sizeof(float));
    if (logits_prefill == NULL) {
        printf("  ERROR: Failed to allocate logits_prefill buffer\n");
        free(logits);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    // CRITICAL: Validate alignment (AVX2 requires 32-byte alignment)
    if (((uintptr_t)logits_prefill % 32) != 0) {
        printf("  ERROR: logits_prefill buffer not properly aligned (address: %p)\n", (void*)logits_prefill);
        free(logits_prefill);
        free(logits);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    memset(logits_prefill, 0, vocab_size * sizeof(float));
    
    // NOTE: Do NOT reset arena - model structures must persist
    
    ret = llama_forward(&model, &ctx, tokens_prefill, seq_len_prefill, 0, logits_prefill);
    if (ret != Q_OK) {
        printf("  ERROR: Forward pass failed (code: %d, %s)\n", ret, q_strerror(ret));
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    printf("  ✓ Forward pass completed\n");
    
    // Validate logits
    has_finite = false;
    for (uint32_t i = 0; i < model.config.vocab_size; i++) {
        if (isfinite(logits_prefill[i])) {
            has_finite = true;
            break;
        }
    }
    
    if (!has_finite) {
        printf("  ERROR: All logits are non-finite\n");
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    printf("  ✓ Logits are finite\n");
    
    // Test 8: Incremental generation (second token)
    printf("\nTest 8: Incremental generation - second token...\n");
    
    // CRITICAL: Validate model structures are still valid before use
    if (model.token_embd == NULL || model.token_embd->type != Q_F32) {
        printf("  ERROR: Model structures corrupted\n");
        free(logits_prefill);
        free(logits);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    
    uint32_t token_incr[1] = {4};
    uint32_t seq_len_incr = 1;
    
    // CRITICAL: Validate arena has sufficient space before forward pass
    size_t estimated_space_incr = (size_t)seq_len_incr * (size_t)model.config.dim * sizeof(float) * 10;
    estimated_space_incr = Q_ALIGN_SIZE(estimated_space_incr);
    size_t available_space_incr = ctx.scratch_size - ctx.scratch_head;
    if (available_space_incr < estimated_space_incr) {
        printf("  WARNING: Arena space may be tight: available=%zu, estimated=%zu\n", 
               available_space_incr, estimated_space_incr);
    }
    
    float* logits_incr = (float*)aligned_alloc(32, vocab_size * sizeof(float));
    if (logits_incr == NULL) {
        printf("  ERROR: Failed to allocate logits_incr buffer\n");
        free(logits_prefill);
        free(logits);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    // CRITICAL: Validate alignment (AVX2 requires 32-byte alignment)
    if (((uintptr_t)logits_incr % 32) != 0) {
        printf("  ERROR: logits_incr buffer not properly aligned (address: %p)\n", (void*)logits_incr);
        free(logits_incr);
        free(logits_prefill);
        free(logits);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    memset(logits_incr, 0, vocab_size * sizeof(float));
    
    // NOTE: Do NOT reset arena - model structures must persist
    
    // Position 1 (second token, KV cache should have first token)
    ret = llama_forward(&model, &ctx, token_incr, seq_len_incr, 1, logits_incr);
    if (ret != Q_OK) {
        printf("  ERROR: Forward pass failed (code: %d, %s)\n", ret, q_strerror(ret));
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    printf("  ✓ Incremental forward pass completed\n");
    
    // Test 9: Error handling - NULL model
    printf("\nTest 9: Error handling - NULL model...\n");
    ret = llama_forward(NULL, &ctx, tokens, 1, 0, logits);
    if (ret == Q_OK) {
        printf("  ERROR: Should return error for NULL model\n");
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    printf("  ✓ Correctly returned error (code: %d, %s)\n", ret, q_strerror(ret));
    
    // Test 10: Error handling - NULL context
    printf("\nTest 10: Error handling - NULL context...\n");
    ret = llama_forward(&model, NULL, tokens, 1, 0, logits);
    if (ret == Q_OK) {
        printf("  ERROR: Should return error for NULL context\n");
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    printf("  ✓ Correctly returned error (code: %d, %s)\n", ret, q_strerror(ret));
    
    // Test 11: Error handling - NULL tokens
    printf("\nTest 11: Error handling - NULL tokens...\n");
    ret = llama_forward(&model, &ctx, NULL, 1, 0, logits);
    if (ret == Q_OK) {
        printf("  ERROR: Should return error for NULL tokens\n");
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    printf("  ✓ Correctly returned error (code: %d, %s)\n", ret, q_strerror(ret));
    
    // Test 12: Error handling - NULL logits
    printf("\nTest 12: Error handling - NULL logits...\n");
    ret = llama_forward(&model, &ctx, tokens, 1, 0, NULL);
    if (ret == Q_OK) {
        printf("  ERROR: Should return error for NULL logits\n");
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    printf("  ✓ Correctly returned error (code: %d, %s)\n", ret, q_strerror(ret));
    
    // Test 13: Error handling - zero sequence length
    printf("\nTest 13: Error handling - zero sequence length...\n");
    ret = llama_forward(&model, &ctx, tokens, 0, 0, logits);
    if (ret == Q_OK) {
        printf("  ERROR: Should return error for zero sequence length\n");
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    printf("  ✓ Correctly returned error (code: %d, %s)\n", ret, q_strerror(ret));
    
    // Test 14: Error handling - invalid position
    printf("\nTest 14: Error handling - invalid position...\n");
    ret = llama_forward(&model, &ctx, tokens, 1, model.config.max_seq_len, logits);
    if (ret == Q_OK) {
        printf("  ERROR: Should return error for invalid position\n");
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    printf("  ✓ Correctly returned error (code: %d, %s)\n", ret, q_strerror(ret));
    
    // Cleanup
    printf("\n=== Cleanup ===\n");
    free(logits_incr);
    free(logits_prefill);
    free(logits);
    llama_free_graph(&model);
    q_free_memory(&ctx);
    printf("✓ All tests passed!\n");
    
    return 0;
}

