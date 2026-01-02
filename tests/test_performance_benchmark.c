// ============================================================================
// TEST: Performance Benchmark Suite
// ============================================================================
// Testes de performance para medir o poder do modelo atual
// Seguindo protocolo /gereteste.md adaptado para benchmarks
//
// Métricas Medidas:
// - Latência por token (prefill e incremental)
// - Throughput (tokens/segundo)
// - Uso de memória (arena, KV cache)
// - Comparação de estratégias de sampling (greedy, top-k, top-p)
// - Impacto de tamanhos de prompt diferentes
// ============================================================================

#include "qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Helper: Get current time in microseconds
static uint64_t get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

// Helper: Get current time in milliseconds
static double get_time_ms(void) {
    return (double)get_time_us() / 1000.0;
}

// Helper: Ensure dummy model exists
static bool ensure_dummy_model(void) {
    FILE* f = fopen("model_dummy.qorus", "rb");
    if (f != NULL) {
        fclose(f);
        return true;
    }
    
    printf("  Generating dummy model...\n");
    int ret = system("python3 tools/convert_llama.py model_dummy.qorus 2 > /dev/null 2>&1");
    return (ret == 0);
}

// Helper: Ensure tokenizer exists
static bool ensure_tokenizer(void) {
    FILE* f = fopen("tokenizer.bin", "rb");
    if (f != NULL) {
        fclose(f);
        return true;
    }
    
    printf("  Generating tokenizer...\n");
    int ret = system("python3 tools/convert_llama.py --tokenizer tokenizer.bin > /dev/null 2>&1");
    return (ret == 0);
}

// Helper: Calculate KV cache size
static size_t calculate_kv_cache_size(const q_llama_config* config) {
    uint32_t head_dim = config->dim / config->n_heads;
    size_t kv_size = (size_t)config->n_layers * 
                     (size_t)config->n_kv_heads * 
                     (size_t)config->max_seq_len * 
                     (size_t)head_dim * 
                     sizeof(float) * 2; // K + V
    return Q_ALIGN_SIZE(kv_size);
}

// Helper: Setup model and context
static bool setup_model(q_context* ctx, q_llama_model* model, q_tokenizer* tok) {
    memset(ctx, 0, sizeof(q_context));
    memset(model, 0, sizeof(q_llama_model));
    memset(tok, 0, sizeof(q_tokenizer));
    
    q_error_code ret = q_init_memory(ctx, "model_dummy.qorus");
    if (ret != Q_OK) {
        fprintf(stderr, "ERROR: q_init_memory failed: %d\n", ret);
        return false;
    }
    
    ret = q_tokenizer_load(tok, "tokenizer.bin");
    if (ret != Q_OK) {
        fprintf(stderr, "ERROR: q_tokenizer_load failed: %d\n", ret);
        q_free_memory(ctx);
        return false;
    }
    
    ret = q_alloc_arena(ctx, 64 * 1024 * 1024); // 64 MB arena
    if (ret != Q_OK) {
        fprintf(stderr, "ERROR: q_alloc_arena failed: %d\n", ret);
        q_tokenizer_free(tok);
        q_free_memory(ctx);
        return false;
    }
    
    ret = llama_build_graph(ctx, model);
    if (ret != Q_OK) {
        fprintf(stderr, "ERROR: llama_build_graph failed: %d\n", ret);
        q_tokenizer_free(tok);
        q_free_memory(ctx);
        return false;
    }
    
    size_t kv_size = calculate_kv_cache_size(&model->config);
    ret = q_alloc_kv_cache(ctx, kv_size);
    if (ret != Q_OK) {
        fprintf(stderr, "ERROR: q_alloc_kv_cache failed: %d\n", ret);
        llama_free_graph(model);
        q_tokenizer_free(tok);
        q_free_memory(ctx);
        return false;
    }
    
    return true;
}

// Helper: Cleanup
static void cleanup_model(q_context* ctx, q_llama_model* model, q_tokenizer* tok) {
    if (model != NULL && (model->token_embd != NULL || model->layers != NULL)) {
        llama_free_graph(model);
    }
    if (tok != NULL && (tok->vocab != NULL || tok->merges != NULL)) {
        q_tokenizer_free(tok);
    }
    if (ctx != NULL && (ctx->weights_mmap != NULL || ctx->scratch_buffer != NULL || ctx->kv_buffer != NULL)) {
        q_free_memory(ctx);
    }
}

// ============================================================================
// PERFORMANCE METRICS STRUCTURE
// ============================================================================

typedef struct {
    double prefill_time_ms;           // Tempo total do prefill
    double prefill_time_per_token_ms; // Tempo por token no prefill
    double incremental_time_ms;      // Tempo total da geração incremental
    double incremental_time_per_token_ms; // Tempo por token incremental
    double total_time_ms;             // Tempo total (prefill + incremental)
    double throughput_tokens_per_sec; // Throughput em tokens/segundo
    size_t memory_arena_bytes;        // Uso de memória da arena
    size_t memory_kv_cache_bytes;     // Uso de memória do KV cache
    uint32_t num_tokens_generated;   // Número de tokens gerados
} performance_metrics;

// ============================================================================
// BENCHMARK FUNCTIONS
// ============================================================================

// Benchmark: Prefill performance
static performance_metrics benchmark_prefill(
    q_context* ctx,
    q_llama_model* model,
    const uint32_t* prompt_tokens,
    uint32_t num_prompt_tokens
) {
    performance_metrics metrics = {0};
    
    // Reset arena
    q_arena_reset(ctx);
    
    // Allocate logits
    size_t logits_size = Q_ALIGN_SIZE((size_t)model->config.vocab_size * sizeof(float));
    float* logits = (float*)aligned_alloc(Q_ALIGN, logits_size);
    if (logits == NULL) {
        return metrics;
    }
    
    // Measure prefill time
    double start_time = get_time_ms();
    
    q_error_code ret = llama_forward(
        model,
        ctx,
        prompt_tokens,
        num_prompt_tokens,
        0, // pos = 0 for prefill
        logits
    );
    
    double end_time = get_time_ms();
    
    if (ret == Q_OK) {
        metrics.prefill_time_ms = end_time - start_time;
        metrics.prefill_time_per_token_ms = metrics.prefill_time_ms / (double)num_prompt_tokens;
    }
    
    free(logits);
    return metrics;
}

// Benchmark: Incremental generation performance
static performance_metrics benchmark_incremental(
    q_context* ctx,
    q_llama_model* model,
    q_tokenizer* tok,
    const uint32_t* prompt_tokens,
    uint32_t num_prompt_tokens,
    uint32_t num_tokens_to_generate,
    float temperature,
    uint32_t top_k,
    float top_p
) {
    performance_metrics metrics = {0};
    
    // Setup generation state
    q_generation_state state = {0};
    state.ctx = ctx;
    state.model = model;
    state.tokenizer = tok;
    state.prompt_tokens = (uint32_t*)prompt_tokens;
    state.num_prompt_tokens = num_prompt_tokens;
    state.generated_tokens = (uint32_t*)malloc(num_tokens_to_generate * sizeof(uint32_t));
    state.max_tokens = num_tokens_to_generate;
    state.temperature = temperature;
    state.top_k = top_k;
    state.top_p = top_p;
    
    if (state.generated_tokens == NULL) {
        return metrics;
    }
    
    // Measure incremental generation time
    double start_time = get_time_ms();
    
    q_error_code ret = q_generate(&state);
    
    double end_time = get_time_ms();
    
    if (ret == Q_OK) {
        metrics.incremental_time_ms = end_time - start_time;
        metrics.num_tokens_generated = state.num_generated_tokens;
        if (metrics.num_tokens_generated > 0) {
            metrics.incremental_time_per_token_ms = metrics.incremental_time_ms / (double)metrics.num_tokens_generated;
            metrics.throughput_tokens_per_sec = 1000.0 / metrics.incremental_time_per_token_ms;
        }
        metrics.total_time_ms = metrics.incremental_time_ms;
    }
    
    free(state.generated_tokens);
    return metrics;
}

// Benchmark: Full pipeline (prefill + incremental)
static performance_metrics benchmark_full_pipeline(
    q_context* ctx,
    q_llama_model* model,
    q_tokenizer* tok,
    const uint32_t* prompt_tokens,
    uint32_t num_prompt_tokens,
    uint32_t num_tokens_to_generate,
    float temperature,
    uint32_t top_k,
    float top_p
) {
    performance_metrics metrics = {0};
    
    // Prefill
    performance_metrics prefill_metrics = benchmark_prefill(ctx, model, prompt_tokens, num_prompt_tokens);
    metrics.prefill_time_ms = prefill_metrics.prefill_time_ms;
    metrics.prefill_time_per_token_ms = prefill_metrics.prefill_time_per_token_ms;
    
    // Incremental generation
    performance_metrics inc_metrics = benchmark_incremental(
        ctx, model, tok, prompt_tokens, num_prompt_tokens,
        num_tokens_to_generate, temperature, top_k, top_p
    );
    
    metrics.incremental_time_ms = inc_metrics.incremental_time_ms;
    metrics.incremental_time_per_token_ms = inc_metrics.incremental_time_per_token_ms;
    metrics.num_tokens_generated = inc_metrics.num_tokens_generated;
    metrics.throughput_tokens_per_sec = inc_metrics.throughput_tokens_per_sec;
    metrics.total_time_ms = metrics.prefill_time_ms + metrics.incremental_time_ms;
    
    // Memory usage
    metrics.memory_arena_bytes = ctx->scratch_size;
    metrics.memory_kv_cache_bytes = calculate_kv_cache_size(&model->config);
    
    return metrics;
}

// ============================================================================
// TEST CASES
// ============================================================================

// Test 1: Prefill Performance Benchmark
static void test_prefill_performance(void) {
    printf("========================================\n");
    printf("TEST 1: Prefill Performance Benchmark\n");
    printf("========================================\n\n");
    
    if (!ensure_dummy_model() || !ensure_tokenizer()) {
        printf("❌ Cannot generate dummy model or tokenizer\n");
        return;
    }
    
    q_context ctx = {0};
    q_llama_model model = {0};
    q_tokenizer tok = {0};
    
    if (!setup_model(&ctx, &model, &tok)) {
        printf("❌ Failed to setup model\n");
        return;
    }
    
    // Encode prompt
    const char* prompt_text = "Hello, how are you?";
    uint32_t prompt_tokens[100];
    uint32_t num_prompt_tokens = 0;
    
    q_error_code ret = q_tokenizer_encode(&tok, prompt_text, prompt_tokens, &num_prompt_tokens, 100, true, true);
    if (ret != Q_OK) {
        printf("❌ Failed to encode prompt: %d\n", ret);
        cleanup_model(&ctx, &model, &tok);
        return;
    }
    
    printf("Prompt: \"%s\" (%u tokens)\n", prompt_text, num_prompt_tokens);
    printf("Model: %u layers, %u dim, vocab_size=%u\n\n", 
           model.config.n_layers, model.config.dim, model.config.vocab_size);
    
    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < 3; i++) {
        benchmark_prefill(&ctx, &model, prompt_tokens, num_prompt_tokens);
    }
    
    // Benchmark
    printf("Running benchmark (10 iterations)...\n");
    double total_time = 0.0;
    double total_time_per_token = 0.0;
    const int iterations = 10;
    
    for (int i = 0; i < iterations; i++) {
        performance_metrics m = benchmark_prefill(&ctx, &model, prompt_tokens, num_prompt_tokens);
        total_time += m.prefill_time_ms;
        total_time_per_token += m.prefill_time_per_token_ms;
    }
    
    double avg_time = total_time / iterations;
    double avg_time_per_token = total_time_per_token / iterations;
    double throughput = 1000.0 / avg_time_per_token;
    
    printf("\n--- Results ---\n");
    printf("Average Prefill Time: %.3f ms\n", avg_time);
    printf("Average Time per Token: %.3f ms/token\n", avg_time_per_token);
    printf("Throughput: %.2f tokens/s\n", throughput);
    printf("✓ Test 1 PASSED\n\n");
    
    cleanup_model(&ctx, &model, &tok);
}

// Test 2: Incremental Generation Performance Benchmark
static void test_incremental_performance(void) {
    printf("========================================\n");
    printf("TEST 2: Incremental Generation Performance\n");
    printf("========================================\n\n");
    
    if (!ensure_dummy_model() || !ensure_tokenizer()) {
        printf("❌ Cannot generate dummy model or tokenizer\n");
        return;
    }
    
    q_context ctx = {0};
    q_llama_model model = {0};
    q_tokenizer tok = {0};
    
    if (!setup_model(&ctx, &model, &tok)) {
        printf("❌ Failed to setup model\n");
        return;
    }
    
    // Encode prompt
    const char* prompt_text = "Hello, how are you?";
    uint32_t prompt_tokens[100];
    uint32_t num_prompt_tokens = 0;
    
    q_error_code ret = q_tokenizer_encode(&tok, prompt_text, prompt_tokens, &num_prompt_tokens, 100, true, true);
    if (ret != Q_OK) {
        printf("❌ Failed to encode prompt: %d\n", ret);
        cleanup_model(&ctx, &model, &tok);
        return;
    }
    
    printf("Prompt: \"%s\" (%u tokens)\n", prompt_text, num_prompt_tokens);
    printf("Generating: 10 tokens\n");
    printf("Sampling: Greedy (temperature=0.0)\n\n");
    
    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < 3; i++) {
        benchmark_incremental(&ctx, &model, &tok, prompt_tokens, num_prompt_tokens, 10, 0.0f, 0, 0.0f);
    }
    
    // Benchmark
    printf("Running benchmark (10 iterations)...\n");
    double total_time = 0.0;
    double total_time_per_token = 0.0;
    double total_throughput = 0.0;
    const int iterations = 10;
    
    for (int i = 0; i < iterations; i++) {
        performance_metrics m = benchmark_incremental(&ctx, &model, &tok, prompt_tokens, num_prompt_tokens, 10, 0.0f, 0, 0.0f);
        total_time += m.incremental_time_ms;
        total_time_per_token += m.incremental_time_per_token_ms;
        total_throughput += m.throughput_tokens_per_sec;
    }
    
    double avg_time = total_time / iterations;
    double avg_time_per_token = total_time_per_token / iterations;
    double avg_throughput = total_throughput / iterations;
    
    printf("\n--- Results ---\n");
    printf("Average Incremental Time: %.3f ms\n", avg_time);
    printf("Average Time per Token: %.3f ms/token\n", avg_time_per_token);
    printf("Average Throughput: %.2f tokens/s\n", avg_throughput);
    printf("✓ Test 2 PASSED\n\n");
    
    cleanup_model(&ctx, &model, &tok);
}

// Test 3: Sampling Strategy Comparison
static void test_sampling_strategies(void) {
    printf("========================================\n");
    printf("TEST 3: Sampling Strategy Comparison\n");
    printf("========================================\n\n");
    
    if (!ensure_dummy_model() || !ensure_tokenizer()) {
        printf("❌ Cannot generate dummy model or tokenizer\n");
        return;
    }
    
    q_context ctx = {0};
    q_llama_model model = {0};
    q_tokenizer tok = {0};
    
    if (!setup_model(&ctx, &model, &tok)) {
        printf("❌ Failed to setup model\n");
        return;
    }
    
    // Encode prompt
    const char* prompt_text = "Hello, how are you?";
    uint32_t prompt_tokens[100];
    uint32_t num_prompt_tokens = 0;
    
    q_error_code ret = q_tokenizer_encode(&tok, prompt_text, prompt_tokens, &num_prompt_tokens, 100, true, true);
    if (ret != Q_OK) {
        printf("❌ Failed to encode prompt: %d\n", ret);
        cleanup_model(&ctx, &model, &tok);
        return;
    }
    
    printf("Prompt: \"%s\" (%u tokens)\n", prompt_text, num_prompt_tokens);
    printf("Generating: 10 tokens\n\n");
    
    struct {
        const char* name;
        float temperature;
        uint32_t top_k;
        float top_p;
    } strategies[] = {
        {"Greedy", 0.0f, 0, 0.0f},
        {"Temperature=1.0", 1.0f, 0, 0.0f},
        {"Top-k=10", 1.0f, 10, 0.0f},
        {"Top-p=0.9", 1.0f, 0, 0.9f},
        {"Top-k=10 + Top-p=0.9", 1.0f, 10, 0.9f},
    };
    
    const int num_strategies = sizeof(strategies) / sizeof(strategies[0]);
    const int iterations = 5;
    
    printf("Running benchmarks...\n\n");
    
    for (int s = 0; s < num_strategies; s++) {
        printf("--- %s ---\n", strategies[s].name);
        
        // Warmup
        for (int i = 0; i < 2; i++) {
            benchmark_incremental(&ctx, &model, &tok, prompt_tokens, num_prompt_tokens, 
                                10, strategies[s].temperature, strategies[s].top_k, strategies[s].top_p);
        }
        
        // Benchmark
        double total_time_per_token = 0.0;
        double total_throughput = 0.0;
        
        for (int i = 0; i < iterations; i++) {
            performance_metrics m = benchmark_incremental(&ctx, &model, &tok, prompt_tokens, num_prompt_tokens,
                                                        10, strategies[s].temperature, strategies[s].top_k, strategies[s].top_p);
            total_time_per_token += m.incremental_time_per_token_ms;
            total_throughput += m.throughput_tokens_per_sec;
        }
        
        double avg_time_per_token = total_time_per_token / iterations;
        double avg_throughput = total_throughput / iterations;
        
        printf("  Time per Token: %.3f ms/token\n", avg_time_per_token);
        printf("  Throughput: %.2f tokens/s\n", avg_throughput);
        printf("\n");
    }
    
    printf("✓ Test 3 PASSED\n\n");
    
    cleanup_model(&ctx, &model, &tok);
}

// Test 4: Prompt Size Impact
static void test_prompt_size_impact(void) {
    printf("========================================\n");
    printf("TEST 4: Prompt Size Impact\n");
    printf("========================================\n\n");
    
    if (!ensure_dummy_model() || !ensure_tokenizer()) {
        printf("❌ Cannot generate dummy model or tokenizer\n");
        return;
    }
    
    q_context ctx = {0};
    q_llama_model model = {0};
    q_tokenizer tok = {0};
    
    if (!setup_model(&ctx, &model, &tok)) {
        printf("❌ Failed to setup model\n");
        return;
    }
    
    const char* prompts[] = {
        "Hello",
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.",
    };
    
    const int num_prompts = sizeof(prompts) / sizeof(prompts[0]);
    const int iterations = 5;
    
    printf("Testing with different prompt sizes...\n\n");
    
    for (int p = 0; p < num_prompts; p++) {
        uint32_t prompt_tokens[200];
        uint32_t num_prompt_tokens = 0;
        
        q_error_code ret = q_tokenizer_encode(&tok, prompts[p], prompt_tokens, &num_prompt_tokens, 200, true, true);
        if (ret != Q_OK) {
            printf("❌ Failed to encode prompt %d: %d\n", p, ret);
            continue;
        }
        
        printf("--- Prompt %d: \"%s\" (%u tokens) ---\n", p + 1, prompts[p], num_prompt_tokens);
        
        // Warmup
        for (int i = 0; i < 2; i++) {
            benchmark_prefill(&ctx, &model, prompt_tokens, num_prompt_tokens);
        }
        
        // Benchmark prefill
        double total_prefill_time = 0.0;
        double total_prefill_time_per_token = 0.0;
        
        for (int i = 0; i < iterations; i++) {
            performance_metrics m = benchmark_prefill(&ctx, &model, prompt_tokens, num_prompt_tokens);
            total_prefill_time += m.prefill_time_ms;
            total_prefill_time_per_token += m.prefill_time_per_token_ms;
        }
        
        double avg_prefill_time = total_prefill_time / iterations;
        double avg_prefill_time_per_token = total_prefill_time_per_token / iterations;
        
        printf("  Prefill Time: %.3f ms (%.3f ms/token)\n", avg_prefill_time, avg_prefill_time_per_token);
        printf("\n");
    }
    
    printf("✓ Test 4 PASSED\n\n");
    
    cleanup_model(&ctx, &model, &tok);
}

// Test 5: Full Pipeline Performance
static void test_full_pipeline_performance(void) {
    printf("========================================\n");
    printf("TEST 5: Full Pipeline Performance\n");
    printf("========================================\n\n");
    
    if (!ensure_dummy_model() || !ensure_tokenizer()) {
        printf("❌ Cannot generate dummy model or tokenizer\n");
        return;
    }
    
    q_context ctx = {0};
    q_llama_model model = {0};
    q_tokenizer tok = {0};
    
    if (!setup_model(&ctx, &model, &tok)) {
        printf("❌ Failed to setup model\n");
        return;
    }
    
    // Encode prompt
    const char* prompt_text = "Hello, how are you?";
    uint32_t prompt_tokens[100];
    uint32_t num_prompt_tokens = 0;
    
    q_error_code ret = q_tokenizer_encode(&tok, prompt_text, prompt_tokens, &num_prompt_tokens, 100, true, true);
    if (ret != Q_OK) {
        printf("❌ Failed to encode prompt: %d\n", ret);
        cleanup_model(&ctx, &model, &tok);
        return;
    }
    
    printf("Prompt: \"%s\" (%u tokens)\n", prompt_text, num_prompt_tokens);
    printf("Generating: 10 tokens\n");
    printf("Sampling: Greedy (temperature=0.0)\n\n");
    
    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < 2; i++) {
        benchmark_full_pipeline(&ctx, &model, &tok, prompt_tokens, num_prompt_tokens, 10, 0.0f, 0, 0.0f);
    }
    
    // Benchmark
    printf("Running benchmark (5 iterations)...\n");
    double total_prefill_time = 0.0;
    double total_prefill_time_per_token = 0.0;
    double total_incremental_time = 0.0;
    double total_incremental_time_per_token = 0.0;
    double total_time = 0.0;
    double total_throughput = 0.0;
    const int iterations = 5;
    
    for (int i = 0; i < iterations; i++) {
        performance_metrics m = benchmark_full_pipeline(&ctx, &model, &tok, prompt_tokens, num_prompt_tokens, 10, 0.0f, 0, 0.0f);
        total_prefill_time += m.prefill_time_ms;
        total_prefill_time_per_token += m.prefill_time_per_token_ms;
        total_incremental_time += m.incremental_time_ms;
        total_incremental_time_per_token += m.incremental_time_per_token_ms;
        total_time += m.total_time_ms;
        total_throughput += m.throughput_tokens_per_sec;
    }
    
    double avg_prefill_time = total_prefill_time / iterations;
    double avg_prefill_time_per_token = total_prefill_time_per_token / iterations;
    double avg_incremental_time = total_incremental_time / iterations;
    double avg_incremental_time_per_token = total_incremental_time_per_token / iterations;
    double avg_total_time = total_time / iterations;
    double avg_throughput = total_throughput / iterations;
    
    printf("\n--- Results ---\n");
    printf("Average Prefill Time: %.3f ms (%.3f ms/token)\n", avg_prefill_time, avg_prefill_time_per_token);
    printf("Average Incremental Time: %.3f ms (%.3f ms/token)\n", avg_incremental_time, avg_incremental_time_per_token);
    printf("Average Total Time: %.3f ms\n", avg_total_time);
    printf("Average Throughput: %.2f tokens/s\n", avg_throughput);
    printf("✓ Test 5 PASSED\n\n");
    
    cleanup_model(&ctx, &model, &tok);
}

// Test 6: Memory Usage Analysis
static void test_memory_usage(void) {
    printf("========================================\n");
    printf("TEST 5: Memory Usage Analysis\n");
    printf("========================================\n\n");
    
    if (!ensure_dummy_model() || !ensure_tokenizer()) {
        printf("❌ Cannot generate dummy model or tokenizer\n");
        return;
    }
    
    q_context ctx = {0};
    q_llama_model model = {0};
    q_tokenizer tok = {0};
    
    if (!setup_model(&ctx, &model, &tok)) {
        printf("❌ Failed to setup model\n");
        return;
    }
    
    printf("Model Configuration:\n");
    printf("  Layers: %u\n", model.config.n_layers);
    printf("  Dimension: %u\n", model.config.dim);
    printf("  Heads: %u\n", model.config.n_heads);
    printf("  KV Heads: %u\n", model.config.n_kv_heads);
    printf("  Max Seq Len: %u\n", model.config.max_seq_len);
    printf("  Vocab Size: %u\n\n", model.config.vocab_size);
    
    size_t kv_cache_size = calculate_kv_cache_size(&model.config);
    size_t arena_size = ctx.scratch_size;
    
    printf("Memory Usage:\n");
    printf("  Arena: %.2f MB\n", (double)arena_size / (1024.0 * 1024.0));
    printf("  KV Cache: %.2f MB\n", (double)kv_cache_size / (1024.0 * 1024.0));
    printf("  Total: %.2f MB\n", (double)(arena_size + kv_cache_size) / (1024.0 * 1024.0));
    printf("\n");
    
    printf("✓ Test 5 PASSED\n\n");
    
    cleanup_model(&ctx, &model, &tok);
}

// ============================================================================
// MAIN
// ============================================================================

int main(void) {
    printf("========================================\n");
    printf("  PERFORMANCE BENCHMARK SUITE\n");
    printf("========================================\n\n");
    
    test_prefill_performance();
    test_incremental_performance();
    test_sampling_strategies();
    test_prompt_size_impact();
    test_full_pipeline_performance();
    test_memory_usage();
    
    printf("========================================\n");
    printf("  ALL PERFORMANCE TESTS COMPLETED\n");
    printf("========================================\n");
    
    return 0;
}

