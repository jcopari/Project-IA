// ============================================================================
// BENCHMARK: Text Generation Performance (FASE 4.2)
// ============================================================================
// Mede latência por token de geração completa
// Métricas: prefill time, incremental generation time, throughput
// ============================================================================

#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#ifdef __AVX2__
#include <immintrin.h>  // Para SIMD argmax
#endif

// ============================================================================
// BENCHMARK CONFIGURATION
// ============================================================================

#define WARMUP_ITERATIONS 3
#define BENCHMARK_ITERATIONS 10
#define MIN_BENCHMARK_TIME_MS 100.0

// ============================================================================
// TIMING UTILITIES
// ============================================================================

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

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

static size_t calculate_kv_cache_size(const q_llama_config* config) {
    uint32_t head_dim = config->dim / config->n_heads;
    size_t kv_size = (size_t)config->n_layers * 
                     (size_t)config->n_kv_heads * 
                     (size_t)config->max_seq_len * 
                     (size_t)head_dim * 
                     sizeof(float) * 2; // K + V
    return Q_ALIGN_SIZE(kv_size);
}

// ============================================================================
// BENCHMARK: Prefill Performance
// ============================================================================

static double benchmark_prefill(
    q_llama_model* model,
    q_context* ctx,
    const uint32_t* tokens,
    uint32_t num_tokens,
    float* logits
) {
    double total_time = 0.0;
    
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        q_arena_reset(ctx);
        
        double start = get_time_ms();
        q_error_code ret = llama_forward(model, ctx, tokens, num_tokens, 0, logits);
        double end = get_time_ms();
        
        if (ret != Q_OK) {
            return -1.0;  // Error
        }
        
        total_time += (end - start);
    }
    
    return total_time / BENCHMARK_ITERATIONS;
}

// ============================================================================
// BENCHMARK: Incremental Generation Performance
// ============================================================================

static double benchmark_incremental_generation(
    q_llama_model* model,
    q_context* ctx,
    q_tokenizer* tokenizer,
    const char* prompt,
    uint32_t num_tokens_to_generate
) {
    // Setup
    uint32_t prompt_tokens[256];
    uint32_t num_prompt_tokens = 0;
    q_error_code ret = q_tokenizer_encode(tokenizer, prompt, prompt_tokens, &num_prompt_tokens, 256, true, false);
    if (ret != Q_OK || num_prompt_tokens == 0) {
        return -1.0;
    }
    
    // Prefill (warmup)
    size_t logits_size = Q_ALIGN_SIZE((size_t)model->config.vocab_size * sizeof(float));
    float* logits = (float*)q_arena_alloc(ctx, logits_size);
    if (logits == NULL) {
        return -1.0;
    }
    
    q_arena_reset(ctx);
    ret = llama_forward(model, ctx, prompt_tokens, num_prompt_tokens, 0, logits);
    if (ret != Q_OK) {
        return -1.0;
    }
    
    // OTIMIZAÇÃO CRÍTICA: Alocar logits uma vez antes do loop (não dentro)
    // Reutilizar logits sem reset de arena desnecessário
    q_arena_reset(ctx);  // Reset uma vez antes do loop
    logits = (float*)q_arena_alloc(ctx, logits_size);
    if (logits == NULL) {
        return -1.0;
    }
    
    // Benchmark incremental generation
    double total_time = 0.0;
    uint32_t current_pos = num_prompt_tokens;
    
    for (uint32_t t = 0; t < num_tokens_to_generate; t++) {
        if (current_pos >= model->config.max_seq_len) {
            break;
        }
        
        // OTIMIZAÇÃO CRÍTICA: Argmax SIMD em vez de escalar
        // Sample token (simplified - just use argmax for benchmark)
        uint32_t token_id = 0;
        float max_logit = logits[0];
        
        #ifdef __AVX2__
        // SIMD argmax: processar 8 elementos por vez
        __m256 max_vec = _mm256_loadu_ps(&logits[0]);
        uint32_t max_indices[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        uint32_t vec_end = model->config.vocab_size & ~7U;
        
        for (uint32_t i = 8; i < vec_end; i += 8) {
            __m256 logits_vec = _mm256_loadu_ps(&logits[i]);
            __m256 cmp = _mm256_cmp_ps(logits_vec, max_vec, _CMP_GT_OQ);
            uint32_t mask = _mm256_movemask_ps(cmp);
            
            if (mask != 0) {
                // Atualizar max_vec
                max_vec = _mm256_max_ps(max_vec, logits_vec);
                
                // Encontrar índices dos elementos maiores
                for (uint32_t j = 0; j < 8; j++) {
                    if (mask & (1U << j)) {
                        max_indices[j] = i + j;
                    }
                }
            }
        }
        
        // Reduzir max_vec para encontrar máximo final
        float max_vals[8];
        _mm256_storeu_ps(max_vals, max_vec);
        for (uint32_t i = 0; i < 8; i++) {
            if (max_vals[i] > max_logit) {
                max_logit = max_vals[i];
                token_id = max_indices[i];
            }
        }
        
        // Processar elementos restantes escalarmente
        for (uint32_t i = vec_end; i < model->config.vocab_size; i++) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                token_id = i;
            }
        }
        #else
        // Fallback escalar
        for (uint32_t i = 1; i < model->config.vocab_size; i++) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                token_id = i;
            }
        }
        #endif
        
        // Incremental forward
        // OTIMIZAÇÃO CRÍTICA: Não resetar arena dentro do loop
        // Reset apenas quando necessário (após forward pass)
        uint32_t incremental_tokens[1] = {token_id};
        double start = get_time_ms();
        ret = llama_forward(model, ctx, incremental_tokens, 1, current_pos, logits);
        double end = get_time_ms();
        
        if (ret != Q_OK) {
            return -1.0;
        }
        
        total_time += (end - start);
        current_pos++;
        
        // Reset arena após forward pass (para próxima iteração)
        q_arena_reset(ctx);
    }
    
    return total_time / num_tokens_to_generate;  // Average per token
}

// ============================================================================
// BENCHMARK: Full Generation Pipeline
// ============================================================================

static double benchmark_full_generation(
    q_generation_state* gen_state,
    uint32_t num_iterations
) {
    // OTIMIZAÇÃO CRÍTICA: Warmup antes de medir
    // Primeira iteração pode ser mais lenta devido a cache misses, page faults, etc.
    for (uint32_t w = 0; w < WARMUP_ITERATIONS; w++) {
        gen_state->num_generated_tokens = 0;
        gen_state->current_pos = 0;
        q_generate(gen_state);  // Warmup sem medir tempo
    }
    
    // Benchmark real
    double total_time = 0.0;
    
    for (uint32_t i = 0; i < num_iterations; i++) {
        // Reset state
        gen_state->num_generated_tokens = 0;
        gen_state->current_pos = 0;
        
        double start = get_time_ms();
        q_error_code ret = q_generate(gen_state);
        double end = get_time_ms();
        
        if (ret != Q_OK) {
            return -1.0;
        }
        
        total_time += (end - start);
    }
    
    return total_time / num_iterations;
}

// ============================================================================
// MAIN BENCHMARK RUNNER
// ============================================================================

int main(int argc, char** argv) {
    (void)argc;  // Unused
    (void)argv;   // Unused
    printf("========================================\n");
    printf("  GENERATION PERFORMANCE BENCHMARK\n");
    printf("========================================\n\n");
    
    if (!ensure_dummy_model() || !ensure_tokenizer()) {
        fprintf(stderr, "ERROR: Cannot generate dummy model/tokenizer\n");
        return 1;
    }
    
    // Initialize
    q_context ctx = {0};
    q_llama_model model = {0};
    q_tokenizer tokenizer = {0};
    q_error_code ret;
    
    ret = q_init_memory(&ctx, "model_dummy.qorus");
    if (ret != Q_OK) {
        fprintf(stderr, "ERROR: q_init_memory failed: %d\n", ret);
        return 1;
    }
    
    ret = q_alloc_arena(&ctx, 64 * 1024 * 1024);  // 64MB
    if (ret != Q_OK) {
        fprintf(stderr, "ERROR: q_alloc_arena failed: %d\n", ret);
        q_free_memory(&ctx);
        return 1;
    }
    
    ret = llama_build_graph(&ctx, &model);
    if (ret != Q_OK) {
        fprintf(stderr, "ERROR: llama_build_graph failed: %d\n", ret);
        q_free_memory(&ctx);
        return 1;
    }
    
    size_t kv_size = calculate_kv_cache_size(&model.config);
    ret = q_alloc_kv_cache(&ctx, kv_size);
    if (ret != Q_OK) {
        fprintf(stderr, "ERROR: q_alloc_kv_cache failed: %d\n", ret);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    
    ret = q_tokenizer_load(&tokenizer, "tokenizer.bin");
    if (ret != Q_OK) {
        fprintf(stderr, "ERROR: q_tokenizer_load failed: %d\n", ret);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    
    printf("Model: %u layers, %u dim, vocab_size=%u\n", 
           model.config.n_layers, model.config.dim, model.config.vocab_size);
    printf("\n");
    
    // Benchmark 1: Prefill performance
    printf("Benchmark 1: Prefill Performance\n");
    printf("-----------------------------------\n");
    const char* prompt = "Hello, how are you?";
    uint32_t prompt_tokens[256];
    uint32_t num_prompt_tokens = 0;
    ret = q_tokenizer_encode(&tokenizer, prompt, prompt_tokens, &num_prompt_tokens, 256, true, false);
    if (ret == Q_OK && num_prompt_tokens > 0) {
        size_t logits_size = Q_ALIGN_SIZE((size_t)model.config.vocab_size * sizeof(float));
        float* logits = (float*)malloc(logits_size);
        
        double prefill_time = benchmark_prefill(&model, &ctx, prompt_tokens, num_prompt_tokens, logits);
        if (prefill_time >= 0.0) {
            printf("  Prefill time: %.3f ms (seq_len=%u)\n", prefill_time, num_prompt_tokens);
            printf("  Time per token: %.3f ms\n", prefill_time / num_prompt_tokens);
        } else {
            printf("  ERROR: Prefill benchmark failed\n");
        }
        
        free(logits);
    }
    printf("\n");
    
    // Benchmark 2: Incremental generation
    printf("Benchmark 2: Incremental Generation Performance\n");
    printf("-----------------------------------\n");
    double incr_time = benchmark_incremental_generation(&model, &ctx, &tokenizer, prompt, 10);
    if (incr_time >= 0.0) {
        printf("  Incremental generation time: %.3f ms/token\n", incr_time);
        printf("  Throughput: %.2f tokens/s\n", 1000.0 / incr_time);
    } else {
        printf("  ERROR: Incremental generation benchmark failed\n");
    }
    printf("\n");
    
    // Benchmark 3: Full generation pipeline
    printf("Benchmark 3: Full Generation Pipeline\n");
    printf("-----------------------------------\n");
    uint32_t generated_tokens[256];
    q_generation_state gen_state = {
        .ctx = &ctx,
        .model = &model,
        .tokenizer = &tokenizer,
        .prompt_tokens = prompt_tokens,
        .num_prompt_tokens = num_prompt_tokens,
        .generated_tokens = generated_tokens,
        .num_generated_tokens = 0,
        .max_tokens = 10,
        .temperature = 0.8f,
        .top_k = 40,
        .top_p = 0.9f,
        .current_pos = 0
    };
    
    double full_time = benchmark_full_generation(&gen_state, BENCHMARK_ITERATIONS);
    if (full_time >= 0.0) {
        printf("  Full generation time: %.3f ms (avg over %d iterations)\n", full_time, BENCHMARK_ITERATIONS);
        if (gen_state.num_generated_tokens > 0) {
            printf("  Time per token: %.3f ms\n", full_time / gen_state.num_generated_tokens);
            printf("  Throughput: %.2f tokens/s\n", 1000.0 * gen_state.num_generated_tokens / full_time);
        }
    } else {
        printf("  ERROR: Full generation benchmark failed\n");
    }
    printf("\n");
    
    // Cleanup
    q_tokenizer_free(&tokenizer);
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    printf("========================================\n");
    printf("  BENCHMARK COMPLETE\n");
    printf("========================================\n");
    
    return 0;
}

