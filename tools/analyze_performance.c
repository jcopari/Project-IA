// ============================================================================
// PERFORMANCE ANALYSIS TOOL (Alternative to perf)
// ============================================================================
// Análise de performance usando timestamps de alta precisão
// Não requer permissões especiais como perf
// ============================================================================

#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

// ============================================================================
// TIMING UTILITIES
// ============================================================================

static double get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

// ============================================================================
// PERFORMANCE ANALYSIS
// ============================================================================

typedef struct {
    const char* function_name;
    uint64_t call_count;
    double total_time_ns;
    double min_time_ns;
    double max_time_ns;
} perf_counter_t;

#define MAX_COUNTERS 50
static perf_counter_t counters[MAX_COUNTERS];
static uint32_t num_counters = 0;

static perf_counter_t* get_or_create_counter(const char* name) {
    for (uint32_t i = 0; i < num_counters; i++) {
        if (strcmp(counters[i].function_name, name) == 0) {
            return &counters[i];
        }
    }
    
    if (num_counters >= MAX_COUNTERS) {
        return NULL;
    }
    
    perf_counter_t* counter = &counters[num_counters++];
    counter->function_name = name;
    counter->call_count = 0;
    counter->total_time_ns = 0.0;
    counter->min_time_ns = 1e9;
    counter->max_time_ns = 0.0;
    return counter;
}

// Macro para medir tempo de função
#define MEASURE_TIME(func_call, counter_name) \
    do { \
        double start = get_time_ns(); \
        func_call; \
        double end = get_time_ns(); \
        double elapsed = end - start; \
        perf_counter_t* counter = get_or_create_counter(counter_name); \
        if (counter) { \
            counter->call_count++; \
            counter->total_time_ns += elapsed; \
            if (elapsed < counter->min_time_ns) counter->min_time_ns = elapsed; \
            if (elapsed > counter->max_time_ns) counter->max_time_ns = elapsed; \
        } \
    } while(0)

// ============================================================================
// MAIN ANALYSIS
// ============================================================================

static bool ensure_dummy_model(void) {
    FILE* f = fopen("model_dummy.qorus", "rb");
    if (f != NULL) {
        fclose(f);
        return true;
    }
    return false;
}

static bool ensure_tokenizer(void) {
    FILE* f = fopen("tokenizer.bin", "rb");
    if (f != NULL) {
        fclose(f);
        return true;
    }
    return false;
}

static size_t calculate_kv_cache_size(const q_llama_config* config) {
    uint32_t head_dim = config->dim / config->n_heads;
    size_t kv_size = (size_t)config->n_layers * 
                     (size_t)config->n_kv_heads * 
                     (size_t)config->max_seq_len * 
                     (size_t)head_dim * 
                     sizeof(float) * 2;
    return Q_ALIGN_SIZE(kv_size);
}

int main(void) {
    printf("========================================\n");
    printf("  PERFORMANCE ANALYSIS TOOL\n");
    printf("========================================\n\n");
    
    if (!ensure_dummy_model() || !ensure_tokenizer()) {
        fprintf(stderr, "ERROR: Model or tokenizer not found\n");
        return 1;
    }
    
    // Initialize
    q_context ctx = {0};
    q_llama_model model = {0};
    q_tokenizer tokenizer = {0};
    q_error_code ret;
    
    MEASURE_TIME(
        ret = q_init_memory(&ctx, "model_dummy.qorus"),
        "q_init_memory"
    );
    
    if (ret != Q_OK) {
        fprintf(stderr, "ERROR: q_init_memory failed\n");
        return 1;
    }
    
    MEASURE_TIME(
        ret = q_alloc_arena(&ctx, 64 * 1024 * 1024),
        "q_alloc_arena"
    );
    
    MEASURE_TIME(
        ret = llama_build_graph(&ctx, &model),
        "llama_build_graph"
    );
    
    size_t kv_size = calculate_kv_cache_size(&model.config);
    MEASURE_TIME(
        ret = q_alloc_kv_cache(&ctx, kv_size),
        "q_alloc_kv_cache"
    );
    
    MEASURE_TIME(
        ret = q_tokenizer_load(&tokenizer, "tokenizer.bin"),
        "q_tokenizer_load"
    );
    
    // Test generation
    const char* prompt = "Hello";
    uint32_t prompt_tokens[256];
    uint32_t num_prompt_tokens = 0;
    
    MEASURE_TIME(
        ret = q_tokenizer_encode(&tokenizer, prompt, prompt_tokens, &num_prompt_tokens, 256, true, false),
        "q_tokenizer_encode"
    );
    
    uint32_t generated_tokens[256];
    q_generation_state gen_state = {
        .ctx = &ctx,
        .model = &model,
        .tokenizer = &tokenizer,
        .prompt_tokens = prompt_tokens,
        .num_prompt_tokens = num_prompt_tokens,
        .generated_tokens = generated_tokens,
        .num_generated_tokens = 0,
        .max_tokens = 5,
        .temperature = 0.8f,
        .top_k = 40,
        .top_p = 0.9f,
        .current_pos = 0
    };
    
    MEASURE_TIME(
        ret = q_generate(&gen_state),
        "q_generate"
    );
    
    // Print results
    printf("\n========================================\n");
    printf("  PERFORMANCE COUNTERS\n");
    printf("========================================\n\n");
    printf("%-30s %12s %15s %15s %15s\n", 
           "Function", "Calls", "Total (ms)", "Avg (ms)", "Min/Max (ms)");
    printf("%-30s %12s %15s %15s %15s\n", 
           "------------------------------", "------------", "---------------", "---------------", "---------------");
    
    for (uint32_t i = 0; i < num_counters; i++) {
        perf_counter_t* c = &counters[i];
        double total_ms = c->total_time_ns / 1e6;
        double avg_ms = (c->call_count > 0) ? (c->total_time_ns / c->call_count) / 1e6 : 0.0;
        double min_ms = c->min_time_ns / 1e6;
        double max_ms = c->max_time_ns / 1e6;
        
        printf("%-30s %12lu %15.3f %15.3f %15.3f/%.3f\n",
               c->function_name, c->call_count, total_ms, avg_ms, min_ms, max_ms);
    }
    
    printf("\n========================================\n");
    
    // Cleanup
    q_tokenizer_free(&tokenizer);
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    return 0;
}

