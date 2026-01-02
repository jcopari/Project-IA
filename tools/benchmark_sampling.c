// ============================================================================
// BENCHMARK: Sampling Performance (SoA Optimizations)
// ============================================================================
// Mede impacto das otimizações SoA e qsort_soa no sampling
// Métricas: tempo por chamada q_sample_token, cache misses
// ============================================================================

#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

// ============================================================================
// BENCHMARK CONFIGURATION
// ============================================================================

#define WARMUP_ITERATIONS 10
#define BENCHMARK_ITERATIONS 1000
#define VOCAB_SIZE 32000  // Typical vocabulary size

// ============================================================================
// TIMING UTILITIES
// ============================================================================

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// ============================================================================
// BENCHMARK: q_sample_token Performance
// ============================================================================

static double benchmark_sampling(
    float* logits,
    uint32_t vocab_size,
    float temperature,
    uint32_t top_k,
    float top_p,
    q_context* restrict ctx
) {
    uint32_t token_id;
    double total_time = 0.0;
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        q_sample_token(logits, vocab_size, temperature, top_k, top_p, &token_id, ctx);
    }
    
    // Benchmark
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        q_arena_reset(ctx);  // Reset arena before each call
        
        double start = get_time_ms();
        q_error_code ret = q_sample_token(logits, vocab_size, temperature, top_k, top_p, &token_id, ctx);
        double end = get_time_ms();
        
        if (ret != Q_OK) {
            return -1.0;  // Error
        }
        
        total_time += (end - start);
    }
    
    return total_time / BENCHMARK_ITERATIONS;  // Average time per call
}

// ============================================================================
// MAIN BENCHMARK RUNNER
// ============================================================================

int main(void) {
    printf("========================================\n");
    printf("  BENCHMARK: Sampling Performance (SoA)\n");
    printf("========================================\n\n");
    
    // Initialize context
    q_context ctx = {0};
    q_error_code ctx_err = q_alloc_arena(&ctx, 1024 * 1024 * 100);  // 100MB scratchpad
    if (ctx_err != Q_OK) {
        fprintf(stderr, "Failed to allocate arena\n");
        return 1;
    }
    
    // Allocate logits array
    float* logits = (float*)malloc(VOCAB_SIZE * sizeof(float));
    if (logits == NULL) {
        fprintf(stderr, "Failed to allocate logits\n");
        q_free_memory(&ctx);
        return 1;
    }
    
    // Initialize logits with random values
    srand(42);  // Fixed seed for reproducibility
    for (uint32_t i = 0; i < VOCAB_SIZE; i++) {
        logits[i] = (float)rand() / (float)RAND_MAX * 10.0f - 5.0f;  // [-5, 5]
    }
    
    printf("Configuration:\n");
    printf("  Vocab Size: %u\n", VOCAB_SIZE);
    printf("  Warmup Iterations: %d\n", WARMUP_ITERATIONS);
    printf("  Benchmark Iterations: %d\n", BENCHMARK_ITERATIONS);
    printf("\n");
    
    // Test Case 1: Greedy Sampling (temperature = 0.0)
    printf("Test Case 1: Greedy Sampling (temperature=0.0)\n");
    printf("---------------------------------------------\n");
    double time_greedy = benchmark_sampling(logits, VOCAB_SIZE, 0.0f, 0, 0.0f, &ctx);
    if (time_greedy < 0) {
        fprintf(stderr, "Benchmark failed\n");
        free(logits);
        q_free_memory(&ctx);
        return 1;
    }
    printf("  Average time per call: %.4f ms\n", time_greedy);
    printf("  Throughput: %.2f calls/sec\n", 1000.0 / time_greedy);
    printf("\n");
    
    // Test Case 2: Top-k Sampling (k=10)
    printf("Test Case 2: Top-k Sampling (k=10)\n");
    printf("----------------------------------\n");
    double time_top_k = benchmark_sampling(logits, VOCAB_SIZE, 1.0f, 10, 0.0f, &ctx);
    if (time_top_k < 0) {
        fprintf(stderr, "Benchmark failed\n");
        free(logits);
        q_free_memory(&ctx);
        return 1;
    }
    printf("  Average time per call: %.4f ms\n", time_top_k);
    printf("  Throughput: %.2f calls/sec\n", 1000.0 / time_top_k);
    printf("\n");
    
    // Test Case 3: Top-p Sampling (p=0.9)
    printf("Test Case 3: Top-p Sampling (p=0.9)\n");
    printf("-----------------------------------\n");
    double time_top_p = benchmark_sampling(logits, VOCAB_SIZE, 1.0f, 0, 0.9f, &ctx);
    if (time_top_p < 0) {
        fprintf(stderr, "Benchmark failed\n");
        free(logits);
        q_free_memory(&ctx);
        return 1;
    }
    printf("  Average time per call: %.4f ms\n", time_top_p);
    printf("  Throughput: %.2f calls/sec\n", 1000.0 / time_top_p);
    printf("\n");
    
    // Test Case 4: Combined Top-k + Top-p (k=10, p=0.9)
    printf("Test Case 4: Combined Top-k + Top-p (k=10, p=0.9)\n");
    printf("--------------------------------------------------\n");
    double time_combined = benchmark_sampling(logits, VOCAB_SIZE, 1.0f, 10, 0.9f, &ctx);
    if (time_combined < 0) {
        fprintf(stderr, "Benchmark failed\n");
        free(logits);
        q_free_memory(&ctx);
        return 1;
    }
    printf("  Average time per call: %.4f ms\n", time_combined);
    printf("  Throughput: %.2f calls/sec\n", 1000.0 / time_combined);
    printf("\n");
    
    // Summary
    printf("========================================\n");
    printf("  SUMMARY\n");
    printf("========================================\n");
    printf("Greedy:      %.4f ms/call (%.2f calls/sec)\n", time_greedy, 1000.0 / time_greedy);
    printf("Top-k:       %.4f ms/call (%.2f calls/sec)\n", time_top_k, 1000.0 / time_top_k);
    printf("Top-p:       %.4f ms/call (%.2f calls/sec)\n", time_top_p, 1000.0 / time_top_p);
    printf("Combined:    %.4f ms/call (%.2f calls/sec)\n", time_combined, 1000.0 / time_combined);
    printf("\n");
    
    // Cleanup
    free(logits);
    q_free_memory(&ctx);
    
    return 0;
}

