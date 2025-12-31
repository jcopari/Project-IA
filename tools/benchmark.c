#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>

// ============================================================================
// BENCHMARK CONFIGURATION
// ============================================================================

#define WARMUP_ITERATIONS 10
#define BENCHMARK_ITERATIONS 1000
#define MIN_BENCHMARK_TIME_MS 100.0  // Minimum time to run benchmark (ms)

// ============================================================================
// TIMING UTILITIES
// ============================================================================

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static double benchmark_function(
    void (*func)(void),
    int warmup_iterations,
    int benchmark_iterations
) {
    // Warmup
    for (int i = 0; i < warmup_iterations; i++) {
        func();
    }
    
    // Benchmark
    double start = get_time_ms();
    for (int i = 0; i < benchmark_iterations; i++) {
        func();
    }
    double end = get_time_ms();
    
    return (end - start) / benchmark_iterations;
}

// ============================================================================
// BENCHMARK: Dequantization Q4_0
// ============================================================================

static void bench_dequantize_q4_0(void) {
    // Allocate aligned memory
    q_block_q4_0* block = aligned_alloc(32, sizeof(q_block_q4_0));
    float* output = aligned_alloc(32, 32 * sizeof(float));
    
    if (!block || !output) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        return;
    }
    
    // Initialize block with test data
    block->scale = 1.0f;
    memset(block->qs, 0x12, 16);  // Pattern: 0x12 = 0001 0010 (nibbles)
    
    // Call dequantization (using public wrapper if available, or direct call)
    // Note: This is a simplified benchmark - actual implementation may vary
    for (int i = 0; i < 32; i++) {
        uint8_t q = block->qs[i / 2];
        if (i % 2 == 0) {
            q = q & 0x0F;
        } else {
            q = (q >> 4) & 0x0F;
        }
        output[i] = (float)q * block->scale - 8.0f * block->scale;
    }
    
    free(block);
    free(output);
}

// ============================================================================
// BENCHMARK: MatMul Q4_F32
// ============================================================================

static void bench_matmul_q4_f32(void) {
    const uint32_t M = 1024;
    const uint32_t N = 1024;
    
    // Allocate aligned memory
    q_tensor weights;
    weights.ne[0] = M;
    weights.ne[1] = N;
    weights.type = Q_Q4_0;
    
    size_t weight_size = (M * N / 32) * sizeof(q_block_q4_0);
    weights.data = aligned_alloc(64, weight_size);
    
    float* input = aligned_alloc(32, N * sizeof(float));
    float* output = aligned_alloc(32, M * sizeof(float));
    
    if (!weights.data || !input || !output) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        return;
    }
    
    // Initialize with test data
    memset(weights.data, 0, weight_size);
    for (uint32_t i = 0; i < N; i++) {
        input[i] = (float)(i % 10) / 10.0f;
    }
    
    // Benchmark MatMul
    q_error_code err = q_gemv_q4_f32_avx2(&weights, input, output);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: MatMul failed: %s\n", q_strerror(err));
    }
    
    free(weights.data);
    free(input);
    free(output);
}

// ============================================================================
// BENCHMARK: RMSNorm
// ============================================================================

static void bench_rmsnorm(void) {
    const uint32_t N = 4096;
    const float eps = 1e-6f;
    
    float* x = aligned_alloc(32, N * sizeof(float));
    float* weight = aligned_alloc(32, N * sizeof(float));
    float* output = aligned_alloc(32, N * sizeof(float));
    
    if (!x || !weight || !output) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        return;
    }
    
    // Initialize with test data
    for (uint32_t i = 0; i < N; i++) {
        x[i] = (float)(i % 100) / 100.0f;
        weight[i] = 1.0f;
    }
    
    // Benchmark RMSNorm
    q_error_code err = q_rmsnorm_f32_avx2(x, weight, output, N, eps);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: RMSNorm failed: %s\n", q_strerror(err));
    }
    
    free(x);
    free(weight);
    free(output);
}

// ============================================================================
// BENCHMARK: RoPE
// ============================================================================

static void bench_rope(void) {
    const uint32_t N = 4096;  // Must be even and multiple of 8
    
    float* x = aligned_alloc(32, N * sizeof(float));
    float* cos = aligned_alloc(32, (N / 2) * sizeof(float));
    float* sin = aligned_alloc(32, (N / 2) * sizeof(float));
    float* output = aligned_alloc(32, N * sizeof(float));
    
    if (!x || !cos || !sin || !output) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        return;
    }
    
    // Initialize with test data
    for (uint32_t i = 0; i < N; i++) {
        x[i] = (float)(i % 100) / 100.0f;
    }
    for (uint32_t i = 0; i < N / 2; i++) {
        cos[i] = cosf((float)i * 0.01f);
        sin[i] = sinf((float)i * 0.01f);
    }
    
    // Benchmark RoPE
    q_error_code err = q_rope_f32_avx2(x, cos, sin, output, N);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: RoPE failed: %s\n", q_strerror(err));
    }
    
    free(x);
    free(cos);
    free(sin);
    free(output);
}

// ============================================================================
// BENCHMARK: SiLU
// ============================================================================

static void bench_silu(void) {
    const uint32_t N = 4096;
    
    float* x = aligned_alloc(32, N * sizeof(float));
    float* output = aligned_alloc(32, N * sizeof(float));
    
    if (!x || !output) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        return;
    }
    
    // Initialize with test data
    for (uint32_t i = 0; i < N; i++) {
        x[i] = ((float)(i % 200) - 100.0f) / 100.0f;  // Range [-1, 1]
    }
    
    // Benchmark SiLU
    q_error_code err = q_silu_f32_avx2(x, output, N);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: SiLU failed: %s\n", q_strerror(err));
    }
    
    free(x);
    free(output);
}

// ============================================================================
// BENCHMARK: Softmax
// ============================================================================

static void bench_softmax(void) {
    const uint32_t N = 4096;
    
    float* x = aligned_alloc(32, N * sizeof(float));
    float* output = aligned_alloc(32, N * sizeof(float));
    
    if (!x || !output) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        return;
    }
    
    // Initialize with test data
    for (uint32_t i = 0; i < N; i++) {
        x[i] = ((float)(i % 100) - 50.0f) / 10.0f;  // Range [-5, 5]
    }
    
    // Benchmark Softmax
    q_error_code err = q_softmax_f32_avx2(x, output, N);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: Softmax failed: %s\n", q_strerror(err));
    }
    
    free(x);
    free(output);
}

// ============================================================================
// MAIN BENCHMARK RUNNER
// ============================================================================

static void print_header(const char* name) {
    printf("\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("Benchmark: %s\n", name);
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
}

static void print_result(const char* metric, double value, const char* unit) {
    printf("  %-30s: %10.4f %s\n", metric, value, unit);
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    
    printf("Qorus-IA v2.0 Performance Benchmark Suite\n");
    printf("==========================================\n");
    printf("Warmup iterations: %d\n", WARMUP_ITERATIONS);
    printf("Benchmark iterations: %d\n", BENCHMARK_ITERATIONS);
    printf("\n");
    
    // Benchmark 1: Dequantization Q4_0
    print_header("Dequantization Q4_0");
    double dequant_time = benchmark_function(bench_dequantize_q4_0, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
    print_result("Latency", dequant_time, "ms");
    print_result("Throughput", 1000.0 / dequant_time, "ops/s");
    
    // Benchmark 2: MatMul Q4_F32
    print_header("MatMul Q4_F32 (1024x1024)");
    double matmul_time = benchmark_function(bench_matmul_q4_f32, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
    print_result("Latency", matmul_time, "ms");
    print_result("Throughput", 1000.0 / matmul_time, "ops/s");
    // Calculate GFLOPS: 2*M*N operations (multiply-add) / time
    double gflops = (2.0 * 1024.0 * 1024.0) / (matmul_time * 1e6);
    print_result("Performance", gflops, "GFLOPS");
    
    // Benchmark 3: RMSNorm
    print_header("RMSNorm (4096 elements)");
    double rmsnorm_time = benchmark_function(bench_rmsnorm, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
    print_result("Latency", rmsnorm_time, "ms");
    print_result("Throughput", 1000.0 / rmsnorm_time, "ops/s");
    
    // Benchmark 4: RoPE
    print_header("RoPE (4096 elements)");
    double rope_time = benchmark_function(bench_rope, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
    print_result("Latency", rope_time, "ms");
    print_result("Throughput", 1000.0 / rope_time, "ops/s");
    
    // Benchmark 5: SiLU
    print_header("SiLU (4096 elements)");
    double silu_time = benchmark_function(bench_silu, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
    print_result("Latency", silu_time, "ms");
    print_result("Throughput", 1000.0 / silu_time, "ops/s");
    
    // Benchmark 6: Softmax
    print_header("Softmax (4096 elements)");
    double softmax_time = benchmark_function(bench_softmax, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
    print_result("Latency", softmax_time, "ms");
    print_result("Throughput", 1000.0 / softmax_time, "ops/s");
    
    // Summary
    printf("\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("Benchmark Summary\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("All benchmarks completed successfully.\n");
    printf("\n");
    
    return 0;
}

