#include "qorus.h"
#include <immintrin.h>

// Dequantize Q4_0 using Pure AVX2 (No Scalar Fallback)
// Force inlining to avoid function call overhead in tight MatMul loops
// This function will be called millions of times per inference
//
// Algorithm:
// 1. Load all 16 bytes (32 nibbles) in single XMM load
// 2. Separate nibbles using vectorized bitwise operations
// 3. Interleave to restore correct order
// 4. Expand to int32, convert to float32, apply FMA
//
// Time Complexity: O(1) - Fixed instruction count, no loops
// Space Complexity: O(1) - Only AVX2 registers
// Latency: ~10-15 cycles (optimized with FMA)
// Throughput: Limited by memory bandwidth and AVX execution ports
static inline __attribute__((always_inline)) void q_dequantize_q4_0_block_avx2(
    const q_block_q4_0* restrict block,
    float* restrict output
) {
    // Precondition: output must be 32-byte aligned
    Q_ASSERT_ALIGNED(output);

    // 1. Setup Constants (Broadcast once, reuse 4 times)
    // Pre-calculate (-8.0 * scale) for FMA: result = q * scale + (-8 * scale)
    const float scale = block->scale;
    const __m256 scale_vec = _mm256_broadcast_ss(&scale);
    const __m256 offset_vec = _mm256_mul_ps(_mm256_set1_ps(-8.0f), scale_vec);
    const __m128i low_mask = _mm_set1_epi8(0x0F);

    // 2. Load all 16 bytes (32 nibbles) in single XMM load
    // Cache-friendly: single aligned load maximizes memory bandwidth
    const __m128i raw_bytes = _mm_loadu_si128((const __m128i*)block->qs);

    // 3. Separate Nibbles (Pure Vectorized)
    // Extract lower 4 bits: positions 0, 2, 4, 6, ...
    // Extract upper 4 bits: positions 1, 3, 5, 7, ...
    const __m128i low_nibbles  = _mm_and_si128(raw_bytes, low_mask);
    const __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(raw_bytes, 4), low_mask);

    // 4. Interleave to restore correct order: 0, 1, 2, 3, ...
    // unpacklo: interleaves lower 64 bits -> [0-7, 8-15] in correct order
    // unpackhi: interleaves upper 64 bits -> [16-23, 24-31] in correct order
    const __m128i v0_15  = _mm_unpacklo_epi8(low_nibbles, high_nibbles);
    const __m128i v16_31 = _mm_unpackhi_epi8(low_nibbles, high_nibbles);

    // 5. Process 4 batches of 8 values using FMA (Fused Multiply-Add)
    // Formula: result = q * scale + (-8 * scale) = (q - 8) * scale
    
    // Batch 1: Values 0-7 (lower 64 bits of v0_15)
    __m256i i0_7 = _mm256_cvtepu8_epi32(v0_15);           // Expand bytes to int32
    __m256 f0_7 = _mm256_cvtepi32_ps(i0_7);               // Convert int32 to float32
    _mm256_store_ps(output, _mm256_fmadd_ps(f0_7, scale_vec, offset_vec));

    // Batch 2: Values 8-15 (upper 64 bits of v0_15)
    // Extract upper 64 bits: shift right by 8 bytes
    __m128i v8_15 = _mm_bsrli_si128(v0_15, 8);            // Move bytes 8-15 to positions 0-7
    __m256i i8_15 = _mm256_cvtepu8_epi32(v8_15);
    __m256 f8_15 = _mm256_cvtepi32_ps(i8_15);
    _mm256_store_ps(output + 8, _mm256_fmadd_ps(f8_15, scale_vec, offset_vec));

    // Batch 3: Values 16-23 (lower 64 bits of v16_31)
    __m256i i16_23 = _mm256_cvtepu8_epi32(v16_31);
    __m256 f16_23 = _mm256_cvtepi32_ps(i16_23);
    _mm256_store_ps(output + 16, _mm256_fmadd_ps(f16_23, scale_vec, offset_vec));

    // Batch 4: Values 24-31 (upper 64 bits of v16_31)
    __m128i v24_31 = _mm_bsrli_si128(v16_31, 8);          // Move bytes 24-31 to positions 0-7
    __m256i i24_31 = _mm256_cvtepu8_epi32(v24_31);
    __m256 f24_31 = _mm256_cvtepi32_ps(i24_31);
    _mm256_store_ps(output + 24, _mm256_fmadd_ps(f24_31, scale_vec, offset_vec));
}

// Public wrapper for testing (non-inline version)
// This allows the function to be called from test code
// Note: Includes NULL validation since this is not in the hot path
//
// Behavior: Returns silently on NULL inputs (defensive programming for tests)
// Production code should always call the inline version with validated inputs
// This wrapper exists solely for testability, not for production use
void q_dequantize_q4_0_block_avx2_public(
    const q_block_q4_0* restrict block,
    float* restrict output
) {
    // ROBUSTNESS: Validate inputs (only in public wrapper, not in hot path)
    // This prevents crashes in test scenarios while maintaining zero overhead
    // in production code paths that use the inline version directly
    if (__builtin_expect(block == NULL || output == NULL, 0)) {
        // Silently return - caller should validate inputs
        // This is acceptable behavior for test code defensive programming
        return;
    }
    
    q_dequantize_q4_0_block_avx2(block, output);
}
