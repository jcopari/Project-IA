#include "qorus.h"
#include <immintrin.h>
#include <string.h>

// Causal Masking AVX2: Set upper triangular elements to mask_value
// Optimized Architecture (Carmack-Approved):
// 1. Scalar Region: Elements before aligned block
// 2. Boundary Block: Contains boundary (i), needs Load+Cmp+Blend
// 3. Right Side Blocks: Store-only (no Load, no Compare) - reduces bandwidth by ~50%
//
// Time Complexity: O(N²) - Must set N(N-1)/2 elements (upper triangle)
// Space Complexity: O(1) - In-place operation, no temporary buffers
//
// Reference: Vaswani et al. (2017) - "Attention is All You Need"

q_error_code q_causal_mask_f32_avx2(
    q_tensor* scores,
    float mask_value
) {
    // STEP 0: Validation (always active)
    Q_VALIDATE_PTR_OR_RETURN(scores, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(scores->data, Q_ERR_INVALID_ARG);
    
    // Extract dimensions
    const uint32_t seq_len = scores->ne[0];
    const uint32_t dim_rows = scores->ne[1];
    
    // Validate square matrix
    Q_VALIDATE_NONZERO_OR_RETURN(seq_len, Q_ERR_INVALID_SIZE);
    Q_VALIDATE_OR_RETURN(seq_len == dim_rows, Q_ERR_INVALID_SIZE);
    
    // Validate type
    Q_VALIDATE_OR_RETURN(scores->type == Q_F32, Q_ERR_INVALID_DTYPE);
    
    // Validate alignment (AVX2 requires 32-byte alignment)
    Q_VALIDATE_ALIGNED_OR_RETURN(scores->data, Q_ERR_MISALIGNED);
    
    // Special case: seq_len = 1 (no masking needed, only diagonal)
    if (seq_len == 1) {
        return Q_OK;
    }
    
    // Get data pointer
    float* restrict matrix = (float*)scores->data;
    
    // Extract stride
    const size_t stride = scores->nb[0] / sizeof(float);
    Q_VALIDATE_OR_RETURN(stride >= seq_len, Q_ERR_INVALID_SIZE);
    
    // Base index vector [0, 1, 2, 3, 4, 5, 6, 7]
    const __m256 vec_idx_base = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
    const __m256 vec_mask_val = _mm256_set1_ps(mask_value);
    
    // Loop through rows (Queries)
    for (uint32_t i = 0; i < seq_len; i++) {
        const __m256 vec_row_idx = _mm256_set1_ps((float)i);
        
        // Strategy: Separate into 3 regions
        // 1. Scalar: Elements before aligned block
        // 2. Boundary Block: Contains boundary (i), needs Load+Cmp+Blend
        // 3. Right Side Blocks: Completely to the right, Store only
        
        // Calculate start of aligned block containing boundary
        // Round (i+1) down to nearest multiple of 8
        uint32_t j_boundary_block_start = (i + 1) & ~7U;
        if (j_boundary_block_start > seq_len) {
            j_boundary_block_start = seq_len;
        }
        
        // Calculate start of blocks completely to the right
        // Round (i+1) up to nearest multiple of 8
        uint32_t j_right_blocks_start = ((i + 1) + 7) & ~7U;
        if (j_right_blocks_start > seq_len) {
            j_right_blocks_start = seq_len;
        }
        
        // ============================================================
        // REGION 1: Scalar elements before aligned block
        // ============================================================
        uint32_t j = i + 1;
        for (; j < j_boundary_block_start && j < seq_len; j++) {
            matrix[(size_t)i * stride + j] = mask_value;
        }
        
        // ============================================================
        // REGION 2: BOUNDARY BLOCK (Load + Compare + Blend)
        // This is the ONLY place where we need to read and compare
        // ============================================================
        if (j_boundary_block_start < j_right_blocks_start) {
            // Ensure we have at least 8 elements to process
            if (j_boundary_block_start + 8 <= seq_len) {
                j = j_boundary_block_start;
                
                // Current column indices: [j, j+1, ... j+7]
                __m256 vec_col_idx = _mm256_add_ps(vec_idx_base, _mm256_set1_ps((float)j));
                
                // Mask: col > row
                __m256 vec_cmp = _mm256_cmp_ps(vec_col_idx, vec_row_idx, _CMP_GT_OQ);
                
                // Load data
                __m256 vec_data = _mm256_loadu_ps(&matrix[(size_t)i * stride + j]);
                
                // Apply mask: If cmp is true, use mask_value, otherwise keep data
                __m256 vec_result = _mm256_blendv_ps(vec_data, vec_mask_val, vec_cmp);
                
                // Store result
                _mm256_storeu_ps(&matrix[(size_t)i * stride + j], vec_result);
                j += 8;
            } else {
                // Not enough elements for AVX2, handle in tail loop
                j = j_boundary_block_start;
            }
        }
        
        // ============================================================
        // REGION 3: RIGHT SIDE BLOCKS (Store-Only, no Load)
        // Here all elements will be masked, so don't read
        // ============================================================
        // Ensure we have at least 8 elements remaining before processing
        // Use j + 8 <= seq_len to avoid underflow when seq_len < 8
        while (j + 8 <= seq_len) {
            // Direct store: No need for Load, Compare or Blend
            _mm256_storeu_ps(&matrix[(size_t)i * stride + j], vec_mask_val);
            j += 8;
        }
        
        // ============================================================
        // REGION 4: Tail loop for remaining elements
        // ============================================================
        for (; j < seq_len; j++) {
            if (j > i) {
                matrix[(size_t)i * stride + j] = mask_value;
            }
        }
    }
    
    return Q_OK;
}

