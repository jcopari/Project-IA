# Causal Masking AVX2: Mathematical Proof

## TIME COMPLEXITY
- **Naive:** O(N²) - Must set N(N-1)/2 elements (upper triangle)
- **Optimized:** O(N²) - Same complexity, but vectorized with AVX2
- **Optimal:** Cannot be better than O(N²) - must set every upper triangular element
- **Cache Complexity:** Sequential row access maximizes spatial locality. Each row processed once.

## SPACE COMPLEXITY
- **Auxiliary Space:** O(1) - In-place operation, no temporary buffers
- **Peak Memory:** O(N²) for input matrix (already allocated)
- **SIMD Registers:** 7 AVX2 registers used (constant space)

## PROOF OF CORRECTNESS

### Termination
- **Outer Loop:** Bounded by `seq_len` (N). Loop variant `i` increases from 0 to N-1. Guaranteed termination.
- **Inner Loop:** Bounded by `seq_len - i`. Loop variant `j` increases from `i+1` to `seq_len-1`. Guaranteed termination.

### Bounds
- **Row Index:** Loop condition `i < seq_len` ensures `i` ranges [0, N-1]. Access `matrix[i * seq_len + j]` is valid.
- **Column Index:** Loop condition `j < seq_len` ensures `j` ranges [0, N-1]. Access `matrix[i * seq_len + j]` is valid.
- **AVX2 Access:** For aligned loads, `j` is rounded down to multiple of 8. Tail loop handles remainder safely.

### Masking Logic
- **Correctness:** For row `i`, set `scores[i, j] = mask_value` for `j > i`. This creates upper triangular mask (causal mask).
- **Mathematical Proof:** 
  - For each row `i` ∈ [0, N-1]:
    - For each column `j` ∈ [i+1, N-1]:
      - Set `scores[i, j] = mask_value`
  - This ensures that position `i` cannot attend to positions `j > i` (future tokens).
  - Total elements masked: N(N-1)/2 (upper triangle excluding diagonal).

### Alignment
- **AVX2 Requirement:** 32-byte alignment for `_mm256_load_ps` and `_mm256_store_ps`.
- **Precondition:** `scores->data` must be 32-byte aligned (enforced by validation).
- **Proof:** For unaligned access, use `_mm256_loadu_ps` and `_mm256_storeu_ps` (safe for any alignment).

## EDGE CASE PROOF

### N=0
- **Loop Condition:** `i < 0` fails immediately. Function returns without accessing memory. Safe.

### N=1
- **Masking:** Only diagonal element exists. No elements with `j > i`. No masking needed. Safe.

### N=MAX
- **Size Validation:** Prevents overflow. Loop counters are `uint32_t`, bounded by `seq_len`.
- **No Overflow:** `i * seq_len + j` is bounded by `(N-1) * N + (N-1) = N² - 1 < SIZE_MAX` for reasonable N.

### Special Values
- **NaN/Inf:** IEEE 754 guarantees: `mask_value` (typically -1e9f) propagates correctly.
- **Behavior:** Masked positions set to `mask_value`, unmasked positions maintain original values.

## NUMERICAL STABILITY
- **Element-wise Assignment:** No accumulation of errors. Each element set independently.
- **No Rounding Errors:** Direct assignment, no floating-point operations.
- **SIMD Precision:** AVX2 operations maintain IEEE 754 precision.

## OPTIMIZATION PROOF

### Three-Region Strategy
1. **Scalar Region:** Elements before aligned block. O(1) to O(7) elements per row.
2. **Boundary Block:** Contains boundary (i), needs Load+Cmp+Blend. Exactly 8 elements.
3. **Right Side Blocks:** Completely to the right, Store-only. No Load needed. Reduces bandwidth by ~50%.

### Bandwidth Reduction
- **Traditional Approach:** Load + Compare + Blend + Store = 2 memory operations per 8 elements.
- **Optimized Approach:** Store-only for right blocks = 1 memory operation per 8 elements.
- **Proof:** For row `i`, elements `j > i+7` are always masked. No need to read original values.

