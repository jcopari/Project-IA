# Element-wise Mul AVX2: Mathematical Proof

## TIME COMPLEXITY
- **Optimal:** O(N) - Must visit every element once
- **Vectorized:** O(N) - Same complexity, but 8x faster (8 elements per AVX2 register)
- **4x Unrolling:** Processes 32 elements per iteration (4x8), maximizing throughput

## SPACE COMPLEXITY
- **Auxiliary Space:** O(1) - Only AVX2 registers, no temporary buffers
- **Peak Memory:** O(N) for input/output tensors (already allocated)

## PROOF OF CORRECTNESS

### Termination
- **Loop:** Bounded by N. Loop variant `i` increases from 0 to N-1 (or N-32 for vectorized). Guaranteed termination.

### Bounds
- **Index Validity:** Loop condition `i < N` ensures indices stay within [0, N-1].
- **AVX2 Access:** For vectorized loop, `i` ranges [0, N-32] (rounded down to multiple of 32). Tail loop handles remainder safely.

### Aliasing Safety
- **Critical:** Output may alias input (`output == a` or `output == b`).
- **Proof:** We read from `a[i]` and `b[i]` before writing to `output[i]`. Even if `output == a`, we read `a[i]` into AVX2 register before writing `output[i]`. Safe.
- **Implementation:** Do NOT use `restrict` on any parameter to allow aliasing.

### Element-wise Multiplication
- **Correctness:** `output[i] = a[i] * b[i]` for all `i` ∈ [0, N-1].
- **Mathematical Proof:** 
  - For each element `i`:
    - Load `a[i]` and `b[i]` into AVX2 registers
    - Compute `a[i] * b[i]` using `_mm256_mul_ps`
    - Store result to `output[i]`
  - This ensures `output[i] = a[i] * b[i]` for all elements.

## EDGE CASE PROOF

### N=0
- **Loop Condition:** `i < 0` fails immediately. Function returns without accessing memory. Safe.

### N=1
- **Vectorized Loop:** Skipped (1 < 32). Scalar tail loop handles single element. Safe.

### N=MAX
- **Size Validation:** Prevents overflow. Loop counters are `uint32_t`, bounded by `N`.
- **No Overflow:** `i + 32 <= N` ensures no buffer overflows.

### NaN/Inf Propagation
- **IEEE 754:** Guarantees correct propagation: `NaN * x = NaN`, `Inf * x = Inf` (unless x is 0).

### Aliasing Cases
- **output == a:** Safe because we read `a[i]` before writing `output[i]`.
- **output == b:** Safe because we read `b[i]` before writing `output[i]`.
- **output == a == b:** Safe because we read both before writing.

## NUMERICAL STABILITY
- **Element-wise Multiplication:** No accumulation of errors. Each operation is independent.
- **No Rounding Errors:** Direct multiplication, no intermediate accumulations.
- **SIMD Precision:** AVX2 operations maintain IEEE 754 precision.

## OPTIMIZATION PROOF

### 4x Unrolling Strategy
- **Throughput:** Processes 32 elements per iteration (4x8 AVX2 registers).
- **Register Pressure:** Uses 12 AVX2 registers (4 for a, 4 for b, 4 for output). Within limits (16 AVX2 registers available).
- **Cache Efficiency:** Sequential access pattern maximizes spatial locality.

### Vectorization Efficiency
- **SIMD Utilization:** 8 elements per AVX2 register = 100% utilization.
- **Tail Handling:** Scalar fallback for remainder (N % 32) elements.

