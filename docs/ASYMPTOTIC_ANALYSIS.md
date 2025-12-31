# Asymptotic Analysis & Preconditions

This document provides comprehensive asymptotic analysis and preconditions for all critical functions in Qorus-IA v2.0.

## Mathematical Notation

- **Time Complexity:** $O(f(n))$ - Upper bound on execution time
- **Space Complexity:** $O(f(n))$ - Auxiliary space (excluding input/output)
- **Cache Complexity:** Analysis of spatial/temporal locality
- **SIMD Efficiency:** Utilization of vector lanes (8 floats per AVX2 register)

---

## Memory Management Functions

### `q_init_memory()`

**Time Complexity:** $O(1)$ - Constant time syscall  
**Space Complexity:** $O(1)$ - No auxiliary allocations  
**Preconditions:**
- `ctx != NULL`
- `model_path != NULL` and points to valid file
- File size >= `Q_HEADER_SIZE` (64 bytes)
- File contains valid magic number (`Q_MAGIC`)

**Proof of Correctness:**
- `mmap()` syscall is O(1) - creates virtual memory mapping
- `madvise()` is O(1) - hint to kernel
- Magic number validation is O(1) - single comparison
- No loops or recursion → guaranteed termination

**Edge Cases:**
- File doesn't exist → Returns `Q_ERR_FILE_OPEN`
- File too small → Returns `Q_ERR_FILE_TOO_SMALL`
- Invalid magic → Returns `Q_ERR_INVALID_MAGIC`
- `mmap()` failure → Returns `Q_ERR_MMAP_FAILED`

---

### `q_alloc_kv_cache()`

**Time Complexity:** $O(1)$ - Single `aligned_alloc()` call  
**Space Complexity:** $O(kv\_size)$ - Allocates `kv_size` bytes  
**Preconditions:**
- `ctx != NULL`
- `ctx->kv_buffer == NULL` (not already allocated)
- `kv_size` can be safely aligned without overflow

**Proof of Correctness:**
- `safe_align_size()` is O(1) - bitwise operations
- `aligned_alloc()` is O(1) - single allocation
- `memset()` is O(kv_size) but typically optimized by kernel
- No loops → guaranteed termination

**Edge Cases:**
- `ctx == NULL` → Returns `Q_ERR_INVALID_ARG`
- Already allocated → Returns `Q_ERR_INVALID_ARG`
- Overflow in alignment → Returns `Q_ERR_OVERFLOW`
- Allocation failure → Returns `Q_ERR_ALLOC_FAILED`

---

### `q_alloc_arena()`

**Time Complexity:** $O(1)$ - Single `aligned_alloc()` call  
**Space Complexity:** $O(arena\_size)$ - Allocates `arena_size` bytes  
**Preconditions:**
- `ctx != NULL`
- `ctx->scratch_buffer == NULL` (not already allocated)
- `arena_size` can be safely aligned without overflow

**Proof of Correctness:** Same as `q_alloc_kv_cache()`

**Edge Cases:** Same as `q_alloc_kv_cache()`

---

### `q_arena_alloc()`

**Time Complexity:** $O(1)$ - Constant time pointer arithmetic  
**Space Complexity:** $O(1)$ - No auxiliary allocations  
**Preconditions:**
- `ctx != NULL`
- `ctx->scratch_buffer != NULL` (arena initialized)
- `ctx->scratch_head` is aligned to `Q_ALIGN` (64 bytes)
- `size` can be safely aligned without overflow
- `ctx->scratch_head + aligned_size <= ctx->scratch_size` (OOM check)

**Proof of Correctness:**
- All operations are O(1): alignment, addition, comparison
- Bounds check ensures `new_head <= scratch_size`
- Alignment invariant maintained: `ptr % Q_ALIGN == 0`
- No loops → guaranteed termination

**Edge Cases:**
- `ctx == NULL` → Returns `NULL` (Release) or `abort()` (DEBUG)
- Arena not initialized → Returns `NULL`
- Misalignment → Returns `NULL` (prevents AVX2 crash)
- Overflow → Returns `NULL`
- OOM → Returns `NULL`

---

### `q_arena_reset()`

**Time Complexity:** $O(\min(N_{used}, 64KB))$ - Poisoning in DEBUG mode  
**Space Complexity:** $O(1)$ - No allocations  
**Preconditions:**
- `ctx != NULL` (optional - function handles NULL gracefully)

**Proof of Correctness:**
- Poisoning limited to 64KB for performance
- `memset()` is optimized by compiler/kernel
- `scratch_head = 0` is O(1)
- Guaranteed termination

**Edge Cases:**
- `ctx == NULL` → Returns silently (DEBUG may abort)
- Arena not initialized → Sets `scratch_head = 0` safely

---

### `q_free_memory()`

**Time Complexity:** $O(1)$ - Constant time deallocation  
**Space Complexity:** $O(1)$ - No allocations  
**Preconditions:**
- `ctx != NULL` (optional - function handles NULL gracefully)

**Proof of Correctness:**
- LIFO order: arena → kv_cache → mmap
- All deallocations are O(1) syscalls
- Pointers cleared to prevent dangling pointers
- Guaranteed termination

**Edge Cases:**
- `ctx == NULL` → Returns silently (DEBUG may abort)
- Partial allocation → Frees only what was allocated

---

## Mathematical Kernels

### `q_dequantize_q4_0_block_avx2()`

**Time Complexity:** $O(1)$ - Fixed 32-element operation  
**Space Complexity:** $O(1)$ - No allocations (uses SIMD registers)  
**Preconditions:**
- `block != NULL`
- `output != NULL` and 32-byte aligned
- `block` contains valid Q4_0 data

**Proof of Correctness:**
- Fixed number of SIMD operations (no loops)
- Nibble extraction: O(1) - bitwise operations
- FMA operations: O(1) - 4 batches of 8 elements
- Alignment: Precondition ensures `output % 32 == 0`
- Guaranteed termination (no loops)

**SIMD Efficiency:**
- Processes 32 elements using 4 AVX2 registers (8 floats each)
- 100% lane utilization
- Fused multiply-add (FMA) reduces instruction count

**Edge Cases:**
- All zeros → Returns zeros (correct)
- All 0xF (max value) → Returns `(15 - 8) * scale` (correct)

---

### `q_gemv_q4_f32_avx2()`

**Time Complexity:** $O(M \times N)$ where M = rows, N = columns  
**Space Complexity:** $O(1)$ - No allocations (uses SIMD registers)  
**Preconditions:**
- `weights != NULL` and valid Q4_0 tensor
- `input != NULL` and 32-byte aligned, length N
- `output != NULL` and 32-byte aligned, length M
- `input != output` (no aliasing)
- `N % 32 == 0` (block alignment)
- `weights->type == Q_Q4_0`

**Proof of Correctness:**
- Outer loop: M iterations (rows)
- Inner loop: N/32 iterations (blocks per row)
- Each block: O(1) dequantization + FMA
- Total: $O(M \times N/32 \times 32) = O(M \times N)$
- Horizontal reduction: O(1) - fixed 4 accumulators
- Alignment: Preconditions ensure 32-byte alignment
- Termination: Both loops bounded by dimensions

**SIMD Efficiency:**
- Processes 32 weights per AVX2 register
- 4 accumulators enable instruction-level parallelism
- Fused dequantization eliminates intermediate memory writes
- ~100% lane utilization

**Cache Complexity:**
- Spatial locality: Sequential access to `input` (good)
- Temporal locality: `input` reused for all rows (excellent)
- Weight access: Row-major, sequential per row (good)

**Edge Cases:**
- `M == 0` or `N == 0` → Returns `Q_ERR_INVALID_SIZE`
- `N % 32 != 0` → Returns `Q_ERR_INVALID_SIZE`
- Aliasing → Returns `Q_ERR_ALIASING`
- Overflow → Returns `Q_ERR_OVERFLOW`

---

### `q_rmsnorm_f32_avx2()`

**Time Complexity:** $O(N)$ where N = vector length  
**Space Complexity:** $O(1)$ - No allocations (uses SIMD registers)  
**Preconditions:**
- `x != NULL` and 32-byte aligned, length N
- `weight != NULL` and 32-byte aligned, length N
- `output != NULL` and 32-byte aligned, length N
- `N % 8 == 0` (SIMD alignment)
- `N > 0`

**Proof of Correctness:**
- Main loop: N/8 iterations (8 elements per AVX2 register)
- Sum reduction: O(N) - accumulates squares
- Horizontal reduction: O(1) - fixed operations
- Normalization loop: O(N) - element-wise operations
- Total: $O(N)$
- Alignment: Precondition ensures 32-byte alignment
- Termination: Loop bounded by N

**SIMD Efficiency:**
- Processes 8 elements per AVX2 register
- 100% lane utilization
- `rsqrt` + Newton-Raphson: Fast and accurate

**Numerical Stability:**
- Uses `rsqrt` with refinement for accuracy
- Epsilon prevents division by zero
- Max-sub trick not needed (no exp/log)

**Edge Cases:**
- `N == 0` → Returns `Q_ERR_INVALID_SIZE`
- `N % 8 != 0` → Returns `Q_ERR_INVALID_SIZE`
- All zeros → Returns zeros (correct, with epsilon)

---

### `q_rope_f32_avx2()`

**Time Complexity:** $O(N)$ where N = vector length  
**Space Complexity:** $O(1)$ - No allocations (uses SIMD registers)  
**Preconditions:**
- `x != NULL` and 32-byte aligned, length N
- `cos != NULL` and 32-byte aligned, length N/2
- `sin != NULL` and 32-byte aligned, length N/2
- `output != NULL` and 32-byte aligned, length N
- `N % 8 == 0` and `N % 2 == 0` (even, SIMD-aligned)

**Proof of Correctness:**
- Main loop: N/8 iterations (processes 4 complex pairs per iteration)
- Complex rotation: O(1) per pair using `_mm256_addsub_ps`
- Total: $O(N)$
- Alignment: Precondition ensures 32-byte alignment
- Termination: Loop bounded by N

**SIMD Efficiency:**
- Processes 4 complex pairs (8 floats) per AVX2 register
- Uses `_mm256_addsub_ps` for efficient complex arithmetic
- 100% lane utilization

**Edge Cases:**
- `N == 0` → Returns `Q_ERR_INVALID_SIZE`
- `N % 8 != 0` or `N % 2 != 0` → Returns `Q_ERR_INVALID_SIZE`

---

### `q_silu_f32_avx2()`

**Time Complexity:** $O(N)$ where N = vector length  
**Space Complexity:** $O(1)$ - No allocations (uses SIMD registers)  
**Preconditions:**
- `x != NULL` and 32-byte aligned, length N
- `output != NULL` and 32-byte aligned, length N
- `N % 8 == 0` (SIMD alignment)
- `N > 0`

**Proof of Correctness:**
- Main loop: N/8 iterations (8 elements per AVX2 register)
- SiLU = `x * sigmoid(x)` = `x / (1 + exp(-x))`
- Uses `exp_approx_avx()` for approximation
- Total: $O(N)$
- Alignment: Precondition ensures 32-byte alignment
- Termination: Loop bounded by N

**SIMD Efficiency:**
- Processes 8 elements per AVX2 register
- 100% lane utilization
- Polynomial `exp` approximation is fast

**Numerical Stability:**
- Uses polynomial approximation for `exp`
- Clamps to prevent overflow/underflow
- Acceptable precision: ~1e-3 for x in [-2, 2]

**Edge Cases:**
- `N == 0` → Returns `Q_ERR_INVALID_SIZE`
- `N % 8 != 0` → Returns `Q_ERR_INVALID_SIZE`
- Very large x → Clamped to prevent overflow

---

### `q_softmax_f32_avx2()`

**Time Complexity:** $O(N)$ where N = vector length  
**Space Complexity:** $O(1)$ - No allocations (uses SIMD registers)  
**Preconditions:**
- `x != NULL` and 32-byte aligned, length N
- `output != NULL` and 32-byte aligned, length N
- `N % 8 == 0` (SIMD alignment)
- `N > 0`

**Proof of Correctness:**
- Max reduction: O(N) - finds maximum
- Exp loop: O(N) - computes exp(x - max)
- Sum reduction: O(N) - accumulates exponentials
- Normalization loop: O(N) - divides by sum
- Total: $O(N)$
- Alignment: Precondition ensures 32-byte alignment
- Termination: All loops bounded by N

**SIMD Efficiency:**
- Processes 8 elements per AVX2 register
- 100% lane utilization
- Horizontal reductions for max/sum

**Numerical Stability:**
- Max-sub trick: `exp(x - max)` prevents overflow
- Sum validation in DEBUG mode
- Polynomial `exp` approximation is stable

**Edge Cases:**
- `N == 0` → Returns `Q_ERR_INVALID_SIZE`
- `N % 8 != 0` → Returns `Q_ERR_INVALID_SIZE`
- All zeros → Returns uniform distribution (1/N each)

---

## Model Building Functions

### `llama_build_graph()`

**Time Complexity:** $O(L)$ where L = number of layers  
**Space Complexity:** $O(L)$ - Allocates layer structures in arena  
**Preconditions:**
- `ctx != NULL`
- `model != NULL`
- `ctx->weights_mmap != NULL` (mmap initialized)
- `ctx->scratch_buffer != NULL` (arena initialized)
- File contains valid model header
- All tensor offsets are valid and within mmap bounds

**Proof of Correctness:**
- Header validation: O(1)
- Tensor creation loop: L iterations (one per layer)
- Each iteration: O(1) - creates tensor views (no data copy)
- Total: $O(L)$
- Termination: Loop bounded by `n_layers`

**Edge Cases:**
- Invalid config → Returns `Q_ERR_INVALID_CONFIG`
- Offset overflow → Returns `Q_ERR_INVALID_CONFIG`
- Arena OOM → Returns `Q_ERR_ARENA_OOM`
- Invalid tensor dimensions → Returns `Q_ERR_INVALID_CONFIG`

---

## Utility Functions

### `q_strerror()`

**Time Complexity:** $O(1)$ - Array lookup  
**Space Complexity:** $O(1)$ - Constant space (static array)  
**Preconditions:**
- Error code is valid (or out of bounds handled gracefully)

**Proof of Correctness:**
- Array lookup: O(1) - direct indexing
- Bounds check: O(1) - single comparison
- Guaranteed termination

**Edge Cases:**
- Invalid code → Returns "Unknown error"
- `INT_MIN` → Returns "Unknown error" (bounds check)

---

## Summary Table

| Function | Time Complexity | Space Complexity | SIMD Efficiency |
|----------|----------------|-------------------|-----------------|
| `q_init_memory()` | $O(1)$ | $O(1)$ | N/A |
| `q_alloc_kv_cache()` | $O(1)$ | $O(kv\_size)$ | N/A |
| `q_alloc_arena()` | $O(1)$ | $O(arena\_size)$ | N/A |
| `q_arena_alloc()` | $O(1)$ | $O(1)$ | N/A |
| `q_arena_reset()` | $O(\min(N, 64KB))$ | $O(1)$ | N/A |
| `q_free_memory()` | $O(1)$ | $O(1)$ | N/A |
| `q_dequantize_q4_0_block_avx2()` | $O(1)$ | $O(1)$ | 100% |
| `q_gemv_q4_f32_avx2()` | $O(M \times N)$ | $O(1)$ | ~100% |
| `q_rmsnorm_f32_avx2()` | $O(N)$ | $O(1)$ | 100% |
| `q_rope_f32_avx2()` | $O(N)$ | $O(1)$ | 100% |
| `q_silu_f32_avx2()` | $O(N)$ | $O(1)$ | 100% |
| `q_softmax_f32_avx2()` | $O(N)$ | $O(1)$ | 100% |
| `llama_build_graph()` | $O(L)$ | $O(L)$ | N/A |
| `q_strerror()` | $O(1)$ | $O(1)$ | N/A |

---

## Notes

1. **Zero-Malloc Constraint:** All hot-path functions (kernels) maintain O(1) space complexity - no allocations during execution.

2. **SIMD Efficiency:** All AVX2 kernels achieve ~100% lane utilization by processing 8 floats per register.

3. **Cache Locality:** Data-oriented design maximizes spatial/temporal locality.

4. **Alignment Requirements:** All SIMD operations require 32-byte alignment (AVX2) or 64-byte alignment (data structures).

5. **Numerical Stability:** Kernels use proven techniques (max-sub trick, epsilon, clamping) to maintain stability.

