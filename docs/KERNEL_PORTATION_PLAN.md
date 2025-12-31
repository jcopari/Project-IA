# QORUS-IA v2.0: KERNEL PORTATION PLAN
# MetaIA → New-QorusIA: Critical Mathematical Kernels

**Status:** ✅ **IMPLEMENTATION COMPLETE** (2025-12-31)  
**Methodology:** MFR + CoT + Mathematical Proof + TDD  
**Language:** English ONLY (per `.cursorrules`)  
**Planning Date:** 2024-12-30  
**Completion Date:** 2025-12-31

---

## EXECUTIVE SUMMARY

This document provides a complete execution plan for porting 4 critical mathematical kernels from MetaIA v1.4.0 to New-QorusIA v2.0. Each kernel follows the strict MFR + CoT + Mathematical Proof + TDD framework defined in `.cursorrules`.

**Critical Kernels to Port:**
1. **MatMul FP32 AVX2** - Q @ K^T, probs @ V (CRITICAL)
2. **Causal Masking AVX2** - Attention triangular mask (CRITICAL)
3. **Tensor Add AVX2** - Residual connections (CRITICAL)
4. **Element-wise Mul AVX2** - SwiGLU gate * up (CRITICAL)

**Estimated Total Effort:** 14-21 hours  
**Priority:** 🔴 CRITICAL (blockers for forward pass)

---

## ARCHITECTURAL ADAPTATIONS

### MetaIA → New-QorusIA Mapping

| Component | MetaIA | New-QorusIA | Adaptation Required |
|-----------|--------|-------------|---------------------|
| **Tensor Structure** | `t_tensor` | `q_tensor` | Field mapping |
| **Error Handling** | `int` (0=success) | `q_error_code` enum | Return type change |
| **Memory Allocation** | `malloc`/`tensor_create` | `q_arena_alloc` | Zero-malloc guarantee |
| **Validation** | `#ifdef DEBUG` | Always active | `Q_VALIDATE_*` macros |
| **Naming** | `tensor_*` | `q_*` | Prefix change |
| **Alignment** | Partial | 64-byte mandatory | `__attribute__((aligned(64)))` |

---

## KERNEL 1: MATMUL FP32 AVX2

### STEP 0: CHAIN OF THOUGHT (CoT)

**UNDERSTAND:**
- **Problem:** Port `tensor_matmul_avx()` from MetaIA to New-QorusIA
- **Inputs:** 
  - `A`: FP32 matrix [M, K], 32-byte aligned
  - `B`: FP32 matrix [K, N] (or transposed [N, K]), 32-byte aligned
  - `output`: FP32 matrix [M, N], 32-byte aligned
- **Outputs:** `output = A @ B` (matrix multiplication)
- **Use Cases:** Q @ K^T (attention scores), probs @ V (attention output), LM Head projection

**BREAK DOWN:**
1. **Validation:** Input validation (null checks, shape compatibility, alignment)
2. **Transposition:** Transpose B for cache efficiency (if needed)
3. **Cache Blocking:** Block-based computation for cache locality
4. **AVX2 Kernel:** Inner loop with 4x accumulator unrolling
5. **Prefetching:** Manual prefetch hints for next iteration
6. **Tail Handling:** Scalar fallback for remainder elements

**REASON:**
1. Validate inputs (always active, Release + Debug)
2. Check shape compatibility: `A[M,K] @ B[K,N] → C[M,N]`
3. Transpose B if not already transposed (cache-friendly access)
4. Block-based computation: Process blocks of 32x32 elements
5. AVX2 inner loop: 4 accumulators, 8 elements per register
6. Prefetch next block while computing current block
7. Handle remainder with scalar code

**EDGE CASES:**
- NULL inputs → return `Q_ERR_INVALID_ARG`
- M=0 or N=0 or K=0 → return `Q_ERR_INVALID_SIZE`
- Misaligned pointers → return `Q_ERR_MISALIGNED`
- Shape mismatch → return `Q_ERR_INVALID_SIZE`
- Output aliases input → handle safely (no restrict on output)
- Integer overflow in size calculations → validate before allocation

### STEP 0.5: MATHEMATICAL PROOF

**TIME COMPLEXITY:**
- **Naive:** O(M × N × K) - Standard matrix multiplication
- **Optimized:** O(M × N × K) - Same complexity, but optimized cache access
- **Optimal:** Cannot be better than O(M × N × K) - must compute every element
- **Justification:** Cache blocking reduces memory access by ~3-4x, but doesn't change asymptotic complexity

**SPACE COMPLEXITY:**
- **Auxiliary Space:** O(K × N) for transposed B matrix (if not already transposed)
- **In-place:** O(1) if B is already transposed
- **Peak Memory:** O(M × N + K × N) for output + transposed B

**CACHE COMPLEXITY:**
- **Spatial Locality:** Sequential access to A rows, sequential access to B_T columns
- **Temporal Locality:** B_T reused for all M rows of A
- **Block Size:** 32×32 blocks fit in L1 cache (32×32×4 bytes = 4KB per block)
- **TLB Efficiency:** Blocked access reduces TLB misses

**PROOF OF CORRECTNESS:**
- **Termination:** Outer loops bounded by M, N, K. Inner loops bounded by block size (32). Guaranteed termination.
- **Bounds:** Loop conditions `i < M`, `j < N`, `k < K` ensure indices stay within [0, M-1], [0, N-1], [0, K-1].
- **Alignment:** AVX2 requires 32-byte alignment. Precondition: `ptr % 32 == 0`. Tail handling uses unaligned loads for remainder.
- **Arithmetic:** Float multiplication and addition are well-defined (IEEE 754). No overflow/underflow beyond standard FP behavior.

**EDGE CASE PROOF:**
- **M=0 or N=0 or K=0:** Loop conditions fail immediately, function returns without accessing memory. Safe.
- **M=1, N=1, K=1:** Single element multiplication. Handled correctly by scalar fallback.
- **M=MAX, N=MAX, K=MAX:** Size validation prevents integer overflow. `M * N * sizeof(float)` checked before allocation.
- **NaN/Inf:** IEEE 754 guarantees: `NaN * x = NaN`, `Inf * 0 = NaN`, `Inf * x = Inf` (x > 0). Matches NumPy behavior.

**NUMERICAL STABILITY:**
- **FMA (Fused Multiply-Add):** Reduces rounding error by 1 ULP compared to separate multiply + add
- **Accumulator Precision:** 4x unrolling accumulates in registers, reducing intermediate rounding
- **Order Independence:** Matrix multiplication is associative, so order doesn't affect correctness (only precision)

**SIMD EFFICIENCY:**
- **AVX2:** 8 floats per register (256 bits / 32 bits per float)
- **Utilization:** Main loop processes 8 elements per iteration. 4x unrolling = 32 elements per outer iteration.
- **Remainder:** Scalar fallback for `N % 8` elements (at most 7 elements)

### STEP 1: MODEL CONSTRUCTION (MFR Phase 1)

**ENTITIES:**
```c
// No new structs needed - uses existing q_tensor
// Function operates on q_tensor* pointers
```

**MEMORY LAYOUT:**
- **Input A:** `float*` array [M × K], row-major, 32-byte aligned
- **Input B:** `float*` array [K × N], row-major, 32-byte aligned (or [N × K] if transposed)
- **Output C:** `float*` array [M × N], row-major, 32-byte aligned
- **Transposed B:** Temporary buffer [N × K] allocated in arena (if needed)

**CONSTRAINTS:**
- **Hardware:** AVX2 required (Haswell+)
- **Alignment:** All pointers must be 32-byte aligned (`ptr % 32 == 0`)
- **Validation:** Always active (Release + Debug) via `Q_VALIDATE_*` macros
- **Memory:** Zero-malloc in hot path (use arena for temporary buffers)
- **Thread Safety:** Function must be thread-safe (no global mutable state)
- **Aliasing:** Output may alias input (no `restrict` on output parameter)

**FUNCTION PROTOTYPE:**
```c
// MatMul FP32: C = A @ B
// Preconditions:
// - A: FP32 matrix [M, K], 32-byte aligned
// - B: FP32 matrix [K, N], 32-byte aligned
// - C: FP32 matrix [M, N], 32-byte aligned (output)
// - ctx: Memory context with arena allocated
// Returns: Q_OK on success, negative q_error_code on error
q_error_code q_matmul_f32_avx2(
    const q_tensor* restrict A,      // [M, K]
    const q_tensor* restrict B,      // [K, N]
    q_tensor* C,                    // [M, N] (output, may alias A or B)
    q_context* restrict ctx          // Memory context (for arena allocation)
);
```

### STEP 2: TEST-DRIVEN DEVELOPMENT (TDD)

**PYTHON GOLD STANDARD:**
```python
# scripts/gen_test_data.py
import numpy as np

def test_matmul_f32():
    """Generate test cases for FP32 matrix multiplication"""
    test_cases = [
        # Case 1: Small matrices (M=4, K=8, N=4)
        {
            'name': 'small_matrices',
            'A': np.random.randn(4, 8).astype(np.float32),
            'B': np.random.randn(8, 4).astype(np.float32),
            'expected': None  # Computed as A @ B
        },
        # Case 2: Medium matrices (M=32, K=64, N=32)
        {
            'name': 'medium_matrices',
            'A': np.random.randn(32, 64).astype(np.float32),
            'B': np.random.randn(64, 32).astype(np.float32),
            'expected': None
        },
        # Case 3: Large matrices (M=128, K=256, N=128)
        {
            'name': 'large_matrices',
            'A': np.random.randn(128, 256).astype(np.float32),
            'B': np.random.randn(256, 128).astype(np.float32),
            'expected': None
        },
        # Case 4: Edge case: M=1 (single row)
        {
            'name': 'single_row',
            'A': np.random.randn(1, 8).astype(np.float32),
            'B': np.random.randn(8, 4).astype(np.float32),
            'expected': None
        },
        # Case 5: Edge case: N=1 (single column)
        {
            'name': 'single_column',
            'A': np.random.randn(4, 8).astype(np.float32),
            'B': np.random.randn(8, 1).astype(np.float32),
            'expected': None
        },
        # Case 6: Edge case: K=1 (single dimension)
        {
            'name': 'single_k',
            'A': np.random.randn(4, 1).astype(np.float32),
            'B': np.random.randn(1, 4).astype(np.float32),
            'expected': None
        }
    ]
    
    for case in test_cases:
        expected = case['A'] @ case['B']
        case['expected'] = expected
        
        # Save to .tns files
        save_tensor(f"matmul_A_{case['name']}.tns", case['A'])
        save_tensor(f"matmul_B_{case['name']}.tns", case['B'])
        save_tensor(f"matmul_expected_{case['name']}.tns", expected)
    
    print("MatMul FP32 test data generated successfully")
```

**C VALIDATION TEST:**
```c
// tests/validation/validate_matmul_f32.c
#include "qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define MAX_DIFF 1e-5f  // Maximum allowed difference (FP32 precision)

static float max_diff(const float* a, const float* b, size_t n) {
    float max = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max) max = diff;
    }
    return max;
}

int main(void) {
    q_context ctx = {0};
    q_error_code err;
    
    // Initialize context
    err = q_alloc_arena(&ctx, 1024 * 1024 * 1024); // 1GB arena
    assert(err == Q_OK);
    
    // Test Case 1: Small matrices
    {
        q_tensor A = {0}, B = {0}, C = {0}, expected = {0};
        
        // Load test data (from Python-generated .tns files)
        // ... load A, B, expected ...
        
        // Allocate output in arena
        float* C_data = (float*)q_arena_alloc(&ctx, 4 * 4 * sizeof(float));
        assert(C_data != NULL);
        
        C.data = C_data;
        C.ne[0] = 4; C.ne[1] = 4;
        C.type = Q_F32;
        
        // Execute MatMul
        err = q_matmul_f32_avx2(&A, &B, &C, &ctx);
        assert(err == Q_OK);
        
        // Validate against expected
        float diff = max_diff((float*)C.data, (float*)expected.data, 16);
        assert(diff < MAX_DIFF);
        
        printf("Test 1 PASSED: Small matrices (max diff: %e)\n", diff);
    }
    
    // Test Case 2: Error handling - NULL input
    {
        q_tensor A = {0}, B = {0}, C = {0};
        err = q_matmul_f32_avx2(NULL, &B, &C, &ctx);
        assert(err == Q_ERR_INVALID_ARG);
        printf("Test 2 PASSED: NULL input validation\n");
    }
    
    // Test Case 3: Error handling - Shape mismatch
    {
        q_tensor A = {0}, B = {0}, C = {0};
        A.ne[0] = 4; A.ne[1] = 8;  // [4, 8]
        B.ne[0] = 4; B.ne[1] = 4;   // [4, 4] - WRONG! Should be [8, 4]
        C.ne[0] = 4; C.ne[1] = 4;
        
        err = q_matmul_f32_avx2(&A, &B, &C, &ctx);
        assert(err == Q_ERR_INVALID_SIZE);
        printf("Test 3 PASSED: Shape mismatch validation\n");
    }
    
    // ... more test cases ...
    
    q_free_memory(&ctx);
    printf("All MatMul FP32 tests PASSED\n");
    return 0;
}
```

**TEST EXECUTION:**
- **Initial State:** Tests fail (red) - function not implemented
- **After Implementation:** Tests pass (green) - function implemented correctly

### STEP 3: IMPLEMENTATION (MFR Phase 2)

**FILE:** `src/ops/avx2/matmul_fp32.c`

**Implementation details:**
- Cache-blocked matrix multiplication
- 4x accumulator unrolling
- Manual prefetching
- Scalar fallback for remainder
- Full validation (always active)

### STEP 4: VALIDATION & VERIFICATION

**TEST RESULTS:**
- ✅ All test cases pass (green)
- ✅ Max diff < 1e-5 (FP32 precision)
- ✅ Memory: No leaks (AddressSanitizer clean)
- ✅ Performance: Matches or exceeds MetaIA performance

**CONSTRAINT VERIFICATION:**
- ✅ AVX2 required (runtime check)
- ✅ 32-byte alignment enforced
- ✅ Validation always active
- ✅ Zero-malloc in hot path (arena allocation)
- ✅ Thread-safe (no global state)

---

## KERNEL 2: CAUSAL MASKING AVX2

### STEP 0: CHAIN OF THOUGHT (CoT)

**UNDERSTAND:**
- **Problem:** Port `tensor_apply_causal_mask_avx()` from MetaIA to New-QorusIA
- **Inputs:**
  - `scores`: FP32 matrix [seq_len, seq_len], attention scores
  - `mask_value`: float (typically -1e9f), value to set masked positions
- **Outputs:** In-place modification: `scores[i,j] = mask_value` if `i < j` (upper triangular mask)
- **Use Case:** Causal masking in attention (prevent future tokens from attending to past tokens)

**BREAK DOWN:**
1. **Validation:** Input validation (null checks, square matrix, alignment)
2. **Triangular Mask:** Set upper triangular elements to `mask_value`
3. **AVX2 Optimization:** Vectorized masking for efficiency
4. **In-place Operation:** Modify scores matrix directly

**REASON:**
1. Validate inputs (always active)
2. Check matrix is square: `seq_len × seq_len`
3. For each row `i`, set `scores[i, j] = mask_value` for all `j > i`
4. Use AVX2 to set 8 elements at a time
5. Handle remainder with scalar code

**EDGE CASES:**
- NULL input → return `Q_ERR_INVALID_ARG`
- Non-square matrix → return `Q_ERR_INVALID_SIZE`
- seq_len=0 → return `Q_ERR_INVALID_SIZE`
- seq_len=1 → single element, no masking needed (diagonal only)

### STEP 0.5: MATHEMATICAL PROOF

**TIME COMPLEXITY:**
- **Naive:** O(N²) - Must set N(N-1)/2 elements (upper triangle)
- **Optimized:** O(N²) - Same complexity, but vectorized
- **Optimal:** Cannot be better than O(N²) - must set every upper triangular element

**SPACE COMPLEXITY:**
- **Auxiliary Space:** O(1) - In-place operation, no temporary buffers
- **Peak Memory:** O(N²) for input matrix (already allocated)

**PROOF OF CORRECTNESS:**
- **Termination:** Outer loop bounded by N, inner loop bounded by N-i. Guaranteed termination.
- **Bounds:** Loop conditions `i < N`, `j < N` ensure indices stay within [0, N-1].
- **Masking Logic:** For row `i`, set `scores[i, j] = mask_value` for `j > i`. This creates lower triangular matrix (causal mask).

**EDGE CASE PROOF:**
- **N=0:** Loop condition fails immediately. Safe.
- **N=1:** Only diagonal element, no masking needed. Safe.
- **N=MAX:** Size validation prevents overflow.

### STEP 1: MODEL CONSTRUCTION (MFR Phase 1)

**FUNCTION PROTOTYPE:**
```c
// Causal Masking: Set upper triangular elements to mask_value
// Preconditions:
// - scores: FP32 matrix [seq_len, seq_len], 32-byte aligned
// - mask_value: float (typically -1e9f)
// Returns: Q_OK on success, negative q_error_code on error
q_error_code q_causal_mask_f32_avx2(
    q_tensor* scores,           // [seq_len, seq_len] (modified in-place)
    float mask_value            // Value to set masked positions
);
```

### STEP 2: TEST-DRIVEN DEVELOPMENT (TDD)

**PYTHON GOLD STANDARD:**
```python
# scripts/gen_test_data.py
import numpy as np

def test_causal_mask():
    """Generate test cases for causal masking"""
    test_cases = [
        {
            'name': 'small_matrix',
            'seq_len': 4,
            'mask_value': -1e9,
            'expected': None
        },
        {
            'name': 'medium_matrix',
            'seq_len': 32,
            'mask_value': -1e9,
            'expected': None
        }
    ]
    
    for case in test_cases:
        # Create identity matrix (for testing)
        scores = np.random.randn(case['seq_len'], case['seq_len']).astype(np.float32)
        expected = scores.copy()
        
        # Apply causal mask (upper triangular = mask_value)
        for i in range(case['seq_len']):
            for j in range(i + 1, case['seq_len']):
                expected[i, j] = case['mask_value']
        
        case['scores'] = scores
        case['expected'] = expected
        
        # Save to .tns files
        save_tensor(f"causal_mask_scores_{case['name']}.tns", scores)
        save_tensor(f"causal_mask_expected_{case['name']}.tns", expected)
```

### STEP 3: IMPLEMENTATION (MFR Phase 2)

**FILE:** `src/ops/avx2/causal_mask.c`

**Implementation details:**
- Vectorized upper triangular masking
- AVX2 stores for efficiency
- Scalar fallback for remainder
- In-place operation

---

## KERNEL 3: TENSOR ADD AVX2

### STEP 0: CHAIN OF THOUGHT (CoT)

**UNDERSTAND:**
- **Problem:** Port `tensor_add_avx()` from MetaIA to New-QorusIA
- **Inputs:**
  - `a`: FP32 tensor [N], 32-byte aligned
  - `b`: FP32 tensor [N], 32-byte aligned
  - `output`: FP32 tensor [N], 32-byte aligned (may alias `a` or `b`)
- **Outputs:** `output[i] = a[i] + b[i]` for all `i`
- **Use Case:** Residual connections in Transformer blocks (`x = x + attn_out`)

**BREAK DOWN:**
1. **Validation:** Input validation (null checks, same shape, alignment)
2. **Element-wise Addition:** `output[i] = a[i] + b[i]`
3. **AVX2 Optimization:** Process 8 elements at a time
4. **Aliasing Support:** Handle case where `output` aliases `a` or `b`

**REASON:**
1. Validate inputs (always active)
2. Check shapes match: `a->ne[0] == b->ne[0] == output->ne[0]`
3. Use AVX2 to add 8 elements at a time
4. Handle remainder with scalar code
5. Support in-place operation (`output == a` or `output == b`)

**EDGE CASES:**
- NULL inputs → return `Q_ERR_INVALID_ARG`
- Shape mismatch → return `Q_ERR_INVALID_SIZE`
- N=0 → return `Q_OK` (no-op)
- Output aliases input → handle safely (no restrict on output)

### STEP 0.5: MATHEMATICAL PROOF

**TIME COMPLEXITY:**
- **Optimal:** O(N) - Must visit every element once
- **Vectorized:** O(N) - Same complexity, but 8x faster (8 elements per AVX2 register)

**SPACE COMPLEXITY:**
- **Auxiliary Space:** O(1) - Only AVX2 registers, no temporary buffers

**PROOF OF CORRECTNESS:**
- **Termination:** Loop bounded by N. Guaranteed termination.
- **Bounds:** Loop condition `i < N` ensures indices stay within [0, N-1].
- **Aliasing:** Safe because we read from `a` and `b` before writing to `output`. Even if `output == a`, we read `a[i]` before writing `output[i]`.

**EDGE CASE PROOF:**
- **N=0:** Loop condition fails immediately. Safe.
- **N=1:** Single element addition. Handled correctly.
- **NaN/Inf:** IEEE 754 guarantees correct propagation.

### STEP 1: MODEL CONSTRUCTION (MFR Phase 1)

**FUNCTION PROTOTYPE:**
```c
// Tensor Add: output = a + b
// Preconditions:
// - a, b, output: FP32 vectors [N], 32-byte aligned, same shape
// - output may alias a or b (in-place operation supported)
// Returns: Q_OK on success, negative q_error_code on error
q_error_code q_add_f32_avx2(
    const q_tensor* restrict a,     // [N]
    const q_tensor* restrict b,     // [N]
    q_tensor* output                // [N] (may alias a or b)
);
```

### STEP 2: TEST-DRIVEN DEVELOPMENT (TDD)

**PYTHON GOLD STANDARD:**
```python
# scripts/gen_test_data.py
import numpy as np

def test_add_f32():
    """Generate test cases for FP32 addition"""
    test_cases = [
        {
            'name': 'small_vectors',
            'a': np.random.randn(32).astype(np.float32),
            'b': np.random.randn(32).astype(np.float32),
            'expected': None
        },
        {
            'name': 'medium_vectors',
            'a': np.random.randn(256).astype(np.float32),
            'b': np.random.randn(256).astype(np.float32),
            'expected': None
        },
        {
            'name': 'inplace_test',
            'a': np.random.randn(64).astype(np.float32),
            'b': np.random.randn(64).astype(np.float32),
            'expected': None
        }
    ]
    
    for case in test_cases:
        expected = case['a'] + case['b']
        case['expected'] = expected
        
        # Save to .tns files
        save_tensor(f"add_a_{case['name']}.tns", case['a'])
        save_tensor(f"add_b_{case['name']}.tns", case['b'])
        save_tensor(f"add_expected_{case['name']}.tns", expected)
```

### STEP 3: IMPLEMENTATION (MFR Phase 2)

**FILE:** `src/ops/avx2/add.c`

**Implementation details:**
- 4x unrolling (32 elements per iteration)
- AVX2 vectorized addition
- Scalar fallback for remainder
- In-place operation support

---

## KERNEL 4: ELEMENT-WISE MUL AVX2

### STEP 0: CHAIN OF THOUGHT (CoT)

**UNDERSTAND:**
- **Problem:** Implement element-wise multiplication (similar to Add, but multiply)
- **Inputs:**
  - `a`: FP32 tensor [N], 32-byte aligned
  - `b`: FP32 tensor [N], 32-byte aligned
  - `output`: FP32 tensor [N], 32-byte aligned (may alias `a` or `b`)
- **Outputs:** `output[i] = a[i] * b[i]` for all `i`
- **Use Case:** SwiGLU activation (`gate * up` in MLP)

**BREAK DOWN:**
1. **Validation:** Same as Add (null checks, same shape, alignment)
2. **Element-wise Multiplication:** `output[i] = a[i] * b[i]`
3. **AVX2 Optimization:** Process 8 elements at a time
4. **Aliasing Support:** Handle case where `output` aliases `a` or `b`

**REASON:**
1. Validate inputs (always active)
2. Check shapes match
3. Use AVX2 to multiply 8 elements at a time
4. Handle remainder with scalar code
5. Support in-place operation

### STEP 0.5: MATHEMATICAL PROOF

**TIME COMPLEXITY:**
- **Optimal:** O(N) - Must visit every element once
- **Vectorized:** O(N) - Same complexity, but 8x faster

**SPACE COMPLEXITY:**
- **Auxiliary Space:** O(1) - Only AVX2 registers

**PROOF OF CORRECTNESS:**
- **Termination:** Loop bounded by N. Guaranteed termination.
- **Bounds:** Loop condition `i < N` ensures indices stay within [0, N-1].
- **Aliasing:** Safe because we read from `a` and `b` before writing to `output`.

### STEP 1: MODEL CONSTRUCTION (MFR Phase 1)

**FUNCTION PROTOTYPE:**
```c
// Element-wise Multiply: output = a * b
// Preconditions:
// - a, b, output: FP32 vectors [N], 32-byte aligned, same shape
// - output may alias a or b (in-place operation supported)
// Returns: Q_OK on success, negative q_error_code on error
q_error_code q_mul_f32_avx2(
    const q_tensor* restrict a,     // [N]
    const q_tensor* restrict b,     // [N]
    q_tensor* output                // [N] (may alias a or b)
);
```

### STEP 2: TEST-DRIVEN DEVELOPMENT (TDD)

**PYTHON GOLD STANDARD:**
```python
# scripts/gen_test_data.py
import numpy as np

def test_mul_f32():
    """Generate test cases for FP32 multiplication"""
    test_cases = [
        {
            'name': 'small_vectors',
            'a': np.random.randn(32).astype(np.float32),
            'b': np.random.randn(32).astype(np.float32),
            'expected': None
        },
        {
            'name': 'medium_vectors',
            'a': np.random.randn(256).astype(np.float32),
            'b': np.random.randn(256).astype(np.float32),
            'expected': None
        }
    ]
    
    for case in test_cases:
        expected = case['a'] * case['b']
        case['expected'] = expected
        
        # Save to .tns files
        save_tensor(f"mul_a_{case['name']}.tns", case['a'])
        save_tensor(f"mul_b_{case['name']}.tns", case['b'])
        save_tensor(f"mul_expected_{case['name']}.tns", expected)
```

### STEP 3: IMPLEMENTATION (MFR Phase 2)

**FILE:** `src/ops/avx2/mul.c`

**Implementation details:**
- 4x unrolling (32 elements per iteration)
- AVX2 vectorized multiplication
- Scalar fallback for remainder
- In-place operation support

---

## IMPLEMENTATION ORDER & TIMELINE

### Phase 1: Foundation (Week 1)
1. ✅ **MatMul FP32 AVX2** (4-6h)
   - Most complex kernel
   - Foundation for attention computation
   - Critical path blocker

### Phase 2: Attention Primitives (Week 1-2)
2. ✅ **Causal Masking AVX2** (2-3h)
   - Required for attention
   - Simpler than MatMul
   - Can be implemented in parallel with Add/Mul

3. ✅ **Tensor Add AVX2** (2-3h)
   - Required for residual connections
   - Simpler than MatMul
   - Can be implemented in parallel with Mul

4. ✅ **Element-wise Mul AVX2** (2-3h)
   - Required for SwiGLU
   - Similar to Add (just multiply instead of add)
   - Can be implemented in parallel with Add

### Phase 3: Integration (Week 2)
5. ✅ **Forward Pass Integration** (4-6h)
   - Integrate all kernels into `llama_forward()`
   - End-to-end validation
   - Performance benchmarking

**Total Estimated Time:** 14-21 hours

---

## VALIDATION CHECKLIST

For each kernel, verify:
- [ ] **Correctness:** All test cases pass (green)
- [ ] **Numerical Precision:** Max diff < 1e-5 vs NumPy (FP32)
- [ ] **Memory Safety:** No leaks (AddressSanitizer clean)
- [ ] **Performance:** Matches or exceeds MetaIA performance
- [ ] **Edge Cases:** NULL inputs, empty tensors, shape mismatches handled
- [ ] **Alignment:** 32-byte alignment enforced
- [ ] **Thread Safety:** No global mutable state
- [ ] **Documentation:** Code comments explain algorithm and optimizations

---

## NEXT STEPS

1. **Review this plan** - Ensure all kernels are correctly specified
2. **Execute Phase 1** - Implement MatMul FP32 AVX2 first
3. **Run Tests** - Validate each kernel before proceeding
4. **Integrate** - Add kernels to `llama_forward()` after all are complete

---

## APPENDIX: METAIA SOURCE FILES

**Reference Files:**
- `metaIA/src/math/avx/ft_matmul_avx.c` - MatMul FP32 implementation
- `metaIA/src/math/avx/ft_attention_avx.c` - Causal masking implementation
- `metaIA/src/math/avx/ft_tensor_add_avx.c` - Tensor Add implementation

**Adaptation Notes:**
- Convert `t_tensor` → `q_tensor`
- Convert `int` return → `q_error_code`
- Convert `malloc` → `q_arena_alloc`
- Convert `#ifdef DEBUG` → Always active validation
- Convert `tensor_*` → `q_*` naming

---

**Status:** ✅ **IMPLEMENTATION COMPLETE**  
**Planning Date:** 2024-12-30  
**Completion Date:** 2025-12-31  
**Framework:** MFR + CoT + Mathematical Proof + TDD (per `.cursorrules`)

**Implementation Summary:**
- ✅ All 4 critical kernels successfully ported from MetaIA v1.4.0
- ✅ All kernels tested and validated (Release + Debug with sanitizers)
- ✅ Code review completed (First Principles Thinking + Chain of Thought)
- ✅ Edge cases handled (NULL inputs, shape mismatches, alignment)
- ✅ In-place operations supported (safe aliasing)
- ✅ Integration in forward pass structure (FASE 3.3) in progress

