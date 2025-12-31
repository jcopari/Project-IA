# QORUS-IA v2.0: PLANNING SUMMARY
# Complete Planning Documentation

**Status:** ✅ PLANNING COMPLETE  
**Date:** 2024-12-30  
**Methodology:** MFR + CoT + Mathematical Proof + TDD

---

## DOCUMENTS CREATED

### 1. KERNEL_PORTATION_PLAN.md
**Purpose:** Complete execution plan following MFR + CoT + Mathematical Proof + TDD framework

**Contents:**
- Executive summary
- Architectural adaptations (MetaIA → New-QorusIA mapping)
- Complete planning for 4 critical kernels:
  1. MatMul FP32 AVX2
  2. Causal Masking AVX2
  3. Tensor Add AVX2
  4. Element-wise Mul AVX2
- Each kernel includes:
  - Step 0: Chain of Thought (CoT) analysis
  - Step 0.5: Mathematical Proof (complexity, correctness, edge cases)
  - Step 1: Model Construction (MFR Phase 1)
  - Step 2: Test-Driven Development (TDD)
  - Step 3: Implementation (MFR Phase 2)
  - Step 4: Validation & Verification
- Implementation order & timeline
- Validation checklist

### 2. KERNEL_IMPLEMENTATION_DETAILS.md
**Purpose:** Complete implementation guide with full code examples

**Contents:**
- Complete function signatures for all 4 kernels
- Full C implementation code for each kernel
- Integration examples showing how to use each kernel in `llama_forward()`
- Header updates needed (`include/qorus.h`)
- Makefile updates needed
- Test structure guidelines

### 3. PLANNING_SUMMARY.md (this document)
**Purpose:** Overview of all planning documents and next steps

---

## KERNELS TO IMPLEMENT

### Priority: 🔴 CRITICAL (Blockers for Forward Pass)

1. **MatMul FP32 AVX2** (`q_matmul_f32_avx2`)
   - **Use Cases:** Q @ K^T, probs @ V, LM Head projection
   - **Complexity:** O(M × N × K)
   - **Estimated Time:** 4-6 hours
   - **File:** `src/ops/avx2/matmul_fp32.c`

2. **Causal Masking AVX2** (`q_causal_mask_f32_avx2`)
   - **Use Cases:** Attention triangular mask
   - **Complexity:** O(N²)
   - **Estimated Time:** 2-3 hours
   - **File:** `src/ops/avx2/causal_mask.c`

3. **Tensor Add AVX2** (`q_add_f32_avx2`)
   - **Use Cases:** Residual connections (`x = x + attn_out`)
   - **Complexity:** O(N)
   - **Estimated Time:** 2-3 hours
   - **File:** `src/ops/avx2/add.c`

4. **Element-wise Mul AVX2** (`q_mul_f32_avx2`)
   - **Use Cases:** SwiGLU activation (`gate * up`)
   - **Complexity:** O(N)
   - **Estimated Time:** 2-3 hours
   - **File:** `src/ops/avx2/mul.c`

**Total Estimated Time:** 14-21 hours

---

## ARCHITECTURAL ADAPTATIONS

### MetaIA → New-QorusIA Mapping

| Component | MetaIA | New-QorusIA | Status |
|-----------|--------|-------------|--------|
| Tensor Structure | `t_tensor` | `q_tensor` | ✅ Defined |
| Error Handling | `int` (0=success) | `q_error_code` enum | ✅ Defined |
| Memory Allocation | `malloc`/`tensor_create` | `q_arena_alloc` | ✅ Defined |
| Validation | `#ifdef DEBUG` | Always active | ✅ Defined |
| Naming | `tensor_*` | `q_*` | ✅ Defined |
| Alignment | Partial | 64-byte mandatory | ✅ Defined |

---

## IMPLEMENTATION ORDER

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

---

## VALIDATION REQUIREMENTS

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

## FILES TO CREATE

### Source Files
- `src/ops/avx2/matmul_fp32.c` - MatMul FP32 implementation
- `src/ops/avx2/causal_mask.c` - Causal masking implementation
- `src/ops/avx2/add.c` - Tensor Add implementation
- `src/ops/avx2/mul.c` - Element-wise Mul implementation

### Test Files
- `tests/validation/validate_matmul_f32.c` - MatMul FP32 validation
- `tests/validation/validate_causal_mask.c` - Causal masking validation
- `tests/validation/validate_add_f32.c` - Tensor Add validation
- `tests/validation/validate_mul_f32.c` - Element-wise Mul validation

### Python Test Data Generation
- `scripts/gen_test_data.py` - Update with new test cases for all 4 kernels

### Header Updates
- `include/qorus.h` - Add function declarations for all 4 kernels

### Makefile Updates
- `Makefile` - Add new source files to build system

---

## NEXT STEPS

1. ✅ **Planning Complete** - All planning documents created
2. ⏳ **Implementation Phase 1** - Implement MatMul FP32 AVX2
   - Create `src/ops/avx2/matmul_fp32.c`
   - Write tests (`tests/validation/validate_matmul_f32.c`)
   - Update `include/qorus.h`
   - Update `Makefile`
   - Run tests and validate
3. ⏳ **Implementation Phase 2** - Implement remaining 3 kernels
   - Causal Masking AVX2
   - Tensor Add AVX2
   - Element-wise Mul AVX2
4. ⏳ **Integration Phase** - Integrate into `q_model_forward()` (generic)
   - Update `src/core/model.c` (generic framework)
   - End-to-end validation
   - Performance benchmarking

---

## REFERENCE DOCUMENTS

### MetaIA Source Files (Reference)
- `metaIA/src/math/avx/ft_matmul_avx.c` - MatMul FP32 implementation
- `metaIA/src/math/avx/ft_attention_avx.c` - Causal masking implementation
- `metaIA/src/math/avx/ft_tensor_add_avx.c` - Tensor Add implementation

### New-QorusIA Documentation
- `docs/.cursorrules` - Framework definition (MFR + CoT + Proof + TDD)
- `docs/KERNEL_PORTATION_PLAN.md` - Complete planning document
- `docs/KERNEL_IMPLEMENTATION_DETAILS.md` - Implementation guide
- `docs/FASE_3.3_ANALYSIS.md` - Forward pass analysis
- `MASTER_BLUEPRINT.md` - Architecture blueprint

---

## FRAMEWORK COMPLIANCE

All planning follows the strict framework defined in `.cursorrules`:

✅ **MFR (Model-First Reasoning):**
- Data structures defined before implementation
- Constraints specified upfront
- Function prototypes defined

✅ **CoT (Chain of Thought):**
- Problem understanding explicit
- Decomposition into sub-problems
- Logical flow documented
- Edge cases identified

✅ **Mathematical Proof:**
- Time complexity analyzed
- Space complexity analyzed
- Correctness proven
- Edge cases proven
- Numerical stability analyzed

✅ **TDD (Test-Driven Development):**
- Python gold standard defined
- C validation tests specified
- Test cases cover happy path + edge cases + error conditions

---

## SUCCESS CRITERIA

Planning is considered complete when:

- [x] All 4 kernels have complete planning documents
- [x] Each kernel follows MFR + CoT + Proof + TDD framework
- [x] Implementation details are specified
- [x] Test structure is defined
- [x] Integration examples are provided
- [x] Timeline and order are established

**Status:** ✅ **PLANNING COMPLETE**

---

**Last Updated:** 2024-12-30  
**Next Phase:** Implementation (Phase 1: MatMul FP32 AVX2)

