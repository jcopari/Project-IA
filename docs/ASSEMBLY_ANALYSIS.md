# Assembly Analysis Guide

This document describes how to analyze the generated assembly code to verify optimizations in Qorus-IA v2.0.

## Purpose

Assembly analysis helps verify that:
1. **SIMD Instructions:** AVX2/FMA instructions are being generated
2. **Register Usage:** No excessive register spilling
3. **Loop Unrolling:** Compiler is unrolling loops when beneficial
4. **Inlining:** Critical functions are being inlined
5. **Instruction Selection:** Optimal instructions are chosen

## Tools

### 1. GCC Assembly Output

Generate assembly with Intel syntax and source annotations:

```bash
gcc -S -masm=intel -fverbose-asm -O3 -mavx2 -mfma \
    -std=c11 -I./include -D_GNU_SOURCE \
    src/ops/avx2/matmul.c -o matmul.s
```

### 2. objdump

Disassemble object files:

```bash
objdump -d -M intel build/ops/avx2/matmul.o | less
```

### 3. Automated Script

Use the provided script:

```bash
./tools/analyze_assembly.sh
```

## What to Look For

### AVX2 Instructions

**Good Signs:**
- `vmulps`, `vaddps`, `vsubps` - Vector multiply/add/subtract
- `vfmadd132ps`, `vfmadd213ps`, `vfmadd231ps` - Fused multiply-add
- `vmaxps`, `vminps` - Vector max/min
- `vrsqrtps` - Vector reciprocal square root
- `vbroadcastss` - Broadcast scalar to vector

**Example:**
```asm
vmulps  ymm0, ymm1, ymm2    ; ymm0 = ymm1 * ymm2
vfmadd231ps ymm3, ymm0, ymm4 ; ymm3 = ymm3 + ymm0 * ymm4
```

### Register Spilling

**Bad Signs:**
- Excessive `mov [rsp+offset], reg` - Storing registers to stack
- Excessive `mov reg, [rsp+offset]` - Loading registers from stack

**Good Signs:**
- Minimal stack usage
- Most operations use registers directly

**Example (Bad):**
```asm
mov     [rsp+16], ymm0    ; Spilling ymm0 to stack
; ... many instructions ...
mov     ymm0, [rsp+16]    ; Reloading from stack
```

### Loop Unrolling

**Good Signs:**
- Multiple similar instruction sequences
- Loop body processes multiple elements
- Reduced loop overhead

**Example:**
```asm
.L2:
    vfmadd231ps ymm0, ymm1, ymm2
    vfmadd231ps ymm3, ymm4, ymm5
    vfmadd231ps ymm6, ymm7, ymm8
    vfmadd231ps ymm9, ymm10, ymm11
    add     rdx, 128
    cmp     rdx, rcx
    jne     .L2
```

### Inlining

**Good Signs:**
- No function prologue (`push rbp`, `mov rbp, rsp`)
- No function epilogue (`pop rbp`, `ret`)
- Code appears directly in caller

**Example (Inlined):**
```asm
; Caller code
vmulps  ymm0, ymm1, ymm2
vaddps  ymm0, ymm0, ymm3
; No function call overhead
```

**Example (Not Inlined):**
```asm
call    q_gemv_q4_f32_avx2
; Function prologue in called function
push    rbp
mov     rbp, rsp
; ... function body ...
pop     rbp
ret
```

## Analysis Checklist

For each critical kernel, verify:

- [ ] AVX2 instructions are present
- [ ] FMA instructions are used (if applicable)
- [ ] Register spilling is minimal (< 10 instances)
- [ ] Loops are unrolled (if beneficial)
- [ ] Functions are inlined (if marked `inline`)
- [ ] No unnecessary memory accesses
- [ ] Horizontal reductions use efficient sequences

## Common Issues

### Issue: No AVX2 Instructions

**Symptoms:** Only SSE or scalar instructions

**Causes:**
- Missing `-mavx2` flag
- Code path not taken (dead code elimination)
- Function not called

**Fix:** Ensure `-mavx2 -mfma` flags are present

### Issue: Excessive Register Spilling

**Symptoms:** Many `mov [rsp+offset], reg` instructions

**Causes:**
- Too many live variables
- Large function with many local variables
- Compiler unable to optimize

**Fix:** Break function into smaller pieces, reduce local variables

### Issue: Functions Not Inlined

**Symptoms:** Function calls present in hot path

**Causes:**
- Missing `inline` keyword
- Function too large
- Function called from multiple translation units

**Fix:** Use `static inline`, ensure function is in header or marked `inline`

## Performance Indicators

### Good Performance Indicators

1. **High SIMD Utilization:** Most operations use AVX2 registers
2. **FMA Usage:** Fused multiply-add reduces instruction count
3. **Minimal Spilling:** Most operations stay in registers
4. **Loop Unrolling:** Reduces loop overhead
5. **Inlining:** Eliminates function call overhead

### Bad Performance Indicators

1. **Scalar Code:** Operations on single elements instead of vectors
2. **Excessive Spilling:** Many stack operations
3. **No Unrolling:** Small loops with high overhead
4. **Function Calls:** Overhead in hot path
5. **Memory Accesses:** Unnecessary loads/stores

## Example Analysis

### MatMul Kernel

**Expected:**
- AVX2 instructions: ✓
- FMA instructions: ✓
- Register spilling: Minimal (< 5 instances)
- Loop unrolling: 4x (4 accumulators)
- Inlining: ✓ (marked `static inline`)

**Assembly Snippet:**
```asm
.L2:
    vfmadd231ps ymm0, ymm1, ymm2    ; Accumulator 0
    vfmadd231ps ymm3, ymm4, ymm5    ; Accumulator 1
    vfmadd231ps ymm6, ymm7, ymm8    ; Accumulator 2
    vfmadd231ps ymm9, ymm10, ymm11  ; Accumulator 3
    add     rdx, 128
    cmp     rdx, rcx
    jne     .L2
```

**Analysis:** ✓ Excellent - 4 accumulators, FMA instructions, minimal overhead

## Automated Analysis

Run the analysis script:

```bash
./tools/analyze_assembly.sh
```

This will:
1. Generate assembly files for all critical kernels
2. Check for AVX2/FMA instructions
3. Analyze register spilling
4. Detect loop unrolling
5. Verify inlining

Results are saved to `assembly_analysis/` directory.

## References

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [x86-64 Assembly Guide](https://www.cs.virginia.edu/~evans/cs216/guides/x86.html)
- [GCC Optimization Options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)

