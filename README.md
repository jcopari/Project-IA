# Qorus-IA v3.0

High-Performance Generic Deep Learning Framework in Pure C - No Architectural Limitations.

**Evolution:**
- **v2.0:** Specialized inference engine
- **v3.0:** Generic deep learning framework (any architecture) maintaining performance and clean architecture

**Priorities:** Performance (zero-malloc, AVX2), Flexibility (any architecture), Clean Architecture (robust validations).

**Constraint:** Zero-Malloc in Hot Path (maintained).

**Status:** 
- **v2.0:** FASE 2 & 3.2 Complete - All mathematical kernels and model graph building implemented and validated. Training capability planning complete (2024-12-30).
- **v3.0:** Generic framework planning complete (2024-12-30). Ready for FASE 5.0 (Core Abstraction).

## Features

### v2.0 (Current - Specialized)
- **Pure C11** implementation (no dependencies)
- **AVX2/FMA** optimized kernels (6/6 implemented, 4 planned)
- **Q4_0 quantization** support
- **Zero-copy** memory management (mmap)
- **Cache-aligned** data structures (128-byte layers)
- **Memory-safe** (AddressSanitizer validated)
- **Training capability** (planned - for custom model fine-tuning)
- **Limited to specific architecture**

### v3.0 (Planned - Generic Framework)
- ✅ **All v2.0 features** maintained
- ✅ **Generic layer abstraction** (polymorphism via function pointers)
- ✅ **Any architecture support** (LLM, MLP, CNN, RNN, custom)
- ✅ **Flexible model composition** (easy to build any model)
- ✅ **No architectural limitations** (do whatever you want)
- ✅ **Easy to extend** (add new layer types easily)

## Implemented Components

### Mathematical Kernels (FASE 2)
✅ **FASE 2.1:** Q4_0 Dequantization (AVX2, fused FMA)  
✅ **FASE 2.2:** MatMul Q4_F32 (fused dequantization, 4x unrolling)  
✅ **FASE 2.3:** RMSNorm & RoPE (AVX2 optimized)  
✅ **FASE 2.4:** SiLU & Softmax (polynomial exp approximation)

### Model Architecture (FASE 3)
✅ **FASE 3.2:** Model Graph Building (`q_model_build_graph()`)
- Zero-copy tensor views from mmap'd `.qorus` files
- Complete model structure initialization
- Comprehensive validation and adversarial testing

### Testing & Tools
✅ **Utility Tests:** `q_strerror()` validation (23 tests)  
✅ **AVX Math Tests:** Polynomial approximation validation (13 tests)  
✅ **Benchmark Tool:** End-to-end performance measurement  
✅ **Documentation:** Asymptotic analysis, assembly analysis guides

### Code Quality & Robustness
✅ **Robust Pointer Arithmetic:** `size_t` for all offset calculations  
✅ **Multi-Layer Overflow Protection:** Validation at multiple critical points  
✅ **Comprehensive Testing:** 128+ test cases, 100% pass rate  
✅ **Security Validations:** Always-active checks (Release + Debug modes)

## Architecture

Qorus-IA uses a three-tier memory architecture:

- **Tier 1 (Static):** Model weights via `mmap` (read-only, zero-copy)
- **Tier 2 (Persistent):** KV Cache (aligned allocation)
- **Tier 3 (Transient):** Scratchpad Arena (reset per token)

## Build

```bash
# Release build (optimized, -O3 -mavx2 -mfma)
make

# Debug build (with AddressSanitizer + UndefinedBehaviorSanitizer)
make DEBUG=1

# Run all tests (Release + Debug)
make test-validation

# Run specific tests
make test-memory
make test-dequantize
make test-utils
make test-avx-math

# Benchmark kernels
make benchmark

# Analyze assembly code
make analyze-assembly
```

## Quick Start

```c
#include "qorus.h"

q_context ctx = {0};

// Initialize memory (load model)
q_error_code err = q_init_memory(&ctx, "model.qorus");
if (err != Q_OK) {
    fprintf(stderr, "Error: %s\n", q_strerror(err));
    return 1;
}

// Allocate KV cache
err = q_alloc_kv_cache(&ctx, 512 * 1024 * 1024); // 512MB
if (err != Q_OK) {
    fprintf(stderr, "Error: %s\n", q_strerror(err));
    q_free_memory(&ctx);
    return 1;
}

// Allocate scratchpad arena
err = q_alloc_arena(&ctx, 512 * 1024 * 1024); // 512MB
if (err != Q_OK) {
    fprintf(stderr, "Error: %s\n", q_strerror(err));
    q_free_memory(&ctx);
    return 1;
}

// ... use model for inference ...

// Cleanup
q_free_memory(&ctx);
```

## Requirements

- **Compiler:** GCC or Clang with C11 support
- **CPU:** x86-64 with AVX2 and FMA support
- **OS:** Linux, macOS, or BSD

## Project Structure

```
qorus-ia/
├── include/          # Public headers (qorus.h, qorus_types.h)
├── src/
│   ├── core/        # Memory management, tensor utilities
│   ├── ops/          # Mathematical kernels (AVX2 optimized)
│   ├── models/       # Model architectures (Llama-3)
│   └── tokenizer/    # Text processing (BPE)
├── tools/            # Conversion scripts (Python)
├── tests/            # Test suite
└── docs/             # Documentation
```

## Documentation

### Executive Documents
- `docs/PROJECT_VISION.md` - **Complete project vision (start → current → finish)**
- `docs/TIMELINE.md` - **Development timeline with estimates and dependencies**
- `docs/INDEX.md` - **Master documentation index - navigation guide**

### Current Status
- `docs/STATUS.md` - **Current project status and progress**
- `docs/QUICK_REFERENCE.md` - Quick reference (what's done, what's missing)

### Architecture & Planning
- `MASTER_BLUEPRINT.md` - **Architecture and implementation roadmap (updated for v3.0)**
- `docs/GENERIC_FRAMEWORK_PLAN.md` - **Complete generic framework plan v3.0 (MFR + CoT + Proof + TDD)**
- `docs/TRAINING_CAPABILITY_PLAN.md` - **Complete training capability plan (MFR + CoT + Proof + TDD)**
- `docs/KERNEL_PORTATION_PLAN.md` - **Complete kernel portation plan (MFR + CoT + Proof + TDD)**
- `docs/KERNEL_IMPLEMENTATION_DETAILS.md` - **Implementation guide with full code examples**
- `docs/PLANNING_SUMMARY.md` - **Planning overview and next steps**

### Quality & Process
- `docs/REFACTORING_CHECKPOINTS.md` - **Refactoring checkpoint procedures and quality assurance**
- `docs/.cursorrules` - Development methodology (MFR + CoT + Proof + TDD)
- `docs/PRECISION_STANDARDS.md` - Numerical precision standards
- `docs/MELHORIAS_ROBUSTEZ.md` - Robustness improvements documentation
- `docs/PERFORMANCE_REPORT.md` - Performance benchmarks and test results
- `docs/CORRECOES_APLICADAS.md` - Applied corrections documentation
- `docs/ASYMPTOTIC_ANALYSIS.md` - Complexity analysis for all functions
- `docs/ASSEMBLY_ANALYSIS.md` - Assembly code analysis guide

## Current Status

### v2.0 (Current - Specialized Engine)
**✅ Completed:** FASE 1 (Infrastructure) + FASE 2 (Mathematical Kernels) + FASE 2.5 (MetaIA Kernel Portation) + FASE 3.2 (Model Graph Building)  
**✅ Robustness Improvements:** Enhanced pointer arithmetic, multi-layer overflow protection (2025-12-31)  
**✅ FASE 2.5 Complete:** All critical kernels ported from MetaIA v1.4.0 (2025-12-31)
- MatMul FP32 AVX2, Causal Masking AVX2, Tensor Add AVX2, Element-wise Mul AVX2
- All kernels implemented, tested, validated, and code-reviewed
**✅ Planning Complete:** 
- Kernel portation plan from MetaIA v1.4.0 (2024-12-30)
- Training capability plan for future-implementations (2024-12-30)  
**🚧 Next:** 
- FASE 3.3 (Forward Pass - in progress)
- FASE 2.6 (Training Kernels) + FASE 3.4 (Backward Pass) + FASE 3.5 (Training Loop)

### v3.0 (Planned - Generic Framework)
**✅ Planning Complete:** Generic framework plan (2024-12-30)  
**🚧 Next:** 
- FASE 5.0 (Core Abstraction - Generic Layer Interface, Model Container)
- FASE 5.1 (Basic Layers - Linear, Activation, Normalization, Softmax)
- FASE 5.2 (Advanced Layers - MHA, FFN, Transformer Block, Embedding)
- FASE 5.3 (Llama-3 Migration to Generic Framework)
- FASE 5.4 (Additional Architectures - MLP, CNN, RNN)

### Implemented Kernels (FASE 2.5 - Inference)
The following critical kernels have been successfully ported from MetaIA v1.4.0:
- ✅ **MatMul FP32 AVX2** - Q @ K^T, probs @ V, LM Head projection
- ✅ **Causal Masking AVX2** - Attention triangular mask
- ✅ **Tensor Add AVX2** - Residual connections
- ✅ **Element-wise Mul AVX2** - SwiGLU activation

**Status:** All kernels implemented, tested, validated, and code-reviewed (2025-12-31)

See `docs/KERNEL_PORTATION_PLAN.md` for complete planning details.

### Planned Training Components (FASE 2.6)
The following training components are planned for custom model training:
- **Optimizers** (Adam, AdamW) - AVX2-optimized weight updates
- **Loss Functions** (MSE, CrossEntropy) - AVX2-optimized loss computation
- **Gradient Clipping** - Training stabilization
- **Backward Pass** (FASE 3.4) - Gradient propagation
- **Training Loop** (FASE 3.5) - Complete training pipeline

See `docs/TRAINING_CAPABILITY_PLAN.md` for complete planning details.

See `docs/STATUS.md` for detailed progress.

## Error Handling

All functions return `q_error_code` (enum). Use `q_strerror()` to convert to string:

```c
q_error_code err = q_init_memory(&ctx, "model.qorus");
if (err != Q_OK) {
    fprintf(stderr, "Error: %s\n", q_strerror(err));
}
```

## License

[To be added]

## Contributing

[To be added]
