# QORUS-IA v3.0: GENERIC FRAMEWORK PLAN
# From Specialized Engine to Generic Deep Learning Framework

**Status:** 📋 PLANNING PHASE  
**Methodology:** MFR + CoT + Mathematical Proof + TDD  
**Language:** English ONLY  
**Date:** 2024-12-30

---

## EXECUTIVE SUMMARY

This document provides a complete plan to transform QorusIA v2.0 from a specialized Llama-3 inference engine into a **generic deep learning framework** (QorusIA v3.0) that combines:

- **MetaIA's Genericity:** Support for multiple architectures (LLM, MLP, CNN, etc.)
- **QorusIA's Performance:** Zero-malloc, AVX2 optimized, cache-aligned
- **QorusIA's Clean Architecture:** Robust validations, standardized error handling

**Strategic Goal:** Create a high-performance, generic deep learning framework in pure C with no architectural limitations.

**Key Principle:** Maintain QorusIA's performance and clean architecture while adding generic layer abstraction.

---

## STRATEGIC RATIONALE

### Why Transform to Generic Framework?

**Current Limitations (QorusIA v2.0):**
- ❌ Limited to specific architecture only
- ❌ Cannot support other architectures (CNN, RNN, custom)
- ❌ Hardcoded model structure
- ❌ Difficult to experiment with new architectures

**Desired Capabilities:**
- ✅ Support any architecture (LLM, MLP, CNN, custom)
- ✅ Easy to add new layer types
- ✅ Flexible model composition
- ✅ No architectural limitations
- ✅ Maintain high performance

**Solution:**
- Add generic layer abstraction (polymorphism)
- Maintain zero-malloc performance
- Keep clean architecture
- Enable flexible model building

---

## ARCHITECTURAL DESIGN: GENERIC FRAMEWORK

### Design Philosophy

**Core Principles:**
1. **Generic Layer Abstraction:** Polymorphism via function pointers (vtable pattern)
2. **Performance First:** Zero-malloc in hot path, AVX2 optimized
3. **Clean Architecture:** Robust validations, standardized errors
4. **Flexible Composition:** Easy to build any architecture

**Key Insight:** Generic abstraction doesn't mean sacrificing performance. With proper design, we can have both.

---

## ARCHITECTURE SPECIFICATION

### 1. Generic Layer Interface

```c
// include/qorus_types.h

/**
 * Generic forward function type (Virtual Method)
 * 
 * @param context Pointer to real layer struct (e.g., q_layer_linear*)
 * @param input Input tensor
 * @param output Output tensor (pre-allocated)
 * @param ctx Memory context (arena, mmap)
 * @param training_mode If true, cache inputs for backward
 * @return q_error_code
 */
typedef q_error_code (*q_forward_func)(
    void* context,
    const q_tensor* input,
    q_tensor* output,
    q_context* ctx,
    bool training_mode
);

/**
 * Generic backward function type (Backpropagation)
 * 
 * @param context Pointer to real layer struct
 * @param input Original input tensor (from forward cache)
 * @param grad_output Gradient from next layer
 * @param grad_input Output gradient (pre-allocated)
 * @param ctx Memory context
 * @return q_error_code
 */
typedef q_error_code (*q_backward_func)(
    void* context,
    const q_tensor* input,
    const q_tensor* grad_output,
    q_tensor* grad_input,
    q_context* ctx
);

/**
 * Generic free function type (Virtual Destructor)
 */
typedef void (*q_free_func)(void* context);

/**
 * Layer types enum
 */
typedef enum {
    Q_LAYER_UNKNOWN = 0,
    Q_LAYER_LINEAR = 1,
    Q_LAYER_ACTIVATION = 2,
    Q_LAYER_SOFTMAX = 3,
    Q_LAYER_DROPOUT = 4,
    Q_LAYER_RMSNORM = 5,
    Q_LAYER_LAYERNORM = 6,
    Q_LAYER_BATCHNORM = 7,
    Q_LAYER_EMBEDDING = 8,
    Q_LAYER_MHA = 9,
    Q_LAYER_FFN = 10,
    Q_LAYER_TRANSFORMER_BLOCK = 11,
    Q_LAYER_CONV2D = 12,        // Future
    Q_LAYER_POOL2D = 13,        // Future
    Q_LAYER_RNN = 14,           // Future
    Q_LAYER_LSTM = 15           // Future
} q_layer_type;

/**
 * Generic Layer Interface (64-byte aligned)
 * 
 * Design:
 * - Function pointers for polymorphism (vtable pattern)
 * - Context pointer for type-specific data
 * - Type enum for runtime type checking
 * - Cache-aligned for performance
 * 
 * Size: 64 bytes (cache-aligned)
 */
typedef struct {
    void* context;              // 8 bytes - Type-specific data (Linear, MHA, etc.)
    q_forward_func forward;     // 8 bytes - Forward function pointer
    q_backward_func backward;   // 8 bytes - Backward function pointer (NULL if not trainable)
    q_free_func free;           // 8 bytes - Free function pointer
    q_layer_type type;          // 4 bytes - Layer type enum
    uint32_t in_dim;            // 4 bytes - Input dimension
    uint32_t out_dim;           // 4 bytes - Output dimension
    uint32_t flags;             // 4 bytes - Layer flags (trainable, etc.)
    char name[16];              // 16 bytes - Layer name (for debugging)
} __attribute__((aligned(64))) q_layer;
```

### 2. Generic Model Container

```c
/**
 * Generic Model Container
 * 
 * Design:
 * - Array of generic layers (polymorphic)
 * - Forward cache for training (arena-allocated)
 * - Zero-copy support via mmap
 * - Cache-aligned for performance
 * 
 * Size: 128 bytes (cache-aligned)
 */
typedef struct {
    q_layer** layers;           // 8 bytes - Array of layer pointers
    uint32_t count;             // 4 bytes - Number of layers
    uint32_t capacity;          // 4 bytes - Array capacity
    
    // Training support
    q_tensor** forward_cache;   // 8 bytes - Cache for backward pass [count+1]
    bool training_mode;          // 1 byte - Training flag
    char _padding1[7];          // 7 bytes - Padding
    
    // Memory management
    q_context* ctx;             // 8 bytes - Memory context (arena, mmap)
    
    // Model metadata
    char name[32];              // 32 bytes - Model name
    
    // Statistics (for debugging/optimization)
    uint64_t forward_calls;     // 8 bytes - Forward pass call count
    uint64_t backward_calls;    // 8 bytes - Backward pass call count
    
    // Padding to 128 bytes
    char _padding2[40];
} __attribute__((aligned(128))) q_model;
```

### 3. Layer-Specific Contexts

```c
// include/qorus_layers.h

/**
 * Linear Layer Context
 */
typedef struct {
    q_tensor* weights;          // Weight matrix [out_dim, in_dim]
    q_tensor* bias;             // Bias vector [out_dim] (optional, NULL if none)
    q_dtype weight_dtype;      // Q4_0 or FP32
    
    // Gradients (training only, arena-allocated)
    q_tensor* grad_weights;
    q_tensor* grad_bias;
    
    // Initialization
    q_init_scheme init_scheme;  // He, Xavier, etc.
} q_layer_linear;

/**
 * Multi-Head Attention Layer Context
 */
typedef struct {
    uint32_t embed_dim;
    uint32_t num_heads;
    uint32_t num_kv_heads;     // GQA support (num_kv_heads <= num_heads)
    uint32_t head_dim;
    
    // Projections
    q_layer_linear* q_proj;
    q_layer_linear* k_proj;
    q_layer_linear* v_proj;
    q_layer_linear* o_proj;
    
    // Workspace buffers (persistent, arena-allocated)
    q_tensor* qkv_buffer;
    q_tensor* attn_scores;
    q_tensor* attn_probs;
} q_layer_mha;

/**
 * Feed-Forward Network Layer Context
 */
typedef struct {
    uint32_t embed_dim;
    uint32_t hidden_dim;
    
    // Projections (SwiGLU: gate, up, down)
    q_layer_linear* gate_proj;  // SwiGLU gate
    q_layer_linear* up_proj;    // SwiGLU up
    q_layer_linear* down_proj;   // Down projection
    
    // Workspace buffers
    q_tensor* gate_buffer;
    q_tensor* up_buffer;
    q_tensor* swiglu_buffer;
} q_layer_ffn;

/**
 * Transformer Block Context
 */
typedef struct {
    q_layer_rmsnorm* norm1;
    q_layer_mha* mha;
    q_layer_rmsnorm* norm2;
    q_layer_ffn* ffn;
    
    // Workspace buffers
    q_tensor* residual_buffer;
} q_layer_transformer_block;
```

---

## API DESIGN

### Generic Model API

```c
// include/qorus.h

/**
 * Create generic model
 */
q_error_code q_model_create(
    q_model* model,
    q_context* ctx,
    const char* name
);

/**
 * Add layer to model (polymorphic)
 */
q_error_code q_model_add_layer(
    q_model* model,
    q_layer* layer
);

/**
 * Forward pass (generic, works with any architecture)
 */
q_error_code q_model_forward(
    q_model* model,
    const q_tensor* input,
    q_tensor* output,
    bool training_mode
);

/**
 * Backward pass (generic, works with any architecture)
 */
q_error_code q_model_backward(
    q_model* model,
    const q_tensor* loss_gradient
);

/**
 * Training loop (generic)
 */
q_error_code q_model_train(
    q_model* model,
    const q_tensor* x,
    const q_tensor* y,
    q_optimizer* optimizer,
    uint32_t epochs,
    uint32_t batch_size,
    q_loss_type loss_type
);

/**
 * Free model and all layers
 */
void q_model_free(q_model* model);
```

### Layer Creation APIs

```c
/**
 * Create Linear Layer
 */
q_layer* q_layer_linear_create(
    uint32_t in_dim,
    uint32_t out_dim,
    q_dtype dtype,
    bool use_bias,
    q_init_scheme init_scheme,
    q_context* ctx
);

/**
 * Create Multi-Head Attention Layer
 */
q_layer* q_layer_mha_create(
    uint32_t embed_dim,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    q_context* ctx
);

/**
 * Create Feed-Forward Network Layer
 */
q_layer* q_layer_ffn_create(
    uint32_t embed_dim,
    uint32_t hidden_dim,
    q_context* ctx
);

/**
 * Create Transformer Block
 */
q_layer* q_layer_transformer_block_create(
    uint32_t embed_dim,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t hidden_dim,
    q_context* ctx
);
```

---

## IMPLEMENTATION ROADMAP

### Phase 1: Core Abstraction (Week 1-2)

**Objective:** Implement generic layer and model infrastructure.

**Tasks:**
1. **Generic Layer Interface** (4-6h)
   - Define `q_layer` structure
   - Implement layer creation/destruction
   - Add validation functions

2. **Generic Model Container** (6-8h)
   - Implement `q_model` structure
   - Implement `q_model_create()`, `q_model_add_layer()`
   - Implement `q_model_free()`

3. **Generic Forward Pass** (4-6h)
   - Implement `q_model_forward()` with polymorphism
   - Forward cache management for training
   - Error handling and validation

4. **Generic Backward Pass** (6-8h)
   - Implement `q_model_backward()` with polymorphism
   - Gradient propagation through layers
   - Forward cache usage

**Total Phase 1:** 20-28 hours

### Phase 2: Basic Layers (Week 2-3)

**Objective:** Implement basic layer types with generic interface.

**Tasks:**
1. **Linear Layer** (6-8h)
   - Convert existing MatMul to generic Linear layer
   - Implement forward/backward with generic interface
   - Support Q4_0 and FP32

2. **Activation Layers** (4-6h)
   - ReLU, GeLU, SiLU, Sigmoid
   - Generic forward/backward interface

3. **Normalization Layers** (6-8h)
   - RMSNorm (existing)
   - LayerNorm (new)
   - BatchNorm (future)

4. **Softmax Layer** (2-3h)
   - Convert existing Softmax to generic layer

**Total Phase 2:** 18-25 hours

### Phase 3: Advanced Layers (Week 3-4)

**Objective:** Implement advanced layer types.

**Tasks:**
1. **Multi-Head Attention** (8-10h)
   - Convert existing MHA to generic layer
   - Support GQA (Grouped Query Attention)
   - Generic forward/backward

2. **Feed-Forward Network** (6-8h)
   - Convert existing FFN to generic layer
   - Support SwiGLU activation
   - Generic forward/backward

3. **Transformer Block** (4-6h)
   - Compose MHA + FFN + RMSNorm
   - Generic forward/backward

4. **Embedding Layer** (4-6h)
   - Token embedding
   - Positional embedding (RoPE)
   - Generic forward/backward

**Total Phase 3:** 22-30 hours

### Phase 4: Example Model Builders (Week 4-5)

**Objective:** Create example models using generic framework.

**Tasks:**
1. **Transformer Model Builder** (6-8h)
   - Create `transformer_build_model()` using generic API
   - Demonstrate framework flexibility
   - Zero-copy weight loading

2. **Example Testing** (4-6h)
   - Test Transformer model with generic framework
   - Verify performance (should be same or better)
   - Validate correctness

3. **Documentation** (2-3h)
   - Create usage examples
   - Update documentation
   - Migration guides

**Total Phase 4:** 12-17 hours

### Phase 5: Additional Architectures (Ongoing)

**Objective:** Support additional architectures.

**Tasks:**
1. **Simple MLP** (4-6h)
   - Example: MNIST classifier
   - Demonstrate generic framework flexibility

2. **CNN Support** (Future)
   - Conv2D layer
   - Pool2D layer
   - CNN architectures

3. **RNN/LSTM Support** (Future)
   - RNN layer
   - LSTM layer
   - Sequence models

**Total Phase 5:** Ongoing

---

## PERFORMANCE CONSIDERATIONS

### Zero-Malloc Guarantee

**Challenge:** Generic abstraction might introduce allocations.

**Solution:**
- All layer contexts allocated in arena (training) or mmap (inference)
- Forward cache pre-allocated in arena
- Workspace buffers persistent and reusable
- No allocations in hot path

### Cache Alignment

**Challenge:** Generic structures must remain cache-aligned.

**Solution:**
- `q_layer`: 64-byte aligned
- `q_model`: 128-byte aligned
- All layer contexts: 64-byte aligned
- Workspace buffers: 64-byte aligned

### Function Pointer Overhead

**Challenge:** Function pointer calls have overhead.

**Solution:**
- Function pointers are predictable (same layer type = same function)
- CPU branch predictor handles this well
- Overhead is negligible compared to computation
- Can inline hot paths if needed (future optimization)

---

## MIGRATION STRATEGY

### Backward Compatibility

**Strategy:** Maintain backward compatibility during migration.

1. **Phase 1-3:** Implement generic framework alongside existing code
2. **Phase 4:** Migrate Llama-3 to use generic framework
3. **Phase 5:** Deprecate old Llama-3 specific code

**Migration Path:**
```c
// Old API (deprecated):
q_error_code model_forward_specific(model_specific* model, ...);

// New API (generic):
q_error_code q_model_forward(q_model* model, ...);

// Example builder (new):
q_error_code transformer_build_model(q_model* model, transformer_config* config);
```

### Gradual Migration

**Strategy:** Migrate one component at a time.

1. Start with basic layers (Linear, Activation)
2. Migrate advanced layers (MHA, FFN)
3. Migrate model container
4. Create example model builders using generic framework

---

## VALIDATION STRATEGY

### Correctness Validation

For each layer type:
- [ ] Forward pass matches reference implementation (NumPy/PyTorch)
- [ ] Backward pass matches reference implementation
- [ ] Gradients are correct (finite differences)
- [ ] Training converges (small test case)

### Performance Validation

- [ ] Generic framework performance >= specialized code
- [ ] Zero-malloc maintained in hot path
- [ ] Cache alignment maintained
- [ ] AVX2 optimizations still effective

### Architecture Validation

- [ ] Can build Transformer models using generic API
- [ ] Can build simple MLP using generic API
- [ ] Can build custom architectures
- [ ] Easy to add new layer types

---

## COMPARISON: v2.0 vs v3.0

### QorusIA v2.0 (Current)
- ✅ High performance (zero-malloc, AVX2)
- ✅ Clean architecture
- ❌ Limited to specific architecture
- ❌ Hardcoded structure

### QorusIA v3.0 (Proposed)
- ✅ High performance (zero-malloc, AVX2)
- ✅ Clean architecture
- ✅ Generic (any architecture)
- ✅ Flexible composition
- ✅ Easy to extend

**Result:** Best of both worlds - MetaIA's flexibility + QorusIA's performance.

---

## NEXT STEPS

1. **Review this plan** - Ensure generic framework aligns with goals
2. **Execute Phase 1** - Implement core abstraction
3. **Execute Phase 2** - Implement basic layers
4. **Execute Phase 3** - Implement advanced layers
5. **Execute Phase 4** - Migrate Llama-3
6. **Execute Phase 5** - Add additional architectures

---

**Status:** 📋 READY FOR IMPLEMENTATION  
**Last Updated:** 2024-12-30  
**Framework:** MFR + CoT + Mathematical Proof + TDD (per `.cursorrules`)

