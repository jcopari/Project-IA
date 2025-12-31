# QORUS-IA v2.0: TRAINING CAPABILITY PLAN
# Adding Training Support for Future-Implementations

**Status:** 📋 PLANNING PHASE  
**Methodology:** MFR + CoT + Mathematical Proof + TDD  
**Language:** English ONLY  
**Date:** 2024-12-30

---

## EXECUTIVE SUMMARY

This document provides a complete plan for adding training capability to New-QorusIA v2.0, enabling custom model training for future-implementations (Code Agent, Customer Behavior Prediction, SEO AI Specialist) while maintaining the clean, specialized architecture.

**Strategic Goal:** New-QorusIA v2.0 will be a **hybrid engine** - optimized for inference (primary) with full training capability (secondary) for custom model development.

**Key Principle:** Maintain inference-first architecture while adding training as a complementary capability.

---

## STRATEGIC RATIONALE

### Why Add Training Capability?

**Future-Implementations Requirements:**
1. **Code Agent** - Needs fine-tuning on code datasets
2. **Customer Behavior Prediction** - Needs training on customer data
3. **SEO AI Specialist** - Needs domain-specific fine-tuning
4. **Custom Models** - Need ability to train specialized models

**Current Limitation:**
- New-QorusIA v2.0 is inference-only
- Cannot fine-tune or train custom models
- Must rely on external frameworks for training

**Solution:**
- Add training capability while maintaining inference-first architecture
- Port proven training components from MetaIA v1.4.0
- Adapt to New-QorusIA architecture (clean, specialized)

---

## ARCHITECTURAL DECISION: HYBRID ENGINE

### Design Philosophy

**Primary Focus:** Inference (optimized for any architecture)
- Zero-malloc in hot path
- Cache-aligned structures
- Quantization support (Q4_0)
- Maximum performance

**Secondary Capability:** Training (for custom models)
- Full backward pass support
- Optimizers (Adam, AdamW)
- Loss functions (CrossEntropy, MSE)
- Training loop with mini-batches

**Key Insight:** Training and inference can coexist in the same architecture:
- **Inference Mode:** Uses optimized forward pass (current implementation)
- **Training Mode:** Uses forward + backward pass (to be added)
- **Memory Strategy:** Training uses arena for gradients (separate from inference)

---

## COMPONENTS TO PORT FROM METAIA

### Phase 1: Core Training Infrastructure

#### 1.1 Optimizers (CRITICAL)

**From MetaIA:**
- `src/optim/ft_optimizer.c` - SGD base optimizer
- `src/optim/ft_optimizer_adam.c` - Adam/AdamW optimizer
- `src/math/avx/ft_adamw_avx.c` - AVX2-optimized AdamW

**To New-QorusIA:**
- `src/optim/optimizer.c` - Base optimizer interface
- `src/optim/adam.c` - Adam/AdamW optimizer (AVX2 optimized)
- `src/optim/scheduler.c` - Learning rate scheduling (optional)

**Estimated Time:** 8-12 hours

**Adaptations:**
- `t_tensor` → `q_tensor`
- `int` return → `q_error_code`
- `malloc` → `q_arena_alloc` (for optimizer state)
- Always-active validation

#### 1.2 Loss Functions (CRITICAL)

**From MetaIA:**
- `src/math/avx/ft_tensor_loss_avx.c` - MSE Loss
- `src/math/avx/ft_tensor_loss_crossentropy_avx.c` - CrossEntropy Loss

**To New-QorusIA:**
- `src/ops/avx2/loss_mse.c` - MSE Loss (AVX2 optimized)
- `src/ops/avx2/loss_crossentropy.c` - CrossEntropy Loss (AVX2 optimized)

**Estimated Time:** 4-6 hours

**Adaptations:**
- `t_tensor` → `q_tensor`
- `int` return → `q_error_code`
- Always-active validation

#### 1.3 Gradient Clipping (HIGH PRIORITY)

**From MetaIA:**
- `src/math/avx/ft_tensor_clip_avx.c` - Gradient clipping

**To New-QorusIA:**
- `src/ops/avx2/clip.c` - Gradient clipping (AVX2 optimized)

**Estimated Time:** 2-3 hours

### Phase 2: Backward Pass Implementation

#### 2.1 Backward Pass Infrastructure

**From MetaIA:**
- `src/core/ft_model.c` - `model_backward()` function
- Forward cache management for backward pass

**To New-QorusIA:**
- `src/core/model.c` - `q_model_backward()` function (generic)
- Gradient propagation through layers

**Estimated Time:** 6-8 hours

**Key Challenge:** Layers need backward implementations:
- Attention backward (complex - Q/K/V gradients)
- MLP backward (SwiGLU backward)
- RMSNorm backward
- Residual connections backward

#### 2.2 Layer Backward Implementations

**From MetaIA:**
- `src/layers/ft_layer_linear_back.c` - Linear layer backward
- `src/layers/ft_layer_activation_back.c` - Activation backward
- `src/layers/ft_layer_layernorm.c` - LayerNorm backward

**To New-QorusIA:**
- `src/core/model.c` - Generic backward pass
  - Attention backward (GQA-aware)
  - MLP backward (SwiGLU)
  - RMSNorm backward
  - Residual backward
  - Works with any architecture

**Estimated Time:** 12-16 hours

**Complexity:** High - Requires careful implementation of attention gradients

### Phase 3: Training Loop

#### 3.1 Training Loop Infrastructure

**From MetaIA:**
- `src/core/ft_model_train.c` - `model_fit()` function
- Mini-batch processing
- Epoch loop
- Early stopping support

**To New-QorusIA:**
- `src/core/model.c` - `q_model_train()` function (generic)
- Mini-batch processing (arena-based)
- Epoch loop
- Early stopping support
- Works with any architecture

**Estimated Time:** 6-8 hours

**Key Features:**
- Mini-batch shuffling (Fisher-Yates)
- Forward → Loss → Backward → Optimizer Step → Zero Grad
- Gradient clipping integration
- Early stopping (validation-based)

#### 3.2 Training Utilities

**From MetaIA:**
- Learning rate scheduling
- Gradient accumulation (for large batches)
- Mixed precision training (optional, future)

**To New-QorusIA:**
- Learning rate scheduling (optional)
- Gradient accumulation (if needed)
- Training metrics tracking

**Estimated Time:** 4-6 hours

---

## IMPLEMENTATION STRATEGY

### Approach: Incremental Addition

**Principle:** Add training capability without breaking inference-first architecture.

**Strategy:**
1. **Separate Training Code:** Training functions in separate files (`*_train.c`, `*_backward.c`)
2. **Conditional Compilation:** Training code can be disabled via `#ifdef Q_ENABLE_TRAINING`
3. **Memory Separation:** Training uses separate arena allocations (doesn't interfere with inference)
4. **API Separation:** Training API separate from inference API

### File Structure

```
src/
├── ops/avx2/              # Mathematical kernels (inference + training)
│   ├── loss_mse.c        # MSE Loss (training)
│   ├── loss_crossentropy.c  # CrossEntropy Loss (training)
│   └── clip.c            # Gradient clipping (training)
├── optim/                # Optimizers (NEW - training only)
│   ├── optimizer.c       # Base optimizer interface
│   ├── adam.c            # Adam/AdamW optimizer
│   └── scheduler.c       # Learning rate scheduling
├── layers/               # Generic Layers (v3.0)
│   ├── linear.c          # Linear layer
│   ├── activation.c      # Activation layers
│   ├── normalization.c   # Normalization layers
│   ├── mha.c             # Multi-Head Attention
│   └── ffn.c             # Feed-Forward Network
├── models/               # Model Builders (Examples)
│   └── example_models.c  # Example models using generic framework
```

### API Design

**Inference API (Generic):**
```c
q_error_code q_model_forward(...);  // Inference (generic, any architecture)
```

**Training API (Generic):**
```c
q_error_code q_model_backward(...);  // Backward pass (generic)
q_error_code q_model_train(...);     // Training loop (generic)
```

**Unified API:**
```c
q_error_code q_model_forward_train(...);  // Forward with cache for backward
```

---

## DETAILED COMPONENT SPECIFICATIONS

### Component 1: Optimizers

#### STEP 0: CHAIN OF THOUGHT (CoT)

**UNDERSTAND:**
- **Problem:** Port optimizer implementations from MetaIA to New-QorusIA
- **Inputs:**
  - Model weights (to be updated)
  - Gradients (from backward pass)
  - Optimizer state (moments for Adam)
- **Outputs:** Updated weights (in-place modification)
- **Use Cases:** Training custom models for future-implementations

**BREAK DOWN:**
1. **Base Optimizer Interface:** Define `q_optimizer` structure
2. **SGD Optimizer:** Simple gradient descent
3. **Adam Optimizer:** Adaptive moment estimation
4. **AdamW Optimizer:** Adam with decoupled weight decay
5. **State Management:** Optimizer state allocation/deallocation

**REASON:**
1. Define optimizer interface compatible with New-QorusIA
2. Port SGD (simplest, baseline)
3. Port Adam/AdamW (most useful for training)
4. Use arena for optimizer state (zero-malloc guarantee)
5. AVX2 optimization for weight updates

#### STEP 0.5: MATHEMATICAL PROOF

**ADAM ALGORITHM:**
- **Time Complexity:** O(N) where N is number of parameters
- **Space Complexity:** O(N) for moments (m and v)
- **Correctness:** Proven convergence under convexity assumptions
- **Numerical Stability:** Bias correction prevents division by small numbers

**ADAMW ALGORITHM:**
- **Difference:** Decoupled weight decay (separate from moments)
- **Advantage:** Better generalization (Loshchilov & Hutter, 2019)
- **Formula:** `w_t = w_{t-1} - lr * (update + lambda * w_{t-1})`

#### STEP 1: MODEL CONSTRUCTION (MFR Phase 1)

**ENTITIES:**
```c
// Optimizer types
typedef enum {
    Q_OPTIM_SGD = 0,
    Q_OPTIM_ADAM = 1,
    Q_OPTIM_ADAMW = 2
} q_optimizer_type;

// Optimizer structure
typedef struct {
    q_optimizer_type type;
    float learning_rate;
    float beta1;           // Adam: momentum decay
    float beta2;           // Adam: variance decay
    float epsilon;         // Adam: numerical stability
    float weight_decay;    // L2 regularization (AdamW: decoupled)
    float l1_regularization; // L1 regularization (optional)
    
    // Adam state (allocated in arena)
    float* m;              // First moment estimate
    float* v;              // Second moment estimate
    uint32_t step;         // Timestep counter
    
    // Memory context (for arena allocation)
    q_context* ctx;
} q_optimizer;
```

**FUNCTION PROTOTYPES:**
```c
// Create optimizer
q_optimizer* q_optimizer_create(
    q_optimizer_type type,
    float learning_rate,
    q_context* ctx
);

// Adam-specific creation
q_optimizer* q_optimizer_adam_create(
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    float l1_regularization,
    q_context* ctx
);

// Optimizer step (update weights)
q_error_code q_optimizer_step(
    q_optimizer* opt,
    q_tensor* weights,
    const q_tensor* gradients
);

// Zero gradients (reset)
void q_optimizer_zero_grad(q_optimizer* opt);

// Free optimizer
void q_optimizer_free(q_optimizer* opt);
```

### Component 2: Loss Functions

#### STEP 0: CHAIN OF THOUGHT (CoT)

**UNDERSTAND:**
- **Problem:** Port loss functions from MetaIA to New-QorusIA
- **Loss Types:**
  - MSE (Mean Squared Error) - Regression tasks
  - CrossEntropy - Classification tasks
- **Inputs:**
  - Predictions (model output)
  - Targets (ground truth)
- **Outputs:**
  - Loss value (scalar)
  - Loss gradient (for backward pass)

**BREAK DOWN:**
1. **MSE Loss:** `loss = mean((pred - target)^2)`
2. **MSE Gradient:** `grad = (pred - target) / batch_size`
3. **CrossEntropy Loss:** `loss = -mean(log(softmax(pred)[target]))`
4. **CrossEntropy Gradient:** `grad = softmax(pred) - one_hot(target)`

#### STEP 1: MODEL CONSTRUCTION (MFR Phase 1)

**FUNCTION PROTOTYPES:**
```c
// MSE Loss
q_error_code q_loss_mse_avx2(
    const q_tensor* predictions,
    const q_tensor* targets,
    float* loss_value
);

// MSE Loss Gradient
q_error_code q_loss_mse_grad_avx2(
    const q_tensor* predictions,
    const q_tensor* targets,
    q_tensor* gradients
);

// CrossEntropy Loss
q_error_code q_loss_crossentropy_avx2(
    const q_tensor* predictions,
    const q_tensor* targets,
    float* loss_value
);

// CrossEntropy Loss Gradient
q_error_code q_loss_crossentropy_grad_avx2(
    const q_tensor* predictions,
    const q_tensor* targets,
    q_tensor* gradients
);
```

### Component 3: Backward Pass

#### STEP 0: CHAIN OF THOUGHT (CoT)

**UNDERSTAND:**
- **Problem:** Implement backward pass for any architecture
- **Complexity:** High - Attention backward is complex
- **Challenge:** Generic framework must work with any layer type

**BREAK DOWN:**
1. **Attention Backward:**
   - Q/K/V gradients
   - Output projection gradient
   - Attention scores gradient
2. **MLP Backward:**
   - SwiGLU backward (gate * up)
   - Down projection gradient
3. **RMSNorm Backward:**
   - Weight gradient
   - Input gradient
4. **Residual Backward:**
   - Simple pass-through (gradient splits)

#### STEP 1: MODEL CONSTRUCTION (MFR Phase 1)

**FUNCTION PROTOTYPE:**
```c
// Backward pass through generic model
q_error_code q_model_backward(
    q_model* model,
    q_context* ctx,
    const q_tensor* loss_gradient,  // Gradient from loss function
    uint32_t layer_idx              // Which layer to start from (for partial backward)
);
```

**Key Challenge:** Forward cache must be maintained for backward pass.

---

## IMPLEMENTATION ORDER & TIMELINE

### Phase 1: Foundation (Week 1-2)
1. **Optimizers** (8-12h)
   - Base optimizer interface
   - SGD implementation
   - Adam/AdamW implementation
   - AVX2 optimization

2. **Loss Functions** (4-6h)
   - MSE Loss
   - CrossEntropy Loss
   - AVX2 optimization

3. **Gradient Clipping** (2-3h)
   - AVX2-optimized clipping

**Total Phase 1:** 14-21 hours

### Phase 2: Backward Pass (Week 2-3)
4. **Backward Infrastructure** (6-8h)
   - `q_model_backward()` function (generic)
   - Forward cache management
   - Gradient propagation framework

5. **Layer Backward Implementations** (12-16h)
   - Attention backward (GQA-aware)
   - MLP backward (SwiGLU)
   - RMSNorm backward
   - Residual backward

**Total Phase 2:** 18-24 hours

### Phase 3: Training Loop (Week 3-4)
6. **Training Loop** (6-8h)
   - `q_model_train()` function (generic)
   - Mini-batch processing
   - Epoch loop
   - Early stopping

7. **Training Utilities** (4-6h)
   - Learning rate scheduling
   - Training metrics
   - Checkpoint saving

**Total Phase 3:** 10-14 hours

**Total Estimated Time:** 42-59 hours

---

## ARCHITECTURAL ADAPTATIONS

### MetaIA → New-QorusIA Mapping (Training Components)

| Component | MetaIA | New-QorusIA | Adaptation |
|-----------|--------|-------------|------------|
| **Optimizer State** | `malloc` | `q_arena_alloc` | Zero-malloc guarantee |
| **Gradients** | `tensor_create` | `q_arena_alloc` | Arena-based allocation |
| **Forward Cache** | `malloc` | Arena (training mode) | Separate from inference |
| **Error Handling** | `int` | `q_error_code` | Standardized |
| **Validation** | `#ifdef DEBUG` | Always active | Robust validation |

### Memory Strategy for Training

**Training Mode Memory:**
- **Tier 1:** Model weights (mmap, read-write in training)
- **Tier 2:** KV Cache (not used in training)
- **Tier 3:** Arena (used for):
  - Optimizer state (m, v moments)
  - Gradients (intermediate and final)
  - Forward cache (for backward pass)
  - Temporary buffers

**Key Insight:** Training uses more memory than inference, but still uses arena for zero-malloc guarantee.

---

## VALIDATION STRATEGY

### Training Validation

For each training component, verify:
- [ ] **Correctness:** Matches MetaIA behavior
- [ ] **Numerical Precision:** Max diff < 1e-5 vs PyTorch
- [ ] **Memory Safety:** No leaks (AddressSanitizer)
- [ ] **Performance:** AVX2 optimization effective
- [ ] **Convergence:** Training loss decreases over epochs

### End-to-End Training Validation

- [ ] **Small Model:** Train small Transformer variant (1-2 layers)
- [ ] **Convergence:** Loss decreases over epochs
- [ ] **Gradient Check:** Verify gradients are correct (finite differences)
- [ ] **Checkpoint:** Save/load model checkpoints

---

## FUTURE-IMPLEMENTATIONS INTEGRATION

### Code Agent Training

**Use Case:** Fine-tune Transformer model on code datasets

**Requirements:**
- Custom dataset loader
- Code-specific tokenizer
- Fine-tuning loop
- Model checkpointing

**Integration:**
```c
// Fine-tune Transformer model for code generation
q_error_code code_agent_finetune(
    q_model* model,
    q_context* ctx,
    const char* code_dataset_path,
    uint32_t epochs
);
```

### Customer Behavior Prediction Training

**Use Case:** Train model on customer data

**Requirements:**
- Tabular data loader
- Feature engineering
- Training loop with validation
- Model evaluation metrics

### SEO AI Specialist Training

**Use Case:** Fine-tune on SEO-specific data

**Requirements:**
- SEO dataset loader
- Domain-specific fine-tuning
- Quality scoring integration
- Performance tracking

---

## COMPARISON: METAIA vs QORUSIA (FINAL STATE)

### MetaIA v1.4.0 (Current)
- ✅ Framework genérico completo
- ✅ Training completo
- ✅ Inferência básica
- ⚠️ Arquitetura orgânica
- ⚠️ Não otimizado para inferência LLM

### New-QorusIA v3.0 (Final - With Training)
- ✅ Framework genérico (qualquer arquitetura)
- ✅ Inferência otimizada (primary)
- ✅ Treinamento completo (secondary)
- ✅ Arquitetura limpa desde o início
- ✅ Zero-malloc garantido
- ✅ Validações robustas
- ✅ Produção-ready (inferência)
- ✅ Custom training (future-implementations)

**Result:** Best of both worlds - optimized inference + full training capability.

---

## NEXT STEPS

1. **Review this plan** - Ensure training capability aligns with goals
2. **Execute Phase 1** - Implement optimizers and loss functions
3. **Execute Phase 2** - Implement backward pass
4. **Execute Phase 3** - Implement training loop
5. **Integration** - Integrate with future-implementations

---

**Status:** 📋 READY FOR IMPLEMENTATION  
**Last Updated:** 2024-12-30  
**Framework:** MFR + CoT + Mathematical Proof + TDD (per `.cursorrules`)

