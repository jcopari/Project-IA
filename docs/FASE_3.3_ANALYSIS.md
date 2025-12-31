# FASE 3.3: Forward Pass Implementation - MFR + CoT + Proof + TDD Analysis

## STEP 0: CHAIN OF THOUGHT (CoT) - Problem Analysis

### UNDERSTAND: What is the exact problem?

**Problem:** Implement `llama_forward()` function that executes inference through a Llama-3 model.

**Inputs:**
- `llama_model* model` - Complete model structure with weights and configuration
- `q_context* ctx` - Memory context (Tier 1: mmap weights, Tier 2: KV cache, Tier 3: scratchpad arena)
- `const uint32_t* tokens` - Input token IDs [seq_len]
- `uint32_t seq_len` - Current sequence length (number of tokens)
- `uint32_t pos` - Current position in sequence (for RoPE positional encoding)

**Expected Outputs:**
- `float* logits` - Output logits [vocab_size] representing probability distribution over vocabulary
- Return value: `q_error_code` (Q_OK on success, error code on failure)

**Mathematical Definition:**
```
Forward Pass Flow:
1. Token Embeddings: E = token_embd[tokens]  // [seq_len, dim]
2. For each layer l in [0..n_layers-1]:
   a. Pre-attention RMSNorm: x_norm = RMSNorm(x, attn_norm[l])
   b. Attention:
      - Q = x_norm @ wq[l]  // [seq_len, dim]
      - K = x_norm @ wk[l]  // [seq_len, dim] (GQA: n_kv_heads)
      - V = x_norm @ wv[l]  // [seq_len, dim] (GQA: n_kv_heads)
      - Apply RoPE to Q, K
      - Update KV cache at position pos
      - Attention scores: scores = Q @ K^T / sqrt(head_dim)
      - Apply causal mask (lower triangular)
      - Softmax: probs = softmax(scores)
      - Output: attn_out = probs @ V
      - Project: attn_out = attn_out @ wo[l]
   c. Residual: x = x + attn_out
   d. Pre-MLP RMSNorm: x_norm = RMSNorm(x, ffn_norm[l])
   e. MLP (SwiGLU):
      - gate = SiLU(x_norm @ w_gate[l])
      - up = x_norm @ w_up[l]
      - mlp_out = (gate * up) @ w_down[l]
   f. Residual: x = x + mlp_out
3. Final RMSNorm: x = RMSNorm(x, output_norm)
4. LM Head: logits = x @ output  // [vocab_size]
```

### BREAK DOWN: What are the sub-problems?

**Sub-problems:**
1. **Token Embedding Lookup**
   - Input: token IDs [seq_len]
   - Output: embeddings [seq_len, dim]
   - Operation: Index into `token_embd` weight matrix

2. **Single Layer Forward Pass** (`llama_layer_forward`)
   - Attention block:
     - RMSNorm (pre-attention)
     - Q/K/V projections (with GQA support)
     - RoPE application
     - KV cache management
     - Attention computation (causal masking)
     - Output projection
     - Residual connection
   - MLP block:
     - RMSNorm (pre-MLP)
     - Gate/Up projections (SwiGLU)
     - SiLU activation
     - Down projection
     - Residual connection

3. **KV Cache Management**
   - Layout: `[n_layers, n_kv_heads, max_seq_len, head_dim]`
   - Update: Append new K/V at position `pos`
   - Access: Read cached K/V for positions [0..pos]

4. **Attention with GQA (Grouped Query Attention)**
   - Query heads: `n_heads` (full)
   - Key/Value heads: `n_kv_heads` (grouped, typically `n_kv_heads < n_heads`)
   - Replication: Each KV head serves `n_heads / n_kv_heads` query heads

5. **RoPE (Rotary Positional Embedding)**
   - Apply rotation to Q and K based on position `pos`
   - Uses pre-computed cos/sin tables (or compute on-the-fly)

6. **Final Processing**
   - Final RMSNorm
   - LM Head projection (output logits)

### REASON: What is the logical flow?

**Logical Flow:**
1. **Validation:** Check inputs (null pointers, valid dimensions, arena initialized)
2. **Reset Arena:** Clear scratchpad for new forward pass (`q_arena_reset`)
3. **Token Embeddings:** Lookup embeddings for input tokens
4. **Layer Loop:** For each layer (0..n_layers-1):
   - Allocate temporary buffers from arena (aligned)
   - Execute attention block
   - Execute MLP block
   - Update residual (in-place where possible)
5. **Final Norm + LM Head:** Apply final normalization and projection
6. **Return:** Copy logits to output buffer, return Q_OK

**Memory Management Strategy:**
- Use arena (`q_arena_alloc`) for all temporary buffers
- Reuse buffers across layers where possible
- Ensure 32-byte alignment for AVX2 operations
- Zero-malloc constraint: No `malloc`/`free` in hot path

### EDGE CASES: What must be handled?

**Edge Cases:**
1. **NULL Pointers:** model, ctx, tokens, logits must be non-NULL
2. **Empty Sequence:** seq_len == 0 (should return error or handle gracefully)
3. **Invalid Position:** pos >= max_seq_len (should return error)
4. **Arena OOM:** Arena exhausted (should return Q_ERR_ARENA_OOM)
5. **KV Cache Overflow:** pos >= max_seq_len (should return error)
6. **GQA Edge Cases:** n_kv_heads == 0, n_heads % n_kv_heads != 0
7. **Single Token:** seq_len == 1 (incremental generation)
8. **Full Sequence:** seq_len == max_seq_len (prefill phase)

**Error Conditions:**
- Invalid model configuration (n_layers == 0, dim == 0, etc.)
- KV cache not allocated (`ctx->kv_buffer == NULL`)
- Arena not allocated (`ctx->scratch_buffer == NULL`)
- Misaligned buffers (should be caught by validation macros)

---

## STEP 0.5: MATHEMATICAL PROOF & COMPLEXITY ANALYSIS

### TIME COMPLEXITY

**Overall Complexity:** O(n_layers × (seq_len × dim + seq_len² × n_heads))

**Breakdown:**
1. **Token Embedding:** O(seq_len × dim)
   - Lookup: O(seq_len) indices
   - Copy: O(seq_len × dim) bytes

2. **Per Layer:**
   - **RMSNorm:** O(seq_len × dim) - Single pass through sequence
   - **Q/K/V Projections:** O(seq_len × dim²) - MatMul operations
   - **RoPE:** O(seq_len × dim) - Element-wise rotation
   - **Attention Scores:** O(seq_len² × n_heads) - Q @ K^T for all heads
   - **Softmax:** O(seq_len² × n_heads) - Normalization per head
   - **Attention Output:** O(seq_len² × n_heads × head_dim) = O(seq_len² × dim)
   - **Output Projection:** O(seq_len × dim²)
   - **MLP:** O(seq_len × dim × hidden_dim) - Gate/Up/Down projections

**Total per Layer:** O(seq_len × dim² + seq_len² × dim)
**Total for n_layers:** O(n_layers × (seq_len × dim² + seq_len² × dim))

**Optimization Note:** For incremental generation (seq_len == 1), complexity reduces to O(n_layers × dim²), which is optimal.

### SPACE COMPLEXITY

**Auxiliary Space:** O(seq_len × dim + seq_len × n_heads × head_dim)

**Breakdown:**
1. **Token Embeddings:** O(seq_len × dim) - Stored in arena
2. **Per-Layer Temporaries:**
   - Attention input/output: O(seq_len × dim)
   - Q/K/V: O(seq_len × dim) each (can reuse buffers)
   - Attention scores: O(seq_len² × n_heads) - Only for current layer
   - MLP intermediate: O(seq_len × hidden_dim)
3. **KV Cache (Persistent):** O(n_layers × n_kv_heads × max_seq_len × head_dim) - Pre-allocated

**Peak Memory:** O(seq_len × dim + seq_len² × n_heads) per layer (can be optimized with buffer reuse)

### CACHE COMPLEXITY

**Spatial Locality:**
- Sequential access patterns in MatMul operations maximize cache line utilization
- KV cache layout `[n_layers, n_kv_heads, max_seq_len, head_dim]` ensures sequential access per head
- Tensor views point to contiguous memory (mmap or arena)

**Temporal Locality:**
- Attention scores computed and used immediately (no reuse)
- Q/K/V can be reused if stored in persistent buffers
- Residual connections require storing input (can use arena)

**Cache Line Utilization:**
- AVX2 processes 8 floats per cache line (32 bytes)
- 64-byte alignment ensures optimal cache line usage
- Sequential access patterns minimize cache misses

### PROOF OF CORRECTNESS

**Termination:**
- Outer loop: `for (uint32_t l = 0; l < n_layers; l++)` - Bounded by `n_layers` (finite, from config)
- Inner loops: All bounded by `seq_len` or `dim` (finite, from config)
- Arena allocation: Bounded by `ctx->scratch_size` (finite, pre-allocated)
- **Conclusion:** All loops guaranteed to terminate

**Bounds:**
- **Array Access:** `tokens[i]` where `0 <= i < seq_len` - Valid by loop condition
- **Embedding Access:** `token_embd[tokens[i]]` where `0 <= tokens[i] < vocab_size` - Must validate
- **KV Cache Access:** `kv_cache[layer][head][pos]` where `0 <= pos < max_seq_len` - Must validate
- **Tensor Access:** All tensor accesses use `q_tensor` views with validated bounds
- **Conclusion:** All array accesses are bounded and safe (with proper validation)

**Arithmetic:**
- **Index Calculations:** `i * stride + j` - Must check for overflow (use `size_t` and validate)
- **KV Cache Offset:** `layer * layer_stride + head * head_stride + pos * pos_stride` - Must validate bounds
- **Arena Allocation:** `ctx->scratch_head + size` - Already validated in `q_arena_alloc`
- **Conclusion:** All arithmetic operations are safe with proper validation

**Alignment:**
- **AVX2 Requirements:** All buffers must be 32-byte aligned
- **Arena Guarantee:** `q_arena_alloc` returns 64-byte aligned pointers (proven in memory.c)
- **Tensor Data:** All tensor data is 64-byte aligned (proven in llama_build_graph)
- **Conclusion:** All SIMD operations access aligned memory

### EDGE CASE PROOF

**N=0 (Empty Sequence):**
- `seq_len == 0`: Loop `for (uint32_t i = 0; i < 0; i++)` fails immediately
- Should return error `Q_ERR_INVALID_ARG` (empty sequence is invalid for inference)

**N=1 (Single Token - Incremental Generation):**
- `seq_len == 1`: All operations process single token
- Attention: `Q[1, dim] @ K[pos, dim]^T` - Valid (1 × pos attention matrix)
- Complexity reduces to O(n_layers × dim²) - Optimal for incremental generation

**N=MAX (Full Sequence - Prefill):**
- `seq_len == max_seq_len`: All operations process full sequence
- KV cache: `pos == max_seq_len - 1` - Valid (last position)
- Memory: Peak usage at maximum - Must ensure arena is large enough

**GQA Edge Cases:**
- `n_kv_heads == 0`: Invalid (division by zero in replication factor) - Should return error
- `n_heads % n_kv_heads != 0`: Invalid (cannot evenly divide) - Should return error
- `n_kv_heads == n_heads`: Standard multi-head attention (no grouping)

**Numerical Stability:**
- **RMSNorm:** Uses epsilon (`rms_norm_eps`) to prevent division by zero
- **Softmax:** Uses max-sub trick for numerical stability (proven in softmax.c)
- **Attention Scores:** Division by `sqrt(head_dim)` prevents large values
- **Residual Connections:** In-place addition maintains precision

---

## STEP 1: MODEL CONSTRUCTION (MFR Phase 1)

### ENTITIES (Data Structures)

**No new structures needed** - Uses existing:
- `llama_model` - Complete model structure
- `llama_layer` - Layer structure with weight pointers
- `q_context` - Memory context
- `q_tensor` - Tensor views

**KV Cache Access Helper:**
```c
// Helper function to get KV cache pointer for a specific layer/head/position
// Layout: [n_layers, n_kv_heads, max_seq_len, head_dim]
static float* get_kv_cache_ptr(
    q_context* restrict ctx,
    uint32_t layer_idx,
    uint32_t kv_head_idx,
    uint32_t pos,
    uint32_t head_dim,
    bool is_key  // true for K, false for V
);
```

### MEMORY LAYOUT

**KV Cache Layout:**
```
kv_buffer layout: [n_layers, n_kv_heads, max_seq_len, head_dim]
Offset calculation:
  layer_offset = layer_idx * (n_kv_heads * max_seq_len * head_dim * sizeof(float))
  head_offset = kv_head_idx * (max_seq_len * head_dim * sizeof(float))
  pos_offset = pos * (head_dim * sizeof(float))
  element_offset = element_idx * sizeof(float)
  
Total offset = layer_offset + head_offset + pos_offset + element_offset
```

**Arena Usage (Temporary Buffers):**
- Token embeddings: `[seq_len, dim]` - O(seq_len × dim × sizeof(float))
- Attention Q/K/V: `[seq_len, dim]` each - Can reuse same buffer
- Attention scores: `[seq_len, seq_len]` per head - O(seq_len² × n_heads × sizeof(float))
- MLP intermediate: `[seq_len, hidden_dim]` - O(seq_len × hidden_dim × sizeof(float))

### CONSTRAINTS (Invariants)

**Hardware Constraints:**
- All buffers must be 32-byte aligned for AVX2 operations
- KV cache must be pre-allocated before forward pass
- Arena must be allocated before forward pass

**Validation Constraints:**
- Input tokens must be valid (`0 <= tokens[i] < vocab_size`)
- Position must be valid (`0 <= pos < max_seq_len`)
- Sequence length must be valid (`0 < seq_len <= max_seq_len`)
- Model configuration must be valid (n_layers > 0, dim > 0, etc.)

**Memory Constraints:**
- Zero-malloc in hot path (use arena only)
- KV cache is persistent (not reset between forward passes)
- Arena is reset at start of forward pass (`q_arena_reset`)

**Concurrency Constraints:**
- Function must be thread-safe (no global mutable state)
- KV cache updates are position-specific (no race conditions if positions differ)

### FUNCTION PROTOTYPES

```c
// Main forward pass function
q_error_code llama_forward(
    llama_model* restrict model,
    q_context* restrict ctx,
    const uint32_t* restrict tokens,  // Input tokens [seq_len]
    uint32_t seq_len,                  // Current sequence length
    uint32_t pos,                      // Position in sequence (for RoPE)
    float* restrict logits             // Output logits [vocab_size], 32-byte aligned
);

// Helper: Single layer forward pass
static q_error_code llama_layer_forward(
    llama_layer* restrict layer,
    q_context* restrict ctx,
    const llama_config* restrict config,
    const float* restrict x,           // Input [seq_len, dim]
    float* restrict output,             // Output [seq_len, dim]
    uint32_t layer_idx,
    uint32_t seq_len,
    uint32_t pos,                      // Position for RoPE
    float* restrict kv_cache           // KV cache base pointer
);

// Helper: Attention computation with GQA
static q_error_code llama_attention_forward(
    llama_layer* restrict layer,
    q_context* restrict ctx,
    const llama_config* restrict config,
    const float* restrict x,           // Input [seq_len, dim]
    float* restrict output,             // Output [seq_len, dim]
    uint32_t layer_idx,
    uint32_t seq_len,
    uint32_t pos,
    float* restrict kv_cache
);

// Helper: MLP forward pass (SwiGLU)
static q_error_code llama_mlp_forward(
    llama_layer* restrict layer,
    q_context* restrict ctx,
    const llama_config* restrict config,
    const float* restrict x,           // Input [seq_len, dim]
    float* restrict output,             // Output [seq_len, dim]
    uint32_t seq_len
);

// Helper: Get KV cache pointer
static float* get_kv_cache_ptr(
    q_context* restrict ctx,
    const llama_config* restrict config,
    uint32_t layer_idx,
    uint32_t kv_head_idx,
    uint32_t pos,
    bool is_key
);
```

---

## NEXT STEPS

1. **STEP 2: TDD** - Write tests before implementation
2. **STEP 3: Implementation** - Implement following the model
3. **STEP 4: Validation** - Run tests and verify
4. **STEP 5: Mandatory Test Execution** - Ensure all tests pass

