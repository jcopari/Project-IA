# 🏛️ QORUS-IA v3.0: MASTER BLUEPRINT
# Generic Deep Learning Framework

**Entendido. Você quer o Blueprint de Execução. Sem teoria desnecessária, apenas a engenharia pura para o Cursor executar.**

Aqui está a Arquitetura Definitiva do Qorus-IA v3.0. Copie este contexto para o Cursor e ele saberá exatamente o que fazer.

## Objetivo
Framework Genérico de Deep Learning em C Puro - Sem Limitações Arquiteturais.

**Evolução:**
- **v2.0:** Engine especializado (inferência otimizada)
- **v3.0:** Framework genérico (qualquer arquitetura) mantendo performance e arquitetura limpa

**Prioridades:** Performance (zero-malloc, AVX2), Flexibilidade (qualquer arquitetura), Arquitetura Limpa (validações robustas).

**Restrição:** Zero-Malloc no Hot Path (mantido).

---

## 1. ESTRUTURA DE DIRETÓRIOS (File System)

Esta organização separa infraestrutura, matemática e lógica de modelo.

```
qorus-ia/
├── include/
│   ├── qorus.h             # Header Único Público (API)
│   └── qorus_types.h       # Structs e Enums fundamentais
├── src/
│   ├── core/               # Infraestrutura de Baixo Nível
│   │   ├── memory.c        # Arena, Aligned Malloc, Mmap
│   │   ├── tensor.c        # Manipulação de Metadados de Tensor
│   │   └── utils.c         # Timing, Logging, SIMD detection
│   ├── ops/                # Kernels Matemáticos (Otimizados)
│   │   ├── cpu/            # Fallbacks em C puro (Referência)
│   │   └── avx2/           # Kernels AVX2 (MatMul Q4, MatMul FP32, RoPE, RMSNorm, Add, Mul, Causal Mask, Loss, Clip)
│   ├── optim/              # Optimizers (Training) - NEW
│   │   ├── optimizer.c     # Base optimizer interface
│   │   ├── adam.c          # Adam/AdamW optimizer
│   │   └── scheduler.c     # Learning rate scheduling
│   ├── layers/             # Camadas Genéricas (Framework v3.0)
│   │   ├── linear.c        # Linear layer
│   │   ├── activation.c    # Activation layers
│   │   ├── normalization.c # Normalization layers
│   │   ├── mha.c           # Multi-Head Attention
│   │   ├── ffn.c           # Feed-Forward Network
│   │   └── transformer_block.c  # Transformer Block
│   ├── models/             # Model Builders (Exemplos)
│   │   └── example_models.c  # Exemplos de modelos usando framework genérico
│   └── tokenizer/          # Processamento de Texto
│       └── bpe.c           # Tokenizer BPE minimalista
├── tools/
│   └── convert_model.py    # Script Python: Model Format -> Qorus Binary (Zero-Parse)
├── tests/                  # Testes Unitários e de Integração
└── Makefile                # Build System (-O3 -mavx2)
```

---

## 2. ARQUITETURA DE DADOS (Memory Layout)

O Cursor deve implementar estas estruturas exatamente como definidas para garantir alinhamento e performance.

### 2.1. Tipos e Tensores (include/qorus_types.h)

**Nota de Segurança:** Todas as funções matemáticas agora retornam `q_error_code` e validam inputs em Release mode usando macros otimizadas (`Q_VALIDATE_OR_RETURN`, `Q_VALIDATE_PTR_OR_RETURN`, etc.).

```c
#include <stdint.h>
#include <stdbool.h>

// Alinhamento obrigatório para AVX2/AVX-512
#define Q_ALIGN 64

typedef enum {
    Q_F32  = 0,
    Q_Q8_0 = 1, // Pesos (Embeddings/Output)
    Q_Q4_0 = 2  // Pesos (Dense Layers)
} q_dtype;

// Tensor View (Não possui a memória, apenas aponta)
typedef struct {
    void*     data;         // Ponteiro para dados (Mmap ou Arena)
    float*    scales;       // Ponteiro para escalas (se quantizado)
    uint32_t  ne[4];        // Dimensões: [Batch, Head, Seq, Dim]
    size_t    nb[4];        // Strides em bytes
    q_dtype   type;         // Tipo de dado
    char      name[32];     // Debugging
} __attribute__((aligned(Q_ALIGN))) q_tensor;

// Contexto Global de Memória
typedef struct {
    void* weights_mmap;     // Ponteiro base do arquivo mapeado
    size_t weights_size;
    
    void* kv_buffer;        // Buffer persistente para KV Cache
    size_t kv_size;
    
    void* scratch_buffer;   // Buffer temporário (Arena)
    size_t scratch_size;
    size_t scratch_head;    // Posição atual na Arena
} q_context;
```

---

## 3. ESTRATÉGIA DE MEMÓRIA (The 3 Arenas)

O Cursor deve seguir estritamente esta lógica de alocação.

### Weights (Read-Only):
- **Origem:** mmap de arquivo binário pré-formatado.
- **Acesso:** Ponteiros `q_tensor.data` apontam diretamente para endereços virtuais do mmap.
- **Custo:** Zero copy.

### KV Cache (Persistent):
- **Alocação:** `aligned_alloc` único na inicialização.
- **Layout:** Contíguo `[n_layers, n_kv_heads, max_seq, head_dim]`.
- **Acesso:** Aritmética de ponteiros simples. Sem indireção.

### Scratchpad (Transient):
- **Alocação:** `aligned_alloc` único na inicialização (ex: 512MB).
- **Uso:** Ativações intermediárias (saída de MatMul, Softmax).
- **Ciclo:** `scratch_head` é resetado para 0 no início de cada token gerado.
- **Regra:** NUNCA dar `free()` em tensores individuais aqui.

---

## 4. ROTEIRO DE IMPLEMENTAÇÃO (Step-by-Step)

Peça ao Cursor para executar uma fase por vez. Não avance sem validar.

### ✅ FASE 1: Infraestrutura & Conversor (A Base) - **COMPLETA**

**Objetivo:** Conseguir carregar pesos do disco sem parsing.

- ✅ **Passo 1.1 (Python):** `tools/convert_llama.py` criado. Gera arquivo `.qorus` com header fixo e tensores alinhados a 64 bytes.

- ✅ **Passo 1.2 (C):** `src/core/memory.c` implementado (mmap, arena). `src/core/tensor.c` implementado (criação de views).

- ✅ **Validação:** Testes de memória validados. Carregamento de modelo dummy funcionando.

### ✅ FASE 2: Kernels Matemáticos (O Motor) - **COMPLETA**

**Objetivo:** Operações vetoriais rápidas.

- ✅ **Passo 2.1:** `src/ops/avx2/dequantize.c` implementado. Q4_0 → 32 floats em YMM, FMA-optimized.

- ✅ **Passo 2.2:** `src/ops/avx2/matmul.c` implementado. GEMV Q4_F32 com dequantização fundida, 4x unrolling.

- ✅ **Passo 2.3:** `src/ops/avx2/rope.c` e `src/ops/avx2/rmsnorm.c` implementados.

- ✅ **Passo 2.4:** `src/ops/avx2/silu.c` e `src/ops/avx2/softmax.c` implementados. Utilitários matemáticos em `avx_math.h`.

- ✅ **Passo 2.5:** **Segurança Implementada** - Todas as funções matemáticas agora retornam `q_error_code` e validam inputs em Release mode:
  - Validação de ponteiros nulos
  - Validação de aliasing (input == output)
  - Validação de overflow
  - Validação de alinhamento
  - Validação de tipo de dados
  - Validação de dimensões (múltiplos de 8/32)
  - Macros de validação otimizadas com `__builtin_expect` para overhead mínimo

- ✅ **Validação:** Todos os kernels testados e validados contra referências NumPy. Testes atualizados para verificar retornos de erro.

### ✅ FASE 2.5: Kernels Adicionais (MetaIA Portation) - **COMPLETA**

**Objetivo:** Portar kernels críticos do MetaIA v1.4.0 para completar o forward pass.

**Status:** ✅ **COMPLETA** (2025-12-31). Todos os kernels implementados, testados e validados.

**Kernels Implementados:**

- ✅ **MatMul FP32 AVX2** (`q_matmul_f32_avx2`)
  - **Arquivo:** `src/ops/avx2/matmul_fp32.c`
  - **Testes:** `tests/test_matmul_f32.c`
  - **Uso:** Q @ K^T (attention scores), probs @ V (attention output), LM Head projection
  - **Complexidade:** O(M × N × K)
  - **Status:** Implementado, testado e validado
  - **Características:**
    - Cache-blocked matrix multiplication
    - 4x accumulator unrolling
    - Manual prefetching
    - Transpose B for cache efficiency

- ✅ **Causal Masking AVX2** (`q_causal_mask_f32_avx2`)
  - **Arquivo:** `src/ops/avx2/causal_mask_fp32.c`
  - **Testes:** `tests/test_causal_mask_f32.c`
  - **Uso:** Attention triangular mask (prevent future tokens from attending to past)
  - **Complexidade:** O(N²)
  - **Status:** Implementado, testado e validado
  - **Características:**
    - Vectorized upper triangular masking
    - AVX2 stores for efficiency
    - In-place operation

- ✅ **Tensor Add AVX2** (`q_add_f32_avx2`)
  - **Arquivo:** `src/ops/avx2/add_fp32.c`
  - **Testes:** `tests/test_add_f32.c`
  - **Uso:** Residual connections (`x = x + attn_out`)
  - **Complexidade:** O(N)
  - **Status:** Implementado, testado, validado e code-reviewed
  - **Características:**
    - 4x unrolling (32 elements per iteration)
    - AVX2 vectorized addition
    - In-place operation support (output may alias input)

- ✅ **Element-wise Mul AVX2** (`q_mul_f32_avx2`)
  - **Arquivo:** `src/ops/avx2/mul_fp32.c`
  - **Testes:** `tests/test_mul_f32.c`
  - **Uso:** SwiGLU activation (`gate * up` in MLP)
  - **Complexidade:** O(N)
  - **Status:** Implementado, testado, validado e code-reviewed
  - **Características:**
    - 4x unrolling (32 elements per iteration)
    - AVX2 vectorized multiplication
    - In-place operation support

**Validação Completa:**
- ✅ Todos os testes passam (Release + Debug with sanitizers)
- ✅ Code review completado (First Principles Thinking + CoT)
- ✅ Edge cases tratados (NULL inputs, shape mismatches, alignment)
- ✅ Operações in-place suportadas (safe aliasing)
- ✅ Validação de precisão (max diff < 1e-5 para FP32)
- ✅ Validação de memória (AddressSanitizer clean)

**Adaptações Arquiteturais Aplicadas (MetaIA → New-QorusIA):**
- ✅ `t_tensor` → `q_tensor` (field mapping)
- ✅ `int` return → `q_error_code` enum
- ✅ `malloc` → `q_arena_alloc` (zero-malloc guarantee)
- ✅ `#ifdef DEBUG` → Always-active validation
- ✅ `tensor_*` → `q_*` naming

**Documentação:**
- `docs/KERNEL_PORTATION_PLAN.md` - Plano completo seguindo MFR + CoT + Mathematical Proof + TDD (Status: ✅ COMPLETA)
- `docs/KERNEL_IMPLEMENTATION_DETAILS.md` - Guia de implementação com código completo
- `docs/PLANNING_SUMMARY.md` - Resumo executivo do planejamento

### ✅ Dívida Técnica de Baixa Prioridade - **COMPLETA**

**Objetivo:** Estabelecer base sólida de testes, benchmarking e documentação antes de avançar para forward pass.

- ✅ **Testes de Utilitários:**
  - `test_utils.c` - 23 testes para `q_strerror()` (validação O(1), todos os códigos de erro)
  - `test_avx_math.c` - 13 testes para utilitários AVX (`exp_approx_avx`, `horizontal_sum_avx`, `horizontal_max_avx`)
  - Tolerâncias ajustadas com justificativa matemática para aproximação polinomial

- ✅ **Ferramenta de Benchmark:**
  - `tools/benchmark.c` - Benchmarks end-to-end para todos os kernels AVX2
  - Mede latência (ms), throughput (ops/s), e GFLOPS (para MatMul)
  - Inclui warmup iterations para medições precisas

- ✅ **Documentação Técnica:**
  - `docs/ASYMPTOTIC_ANALYSIS.md` - Análise assintótica completa de todas as funções críticas
  - `docs/ASSEMBLY_ANALYSIS.md` - Guia para análise de código assembly gerado
  - `tools/analyze_assembly.sh` - Script automatizado para análise de assembly
  - `docs/PRECISION_STANDARDS.md` - Atualizado com justificativas técnicas das tolerâncias

### ✅ FASE 3: Model Graph Building (O Corpo) - **PARCIALMENTE COMPLETA**

**Objetivo:** Conectar os kernels na ordem correta usando framework genérico.

- ✅ **Passo 3.1:** Definir estruturas genéricas em `qorus_types.h`.  
  **Status:** Estruturas definidas e validadas com `_Static_assert`.

- ✅ **Passo 3.2:** Implementar `q_model_build_graph()`. Configurar ponteiros dos tensores baseados no arquivo mmap.
  **Status:** Implementado e testado (31 testes, 100% pass rate).
  - Zero-copy tensor views
  - Validação completa de configuração
  - Suporte a Q4_0 e FP32
  - Testes adversarial completos

- ⏳ **Passo 3.3:** Implementar `llama_forward()`. Orquestrar passagem dos dados pelos kernels usando framework genérico.
  **Status:** Em progresso (estrutura completa, atenção e LM Head precisam de conclusão).
  **Dependências:** ✅ Todas resolvidas (FASE 2.5 completa)
    - ✅ MatMul FP32 AVX2 (Q @ K^T, probs @ V, projection layers)
    - ✅ Causal Masking AVX2 (attention mask)
    - ✅ Tensor Add AVX2 (residual connections)
    - ✅ Element-wise Mul AVX2 (SwiGLU activation)
  **Progresso:**
    - ✅ Estrutura do forward pass completa
    - ✅ KV cache helper implementado
    - ✅ MLP forward pass completo (SwiGLU)
    - ✅ Layer forward pass completo (attention + MLP com residuals)
    - ✅ Final RMSNorm implementado
    - ⏳ Attention forward pass (Q/K/V projections feito, RoPE/KV cache/causal mask/softmax TODO)
    - ⏳ LM Head projection (precisa transpose ou GEMV)

**Nota:** Framework genérico permite qualquer arquitetura, não apenas Transformers.

### ⏳ FASE 2.6: Training Kernels (Planejamento Completo) - **PLANEJAMENTO COMPLETO**

**Objetivo:** Adicionar capacidade de treinamento para future-implementations (Code Agent, Customer Behavior Prediction, SEO AI Specialist).

**Status:** 📋 Planejamento completo (2024-12-30). Pronto para implementação.

**Componentes Planejados:**

- ⏳ **Optimizers** (`src/optim/`)
  - **Arquivo:** `src/optim/optimizer.c`, `src/optim/adam.c`
  - **Uso:** Atualização de pesos durante treinamento (Adam, AdamW)
  - **Tempo Estimado:** 8-12 horas
  - **Características:**
    - AVX2-optimized weight updates
    - Arena-based state allocation (zero-malloc)
    - Support for SGD, Adam, AdamW

- ⏳ **Loss Functions** (`src/ops/avx2/`)
  - **Arquivos:** `src/ops/avx2/loss_mse.c`, `src/ops/avx2/loss_crossentropy.c`
  - **Uso:** Cálculo de loss e gradientes para backward pass
  - **Tempo Estimado:** 4-6 horas
  - **Características:**
    - AVX2-optimized loss computation
    - Gradient computation for backward pass

- ⏳ **Gradient Clipping** (`src/ops/avx2/`)
  - **Arquivo:** `src/ops/avx2/clip.c`
  - **Uso:** Estabilização de gradientes durante treinamento
  - **Tempo Estimado:** 2-3 horas
  - **Características:**
    - AVX2-optimized clipping
    - In-place operation

**Total Estimado (FASE 2.6):** 14-21 horas

**Documentação de Planejamento:**
- `docs/TRAINING_CAPABILITY_PLAN.md` - Plano completo de capacidade de treinamento

### ⏳ FASE 3.4: Backward Pass (Training) - **PLANEJAMENTO COMPLETO**

**Objetivo:** Implementar backward pass para propagação de gradientes.

**Status:** 📋 Planejamento completo (2024-12-30). Bloqueado por FASE 2.6.

**Componentes Planejados:**

- ⏳ **Backward Infrastructure** (`src/core/model.c`)
  - **Função:** `q_model_backward()`
  - **Uso:** Propagação de gradientes através das camadas (genérico)
  - **Tempo Estimado:** 6-8 horas
  - **Características:**
    - Forward cache management
    - Gradient propagation framework
    - Funciona com qualquer arquitetura

- ⏳ **Layer Backward Implementations**
  - **Attention Backward:** Q/K/V gradients, GQA-aware
  - **MLP Backward:** SwiGLU backward, down projection gradient
  - **RMSNorm Backward:** Weight gradient, input gradient
  - **Residual Backward:** Gradient pass-through
  - **Tempo Estimado:** 12-16 horas

**Total Estimado (FASE 3.4):** 18-24 horas

**Dependências:** FASE 2.6 (Optimizers, Loss Functions, Gradient Clipping)

### ⏳ FASE 3.5: Training Loop (Training) - **PLANEJAMENTO COMPLETO**

**Objetivo:** Implementar loop de treinamento completo.

**Status:** 📋 Planejamento completo (2024-12-30). Bloqueado por FASE 3.4.

**Componentes Planejados:**

- ⏳ **Training Loop** (`src/core/model.c`)
  - **Função:** `q_model_train()`
  - **Uso:** Loop completo de treinamento (epochs, mini-batches) - genérico
  - **Tempo Estimado:** 6-8 horas
  - **Características:**
    - Mini-batch shuffling (Fisher-Yates)
    - Forward → Loss → Backward → Optimizer Step → Zero Grad
    - Gradient clipping integration
    - Early stopping support
    - Funciona com qualquer arquitetura

- ⏳ **Training Utilities**
  - Learning rate scheduling
  - Training metrics tracking
  - Checkpoint saving
  - **Tempo Estimado:** 4-6 horas

**Total Estimado (FASE 3.5):** 10-14 horas

**Dependências:** FASE 3.4 (Backward Pass)

### ⏳ FASE 4: Tokenizer & Loop (A Vida) - **NÃO INICIADA**

**Objetivo:** Texto entra, texto sai.

- ⏳ **Passo 4.1:** Implementar `src/tokenizer/bpe.c`. Carregar `tokenizer.bin` (extraído do modelo original).

- ⏳ **Passo 4.2:** Criar `main.c`. Loop: Tokenize -> Forward -> Sample -> Print -> Update Cache.
  **Nota:** Todas as chamadas de funções matemáticas devem verificar retorno `q_error_code`.

---

## 🚀 EVOLUÇÃO PARA v3.0: FRAMEWORK GENÉRICO

### Objetivo v3.0

Transformar QorusIA de engine especializado em **framework genérico** sem limitações arquiteturais, mantendo:
- ✅ Performance máxima (zero-malloc, AVX2)
- ✅ Arquitetura limpa (validações robustas)
- ✅ Flexibilidade total (qualquer arquitetura)

### ⏳ FASE 5.0: Core Abstraction (Framework Genérico) - **PLANEJAMENTO COMPLETO**

**Objetivo:** Implementar abstração genérica de camadas e modelos.

**Status:** 📋 Planejamento completo (2024-12-30). Pronto para implementação.

**Componentes Planejados:**

- ⏳ **Generic Layer Interface** (`include/qorus_types.h`)
  - **Estrutura:** `q_layer` com function pointers (polimorfismo)
  - **Tempo Estimado:** 4-6 horas
  - **Características:**
    - 64-byte aligned
    - Function pointers para forward/backward/free
    - Type enum para runtime checking

- ⏳ **Generic Model Container** (`src/core/model.c`)
  - **Estrutura:** `q_model` com array de camadas genéricas
  - **Tempo Estimado:** 6-8 horas
  - **Características:**
    - 128-byte aligned
    - Forward cache para treinamento
    - Suporte a mmap (zero-copy)

- ⏳ **Generic Forward Pass** (`src/core/model.c`)
  - **Função:** `q_model_forward()` genérica
  - **Tempo Estimado:** 4-6 horas
  - **Características:**
    - Polimorfismo via function pointers
    - Forward cache management
    - Validações robustas

- ⏳ **Generic Backward Pass** (`src/core/model.c`)
  - **Função:** `q_model_backward()` genérica
  - **Tempo Estimado:** 6-8 horas
  - **Características:**
    - Propagação de gradientes genérica
    - Uso de forward cache
    - Suporte a camadas não-treináveis

**Total Estimado (FASE 5.0):** 20-28 horas

**Documentação de Planejamento:**
- `docs/GENERIC_FRAMEWORK_PLAN.md` - Plano completo de framework genérico

### ⏳ FASE 5.1: Basic Layers (Framework Genérico) - **PLANEJAMENTO COMPLETO**

**Objetivo:** Implementar camadas básicas com interface genérica.

**Status:** 📋 Planejamento completo (2024-12-30).

**Camadas Planejadas:**

- ⏳ **Linear Layer** (`src/layers/linear.c`)
  - **Interface:** Genérica (`q_layer`)
  - **Tempo Estimado:** 6-8 horas
  - **Características:**
    - Forward/backward genéricos
    - Suporte Q4_0 e FP32
    - Gradientes para treinamento

- ⏳ **Activation Layers** (`src/layers/activation.c`)
  - **Tipos:** ReLU, GeLU, SiLU, Sigmoid
  - **Tempo Estimado:** 4-6 horas
  - **Características:**
    - Forward/backward genéricos
    - AVX2 optimized

- ⏳ **Normalization Layers** (`src/layers/normalization.c`)
  - **Tipos:** RMSNorm, LayerNorm, BatchNorm
  - **Tempo Estimado:** 6-8 horas
  - **Características:**
    - Forward/backward genéricos
    - AVX2 optimized

- ⏳ **Softmax Layer** (`src/layers/softmax.c`)
  - **Interface:** Genérica
  - **Tempo Estimado:** 2-3 horas
  - **Características:**
    - Forward/backward genéricos
    - AVX2 optimized

**Total Estimado (FASE 5.1):** 18-25 horas

### ⏳ FASE 5.2: Advanced Layers (Framework Genérico) - **PLANEJAMENTO COMPLETO**

**Objetivo:** Implementar camadas avançadas com interface genérica.

**Status:** 📋 Planejamento completo (2024-12-30).

**Camadas Planejadas:**

- ⏳ **Multi-Head Attention** (`src/layers/mha.c`)
  - **Interface:** Genérica
  - **Tempo Estimado:** 8-10 horas
  - **Características:**
    - Suporte GQA
    - Forward/backward genéricos
    - AVX2 optimized

- ⏳ **Feed-Forward Network** (`src/layers/ffn.c`)
  - **Interface:** Genérica
  - **Tempo Estimado:** 6-8 horas
  - **Características:**
    - Suporte SwiGLU
    - Forward/backward genéricos
    - AVX2 optimized

- ⏳ **Transformer Block** (`src/layers/transformer_block.c`)
  - **Interface:** Genérica
  - **Tempo Estimado:** 4-6 horas
  - **Características:**
    - Composição de MHA + FFN + RMSNorm
    - Forward/backward genéricos

- ⏳ **Embedding Layer** (`src/layers/embedding.c`)
  - **Interface:** Genérica
  - **Tempo Estimado:** 4-6 horas
  - **Características:**
    - Token embedding
    - Positional embedding (RoPE)
    - Forward/backward genéricos

**Total Estimado (FASE 5.2):** 22-30 horas

### ⏳ FASE 5.3: Example Model Builders (Framework Genérico) - **PLANEJAMENTO COMPLETO**

**Objetivo:** Criar exemplos de modelos usando framework genérico.

**Status:** 📋 Planejamento completo (2024-12-30). Bloqueado por FASE 5.0-5.2.

**Componentes Planejados:**

- ⏳ **Transformer Model Builder** (`src/models/transformer_builder.c`)
  - **Função:** `transformer_build_model()` usando API genérica
  - **Tempo Estimado:** 6-8 horas
  - **Características:**
    - Exemplo de modelo Transformer usando framework genérico
    - Zero-copy weight loading
    - Demonstra flexibilidade do framework

- ⏳ **Example Testing**
  - **Tempo Estimado:** 4-6 horas
  - **Características:**
    - Testes de performance
    - Validação de correção
    - Demonstra uso do framework genérico

- ⏳ **Documentation**
  - **Tempo Estimado:** 2-3 horas
  - **Características:**
    - Exemplos de uso
    - Guias de migração
    - Documentação de API

**Total Estimado (FASE 5.3):** 12-17 horas

**Dependências:** FASE 5.0 (Core Abstraction), FASE 5.1 (Basic Layers), FASE 5.2 (Advanced Layers)

### ⏳ FASE 5.4: Additional Architectures (Framework Genérico) - **FUTURO**

**Objetivo:** Suportar arquiteturas adicionais usando framework genérico.

**Arquiteturas Planejadas:**

- ⏳ **Simple MLP** (Exemplo: MNIST classifier)
  - **Tempo Estimado:** 4-6 horas
  - **Demonstra:** Flexibilidade do framework genérico

- ⏳ **CNN Support** (Futuro)
  - Conv2D layer
  - Pool2D layer
  - Arquiteturas CNN

- ⏳ **RNN/LSTM Support** (Futuro)
  - RNN layer
  - LSTM layer
  - Modelos de sequência

---

## COMPARAÇÃO: v2.0 vs v3.0

### QorusIA v2.0 (Atual)
- ✅ Performance máxima (zero-malloc, AVX2)
- ✅ Arquitetura limpa
- ❌ Limitado a arquitetura específica
- ❌ Estrutura hardcoded

### QorusIA v3.0 (Proposto)
- ✅ Performance máxima (zero-malloc, AVX2)
- ✅ Arquitetura limpa
- ✅ Genérico (qualquer arquitetura)
- ✅ Composição flexível
- ✅ Fácil de estender

**Resultado:** MetaIA's flexibilidade + QorusIA's performance = Framework genérico sem limitações.

---

## 5. REGRAS DE CODIFICAÇÃO (Para o Cursor)

Cole isso no prompt do Cursor para garantir qualidade:

- **Strict C11:** Use C11 padrão. Sem extensões GNU a menos que estritamente necessário para AVX.

- **No Mallocs:** Proibido usar `malloc` ou `free` dentro de `src/ops` ou `src/models`. Use a API da Arena.

- **Restrict Pointers:** Use `float *restrict a` em kernels matemáticos para permitir otimizações agressivas do compilador.

- **Error Handling:** 
  - Funções matemáticas retornam `q_error_code` (enum padronizado).
  - Use macros `Q_VALIDATE_OR_RETURN` para validações críticas (sempre ativas em Release).
  - Em DEBUG mode: validações abortam com mensagem detalhada.
  - Em Release mode: validações retornam código de erro apropriado.
  - Crash (`abort`) apenas em DEBUG mode para facilitar debugging.

- **Comments:** Documente o layout de memória esperado em cima de cada kernel (ex: "Espera que A seja [K, N] transposto").

---

## 6. SEGURANÇA E VALIDAÇÃO

### Validações Críticas (Sempre Ativas)

Todas as funções matemáticas implementam validações críticas que estão **sempre ativas**, mesmo em Release mode:

- ✅ **Validação de Ponteiros Nulos:** Previne segfaults
- ✅ **Validação de Aliasing:** Previne corrupção de dados (input == output)
- ✅ **Validação de Overflow:** Previne wraparound em cálculos de índices
- ✅ **Validação de Alinhamento:** Previne crashes em instruções AVX2
- ✅ **Validação de Tipo:** Previne uso incorreto de dados quantizados
- ✅ **Validação de Dimensões:** Previne acesso fora dos limites

### Macros de Validação

```c
// Exemplo de uso em funções matemáticas
q_error_code q_gemv_q4_f32_avx2(...) {
    Q_VALIDATE_PTR_OR_RETURN(weights, Q_ERR_INVALID_ARG);
    Q_VALIDATE_OR_RETURN(input != output, Q_ERR_ALIASING);
    Q_VALIDATE_MULTIPLE_OR_RETURN(N, 32, Q_ERR_INVALID_SIZE);
    // ... implementação ...
    return Q_OK;
}
```

### Códigos de Erro Padronizados

```c
typedef enum {
    Q_OK = 0,
    Q_ERR_INVALID_ARG = -10,      // Argumento inválido
    Q_ERR_ALIASING = -11,         // Aliasing detectado
    Q_ERR_OVERFLOW = -12,         // Overflow detectado
    Q_ERR_MISALIGNED = -13,       // Ponteiro desalinhado
    Q_ERR_INVALID_DTYPE = -14,    // Tipo de dado inválido
    Q_ERR_INVALID_SIZE = -15      // Tamanho inválido
    // ... outros códigos ...
} q_error_code;
```

### Performance

- **Overhead Mínimo:** Validações usam `__builtin_expect` para otimizar branch prediction
- **Custo Estimado:** < 1 ciclo por validação quando passa (caso comum)
- **Custo Quando Falha:** Retorno imediato de erro (sem processamento desnecessário)

---

## 7. MELHORIAS DE ROBUSTEZ

### Aritmética de Ponteiros Robusta

**Implementação:** Todas as operações de aritmética de ponteiros usam `size_t` para cálculos de offset, garantindo máxima robustez mesmo em casos extremos.

**Exemplo em `q_gemv_q4_f32_avx2`:**
```c
// ROBUSTNESS: Use size_t for offset calculations to prevent uint32_t wraparound
const size_t block_base = (size_t)(bg * 4);
const size_t tail_start = (size_t)(num_block_groups * 4);
const size_t row_offset = (size_t)i * (size_t)blocks_per_row;
```

**Benefícios:**
- ✅ Elimina qualquer possibilidade de wraparound em `uint32_t` antes da conversão para aritmética de ponteiros
- ✅ Consistência de tipos em todo o código
- ✅ Zero overhead: compilador otimiza igualmente
- ✅ Dupla camada de proteção (validação + tipo mais seguro)

### Documentação de Comportamento

**Wrapper Público para Testes:** Funções públicas de teste incluem validação NULL e documentação clara do comportamento esperado.

**Exemplo em `q_dequantize_q4_0_block_avx2_public`:**
```c
// ROBUSTNESS: Validate inputs (only in public wrapper, not in hot path)
// This prevents crashes in test scenarios while maintaining zero overhead
// in production code paths that use the inline version directly
if (__builtin_expect(block == NULL || output == NULL, 0)) {
    return; // Silently return - acceptable for test code defensive programming
}
```

**Filosofia:**
- Hot path usa versões inline sem overhead de validação
- Wrappers públicos para testes incluem validação defensiva
- Comportamento claramente documentado

### Validação de Overflow em Múltiplas Camadas

**Estratégia:** Validações de overflow em múltiplos pontos críticos:

1. **Validação de Dimensões:** `Q_VALIDATE_NO_OVERFLOW_OR_RETURN(M, blocks_per_row)`
2. **Cálculo Seguro de Offset:** Uso de `size_t` para aritmética de ponteiros
3. **Validação de Alinhamento:** `safe_align_size()` previne overflow no alinhamento
4. **Validação de Adição:** `ctx->scratch_head > SIZE_MAX - aligned_size` previne overflow na adição

**Resultado:** Múltiplas camadas de proteção garantem robustez máxima sem impacto na performance.

---

## 7.5. CHECKPOINTS DE REFATORAÇÃO

### Objetivo

**Prevenir acúmulo de dívida técnica** através de refatoração sistemática em checkpoints estratégicos entre fases, garantindo:
- Qualidade de código mantida
- Arquitetura limpa preservada
- Performance mantida
- Dívida técnica minimizada
- Retrabalho reduzido

**Princípio Chave:** Refatorar incrementalmente, não reativamente.

### Quando Refatorar

#### Checkpoints Obrigatórios (Após Cada Fase)
- **Após FASE 2.5:** Refatorar consistência de interface de kernels
- **Após FASE 3.3:** Refatorar arquitetura do forward pass
- **Após FASE 3.5:** Refatorar arquitetura do training loop
- **Após FASE 5.0:** Refatorar design da abstração core
- **Após FASE 5.1:** Refatorar consistência de interface de layers
- **Após FASE 5.2:** Refatorar arquitetura de layers avançadas
- **Após FASE 5.3:** Refatorar estratégia de migração de arquitetura
- **Após FASE 5.4:** Refatoração final antes de produção

#### Checkpoints Opcionais (Durante Desenvolvimento)
- Quando duplicação de código é detectada
- Quando performance degrada inesperadamente
- Quando arquitetura fica confusa
- Quando testes ficam difíceis de manter

### Procedimento de Checkpoint

#### Fase 1: Avaliação (30 minutos)
1. **Revisão de Código:**
   - Revisar todo código adicionado na fase
   - Identificar code smells (duplicação, complexidade, inconsistência)
   - Verificar aderência a padrões de codificação
   - Verificar consistência de tratamento de erros
   - Verificar padrões de gerenciamento de memória

2. **Revisão de Arquitetura:**
   - Verificar separação de responsabilidades
   - Verificar consistência de interfaces
   - Revisar alinhamento de estruturas de dados
   - Verificar convenções de nomenclatura
   - Verificar completude de documentação

3. **Revisão de Performance:**
   - Executar benchmarks de performance
   - Comparar com fase anterior
   - Identificar regressões de performance
   - Verificar padrões de uso de memória
   - Verificar conformidade zero-malloc

4. **Revisão de Testes:**
   - Verificar cobertura de testes
   - Verificar qualidade de testes
   - Revisar organização de testes
   - Verificar manutenibilidade de testes
   - Verificar cobertura de testes adversariais

#### Fase 2: Planejamento de Refatoração (30 minutos)
1. **Identificar Alvos de Refatoração:**
   - Listar code smells a corrigir
   - Identificar melhorias arquiteturais
   - Planejar padronização de interfaces
   - Identificar otimizações de performance
   - Planejar atualizações de documentação

2. **Priorizar Tarefas de Refatoração:**
   - Alta prioridade: Problemas críticos
   - Média prioridade: Melhorias importantes
   - Baixa prioridade: Melhorias desejáveis

3. **Estimar Esforço de Refatoração:**
   - Estimar tempo para cada tarefa
   - Identificar dependências
   - Planejar sequência de refatoração
   - Definir limites de tempo (máx 1-2 dias por checkpoint)

#### Fase 3: Execução de Refatoração (1-2 dias)
1. **Refatoração de Código:**
   - Remover duplicação de código
   - Simplificar funções complexas
   - Padronizar interfaces
   - Melhorar tratamento de erros
   - Otimizar uso de memória

2. **Refatoração de Arquitetura:**
   - Melhorar separação de responsabilidades
   - Padronizar estruturas de dados
   - Melhorar convenções de nomenclatura
   - Melhorar modularidade
   - Melhorar extensibilidade

3. **Refatoração de Performance:**
   - Otimizar hot paths
   - Reduzir alocações de memória
   - Melhorar localidade de cache
   - Otimizar uso de SIMD
   - Reduzir overhead de chamadas de função

4. **Refatoração de Testes:**
   - Melhorar organização de testes
   - Adicionar casos de teste faltantes
   - Melhorar legibilidade de testes
   - Reduzir duplicação de testes
   - Melhorar manutenibilidade de testes

#### Fase 4: Validação (1-2 horas)
1. **Validação de Código:**
   - Executar todos os testes (devem passar)
   - Executar benchmarks de performance (devem manter ou melhorar)
   - Executar sanitizadores de memória (devem passar)
   - Executar ferramentas de análise estática
   - Verificar conformidade zero-malloc

2. **Validação de Documentação:**
   - Atualizar comentários de código
   - Atualizar documentação de arquitetura
   - Atualizar documentação de API
   - Atualizar documentos de status
   - Atualizar timeline se necessário

3. **Validação de Qualidade:**
   - Verificar métricas de qualidade de código
   - Verificar cobertura de testes (deve manter ou melhorar)
   - Verificar métricas de performance
   - Verificar completude de documentação
   - Verificar conclusão do checkpoint

### Checklist de Checkpoint

#### Após Conclusão de Cada Fase

**Qualidade de Código:**
- [ ] Sem duplicação de código
- [ ] Funções são focadas e simples
- [ ] Interfaces são consistentes
- [ ] Tratamento de erros é padronizado
- [ ] Gerenciamento de memória está correto

**Qualidade de Arquitetura:**
- [ ] Separação de responsabilidades está clara
- [ ] Estruturas de dados estão bem projetadas
- [ ] Convenções de nomenclatura são consistentes
- [ ] Modularidade está mantida
- [ ] Extensibilidade está preservada

**Qualidade de Performance:**
- [ ] Performance está mantida ou melhorada
- [ ] Conformidade zero-malloc verificada
- [ ] Localidade de cache otimizada
- [ ] Uso de SIMD está otimizado
- [ ] Sem regressões de performance

**Qualidade de Testes:**
- [ ] Cobertura de testes mantida ou melhorada
- [ ] Testes estão bem organizados
- [ ] Testes são manuteníveis
- [ ] Testes adversariais são abrangentes
- [ ] Todos os testes passam

**Qualidade de Documentação:**
- [ ] Comentários de código atualizados
- [ ] Documentos de arquitetura atualizados
- [ ] Documentos de API atualizados
- [ ] Documentos de status atualizados
- [ ] Timeline atualizada se necessário

### Requisitos Específicos por Checkpoint

#### Checkpoint: Após FASE 2.5 (Kernels de Inferência Adicionais)
**Áreas de Foco:**
- Consistência de interface de kernels
- Padronização de tratamento de erros
- Otimização de performance
- Cobertura de testes

**Tarefas Específicas:**
- [ ] Padronizar assinaturas de funções de kernel
- [ ] Garantir tratamento de erros consistente
- [ ] Verificar padrões de otimização AVX2
- [ ] Adicionar casos de teste faltantes
- [ ] Atualizar documentação de kernels

**Limite de Tempo:** 1 dia

#### Checkpoint: Após FASE 3.3 (Forward Pass)
**Áreas de Foco:**
- Arquitetura do forward pass
- Integração de layers
- Otimização de performance
- Propagação de erros

**Tarefas Específicas:**
- [ ] Revisar estrutura do forward pass
- [ ] Padronizar integração de layers
- [ ] Otimizar performance do forward pass
- [ ] Melhorar tratamento de erros
- [ ] Adicionar testes de forward pass

**Limite de Tempo:** 1-2 dias

#### Checkpoint: Após FASE 3.5 (Training Loop)
**Áreas de Foco:**
- Arquitetura do training loop
- Integração de optimizer
- Integração de loss function
- Fluxo de gradientes

**Tarefas Específicas:**
- [ ] Revisar estrutura do training loop
- [ ] Padronizar interface de optimizer
- [ ] Otimizar performance de treinamento
- [ ] Melhorar fluxo de gradientes
- [ ] Adicionar testes de treinamento

**Limite de Tempo:** 1-2 dias

#### Checkpoint: Após FASE 5.0 (Core Abstraction)
**Áreas de Foco:**
- Interface genérica de layer
- Design de container de modelo
- Implementação de polimorfismo
- Overhead de performance

**Tarefas Específicas:**
- [ ] Revisar design de interface genérica
- [ ] Otimizar overhead de function pointers
- [ ] Padronizar interface de layer
- [ ] Verificar zero overhead de performance
- [ ] Adicionar testes de framework

**Limite de Tempo:** 1-2 dias

#### Checkpoint: Após FASE 5.1 (Basic Layers)
**Áreas de Foco:**
- Consistência de interface de layers
- Qualidade de implementação de layers
- Otimização de performance
- Cobertura de testes

**Tarefas Específicas:**
- [ ] Padronizar implementações de layers
- [ ] Otimizar performance de layers
- [ ] Melhorar tratamento de erros de layers
- [ ] Adicionar testes de layers
- [ ] Atualizar documentação de layers

**Limite de Tempo:** 1 dia

#### Checkpoint: Após FASE 5.2 (Advanced Layers)
**Áreas de Foco:**
- Arquitetura de layers avançadas
- Composição de layers
- Otimização de performance
- Testes de layers complexas

**Tarefas Específicas:**
- [ ] Revisar design de layers avançadas
- [ ] Otimizar composição de layers
- [ ] Melhorar performance de layers complexas
- [ ] Adicionar testes de layers avançadas
- [ ] Atualizar documentação de layers avançadas

**Limite de Tempo:** 1-2 dias

#### Checkpoint: Após FASE 5.3 (Architecture Migration)
**Áreas de Foco:**
- Completude de migração
- Compatibilidade reversa
- Validação de performance
- Limpeza de código

**Tarefas Específicas:**
- [ ] Verificar completude de migração
- [ ] Remover código de arquitetura antiga
- [ ] Validar compatibilidade reversa
- [ ] Verificar performance mantida
- [ ] Limpar código não utilizado

**Limite de Tempo:** 1 dia

#### Checkpoint: Após FASE 5.4 (Produção Final)
**Áreas de Foco:**
- Prontidão para produção
- Revisão final de qualidade de código
- Validação final de performance
- Completude de documentação

**Tarefas Específicas:**
- [ ] Revisão final de código
- [ ] Validação final de performance
- [ ] Revisão final de cobertura de testes
- [ ] Completar documentação
- [ ] Checklist de prontidão para produção

**Limite de Tempo:** 2 dias

### Métricas para Acompanhar

#### Métricas de Qualidade de Código
- **Complexidade Ciclomática:** Deve diminuir ou permanecer estável
- **Duplicação de Código:** Deve diminuir
- **Comprimento de Função:** Deve permanecer razoável (< 100 linhas)
- **Cobertura de Comentários:** Deve manter ou melhorar

#### Métricas de Performance
- **Latência de Inferência:** Deve manter ou melhorar
- **Throughput de Treinamento:** Deve manter ou melhorar
- **Uso de Memória:** Deve manter ou diminuir
- **Conformidade Zero-Malloc:** Deve ser 100%

#### Métricas de Qualidade de Testes
- **Cobertura de Testes:** Deve manter ou melhorar
- **Taxa de Passagem de Testes:** Deve ser 100%
- **Tempo de Execução de Testes:** Deve permanecer razoável
- **Cobertura de Testes Adversariais:** Deve manter ou melhorar

### Critérios de Sucesso

**Checkpoint é Bem-Sucedido Quando:**
- [ ] Todos os testes passam
- [ ] Performance está mantida ou melhorada
- [ ] Métricas de qualidade de código melhoram ou permanecem estáveis
- [ ] Documentação está atualizada
- [ ] Dívida técnica está reduzida
- [ ] Arquitetura está mais limpa
- [ ] Código está mais manutenível

**Documentação Detalhada:** Ver `docs/REFACTORING_CHECKPOINTS.md` para procedimentos completos de checkpoint.

---

## 8. PRÓXIMOS PASSOS

### ✅ Implementação Completa: FASE 2.5 (Inference Kernels)

**Status:** ✅ **COMPLETA** (2025-12-31)

Todos os kernels críticos foram implementados, testados e validados:
1. ✅ MatMul FP32 AVX2
2. ✅ Causal Masking AVX2
3. ✅ Tensor Add AVX2
4. ✅ Element-wise Mul AVX2

**Próximo Passo:** Completar integração no forward pass (FASE 3.3)

### Implementação Futura: FASE 2.6 (Training Kernels)

Para implementar capacidade de treinamento:

> **"Atue como Qorus-Architect. Vamos implementar a FASE 2.6. Comece com os Optimizers seguindo o planejamento completo em `docs/TRAINING_CAPABILITY_PLAN.md`. Use o framework MFR + CoT + Mathematical Proof + TDD conforme `docs/.cursorrules`."**

**Ordem de Implementação Recomendada:**
1. Optimizers (Adam, AdamW) - Base para treinamento
2. Loss Functions (MSE, CrossEntropy) - Necessário para backward
3. Gradient Clipping - Estabilização de treinamento
4. Backward Pass (FASE 3.4) - Propagação de gradientes
5. Training Loop (FASE 3.5) - Loop completo de treinamento

### Implementação Futura: FASE 5.0+ (Generic Framework v3.0)

Para transformar QorusIA em framework genérico sem limitações:

> **"Atue como Qorus-Architect. Vamos implementar a FASE 5.0. Comece com a Generic Layer Interface seguindo o planejamento completo em `docs/GENERIC_FRAMEWORK_PLAN.md`. Use o framework MFR + CoT + Mathematical Proof + TDD conforme `docs/.cursorrules`."**

**Ordem de Implementação Recomendada:**
1. FASE 5.0: Core Abstraction (Generic Layer Interface, Model Container)
2. FASE 5.1: Basic Layers (Linear, Activation, Normalization, Softmax)
3. FASE 5.2: Advanced Layers (MHA, FFN, Transformer Block, Embedding)
4. FASE 5.3: Example Model Builders (demonstrar uso do framework genérico)
5. FASE 5.4: Additional Architectures (MLP, CNN, RNN - futuro)

### Comando Inicial (Para Novos Desenvolvedores)

Para começar do zero, peça ao Cursor:

> **"Atue como Qorus-Architect. Vamos iniciar a Fase 1. Primeiro, crie a estrutura de diretórios e os arquivos de cabeçalho `include/qorus_types.h` e `include/qorus.h` com as definições de Tensor e Contexto conforme o Blueprint v2.0."**

Isso garante que o projeto comece com a estrutura correta.

---

## 9. REFERÊNCIAS DE PLANEJAMENTO

**Documentos Executivos:**
- `docs/PROJECT_VISION.md` - **Visão completa do projeto (início → atual → fim)**
- `docs/TIMELINE.md` - **Timeline de desenvolvimento com estimativas e dependências**
- `docs/INDEX.md` - **Índice mestre da documentação - guia de navegação**

**Documentos de Planejamento (FASE 2.5 - Inference Kernels):**
- `docs/KERNEL_PORTATION_PLAN.md` - Plano completo de portação seguindo MFR + CoT + Mathematical Proof + TDD
- `docs/KERNEL_IMPLEMENTATION_DETAILS.md` - Guia de implementação com código completo e exemplos
- `docs/PLANNING_SUMMARY.md` - Resumo executivo do planejamento

**Documentos de Planejamento (FASE 2.6 - Training Capability):**
- `docs/TRAINING_CAPABILITY_PLAN.md` - Plano completo de capacidade de treinamento (MFR + CoT + Proof + TDD)

**Documentos de Planejamento (FASE 5.0+ - Generic Framework v3.0):**
- `docs/GENERIC_FRAMEWORK_PLAN.md` - **Plano completo de framework genérico (MFR + CoT + Proof + TDD)**

**Documentação de Qualidade:**
- `docs/REFACTORING_CHECKPOINTS.md` - **Procedimentos de checkpoint de refatoração e garantia de qualidade**

**Documentação Técnica:**
- `docs/STATUS.md` - Status detalhado do projeto
- `docs/QUICK_REFERENCE.md` - Referência rápida
- `docs/FASE_3.3_ANALYSIS.md` - Análise do forward pass
- `docs/PRECISION_STANDARDS.md` - Padrões de precisão numérica
- `docs/ASYMPTOTIC_ANALYSIS.md` - Análise assintótica
- `docs/.cursorrules` - Metodologia de desenvolvimento (MFR + CoT + Proof + TDD)

