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
│   │   ├── avx2/           # Kernels AVX2 (MatMul Q4, MatMul FP32, RoPE, RMSNorm, Add, Mul, Causal Mask, Loss, Clip)
│   │   └── cuda/           # Kernels CUDA (Para Google Colab / GPU) - FUTURO
│   │       ├── q_cuda_utils.cu  # Gerenciamento de memória GPU
│   │       ├── matmul.cu        # Calls to cuBLAS
│   │       └── rope.cu          # Custom Kernel
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
│       └── dummy_tokenizer.c  # Dummy Tokenizer (Testing Only - NOT real BPE)
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

// Device type (CPU or GPU) - NEW for CUDA support
typedef enum {
    Q_DEVICE_CPU = 0,
    Q_DEVICE_CUDA = 1
} q_device_type;

// Tensor View (Não possui a memória, apenas aponta)
typedef struct {
    void*     data;         // Ponteiro para dados (Mmap, Arena, ou GPU)
    float*    scales;       // Ponteiro para escalas (se quantizado)
    uint32_t  ne[4];        // Dimensões: [Batch, Head, Seq, Dim]
    size_t    nb[4];        // Strides em bytes
    q_dtype   type;         // Tipo de dado
    q_device_type device;  // NEW: CPU ou CUDA (para seleção automática de kernel)
    char      name[32];     // Debugging
} __attribute__((aligned(Q_ALIGN))) q_tensor;

// Contexto Global de Memória
typedef struct {
    // Tier 1: Weights (Read-Only)
    void* weights_mmap;     // CPU: mmap, GPU: NULL (pesos ficam em GPU)
    size_t weights_size;
    
    // Tier 2: KV Cache (Persistent)
    void* kv_buffer;        // CPU: aligned_alloc, GPU: cudaMalloc
    size_t kv_size;
    q_device_type kv_device;  // NEW: Onde está o KV cache
    
    // Tier 3: Scratchpad (Transient)
    void* scratch_buffer;   // CPU: aligned_alloc, GPU: cudaMalloc
    size_t scratch_size;
    size_t scratch_head;    // Posição atual na Arena
    q_device_type scratch_device;  // NEW: Onde está o scratchpad
    
    // NEW: CUDA context (se disponível)
    void* cuda_context;     // NULL se não usar CUDA
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
  - **CPU:** `aligned_alloc` (como antes)
  - **CUDA:** `cudaMalloc` (zero-malloc mantido no hot path)
- **Uso:** Ativações intermediárias (saída de MatMul, Softmax).
- **Ciclo:** `scratch_head` é resetado para 0 no início de cada token gerado.
- **Regra:** NUNCA dar `free()` em tensores individuais aqui.

### Adaptação para CUDA (FASE 2.7 - Planejamento):

**Estratégia de Memória GPU:**
- **Weights:** Transferir do mmap para GPU na inicialização (uma vez)
- **KV Cache:** Pode ficar em GPU ou CPU (configurável)
- **Scratchpad:** Usar `cudaMalloc` normal (zero-malloc mantido)
- **Pinned Memory:** Apenas para buffers persistentes (Tier 2), não no hot path

**Problema do mmap no Google Drive:**
- Google Drive usa fuse filesystem (muito lento para mmap)
- **Solução:** Detectar fuse e copiar modelo para `/tmp` antes de mmap
- Implementado em `q_init_memory_smart()` (FASE 2.7)

---

## 4. ROTEIRO DE IMPLEMENTAÇÃO (Step-by-Step)

**ORDEM CORRETA DE IMPLEMENTAÇÃO:** Execute as fases nesta ordem exata. Não avance sem validar critérios objetivos.

**Estrutura do Roteiro:**
- **PARTE 1:** Inferência (FASE 1-4) - Sistema completo de inferência
- **PARTE 2:** Treinamento (FASE 2.6-3.5) - Capacidade de treinamento
- **PARTE 3:** Framework Genérico (FASE 5.0+) - Evolução para v3.0

---

## PARTE 1: INFERÊNCIA (v2.0) - Sistema Completo de Inferência

**Objetivo:** Sistema completo de inferência funcional, do carregamento de modelo até geração de texto.

---

### ✅ FASE 1: Infraestrutura & Conversor (A Base) - **COMPLETA**

**Objetivo:** Conseguir carregar pesos do disco sem parsing.

**Implementação:**
- ✅ **Passo 1.1 (Python):** `tools/convert_llama.py` criado. Gera arquivo `.qorus` com header fixo e tensores alinhados a 64 bytes.
- ✅ **Passo 1.2 (C):** `src/core/memory.c` implementado (mmap, arena). `src/core/tensor.c` implementado (criação de views).
- ✅ **Validação:** Testes de memória validados. Carregamento de modelo dummy funcionando.

**Critérios Objetivos de Qualidade (FASE 1):**
- ✅ **Testes:** 100% pass rate em todos os testes de memória e tensor
- ✅ **Zero-Malloc:** Nenhuma alocação dinâmica no hot path (apenas inicialização)
- ✅ **Alinhamento:** Todos os buffers alinhados a 64 bytes (verificado com `_Static_assert`)
- ✅ **Validação:** Modelo dummy carregado e validado com sucesso
- ✅ **Sanitizers:** AddressSanitizer e MemorySanitizer passam sem erros

**Checkpoint de Refatoração (Após FASE 1):**
- ✅ **Status:** Concluído
- ✅ **Áreas Verificadas:**
  - Consistência de alinhamento de memória
  - Padronização de tratamento de erros em `memory.c`
  - Validação de mmap e arena allocation
- ✅ **Métricas:** Zero regressões de performance, todos os testes passando

---

### ✅ FASE 2: Kernels Matemáticos Básicos (O Motor) - **COMPLETA**

**Objetivo:** Operações vetoriais rápidas com validação robusta.

**Implementação:**
- ✅ **Passo 2.1:** `src/ops/avx2/dequantize.c` implementado. Q4_0 → 32 floats em YMM, FMA-optimized.
- ✅ **Passo 2.2:** `src/ops/avx2/matmul.c` implementado. GEMV Q4_F32 com dequantização fundida, 4x unrolling.
  - ✅ **Validação de Contiguidade (2025-01-02):** Validação crítica de que tensor é contíguo em memória antes de execução
    - Valida `nb[0] == expected_stride` para prevenir leitura de memória inválida
    - Falha com erro claro se tensor não for contíguo (v1.0 limitation)
    - Documentação clara de limitação arquitetural
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

**Critérios Objetivos de Qualidade (FASE 2):**
- ✅ **Testes:** 100% pass rate em todos os testes de kernel
- ✅ **Precisão Numérica:** Max absolute difference < 1e-5, Max relative difference < 1e-4 (FP32)
- ✅ **Validação:** Todos os kernels validados contra referências NumPy/PyTorch
- ✅ **Zero-Malloc:** Nenhuma alocação dinâmica no hot path
- ✅ **Performance:** Benchmarks mantidos ou melhorados vs referência
- ✅ **Sanitizers:** AddressSanitizer, MemorySanitizer, UndefinedBehaviorSanitizer passam
- ✅ **Validação de Erros:** Todos os códigos de erro testados e documentados

**Checkpoint de Refatoração (Após FASE 2):**
- ✅ **Status:** Concluído
- ✅ **Áreas Verificadas:**
  - Consistência de interface de kernels (assinaturas padronizadas)
  - Tratamento de erros consistente (`q_error_code` em todas as funções)
  - Padrões de otimização AVX2 verificados
- ✅ **Métricas:** Zero regressões, performance mantida, todos os testes passando

---

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

**Critérios Objetivos de Qualidade (FASE 2.5):**
- ✅ **Testes:** 100% pass rate (Release + Debug with sanitizers)
- ✅ **Precisão Numérica:** Max absolute difference < 1e-5, Max relative difference < 1e-4 (FP32)
- ✅ **Code Review:** Completado (First Principles Thinking + CoT)
- ✅ **Edge Cases:** Tratados (NULL inputs, shape mismatches, alignment)
- ✅ **Operações In-Place:** Suportadas (safe aliasing)
- ✅ **Validação de Precisão:** Max diff < 1e-5 para FP32
- ✅ **Validação de Memória:** AddressSanitizer clean
- ✅ **Validação:** Todos os kernels validados contra referências NumPy

**Checkpoint de Refatoração (Após FASE 2.5):**
- ✅ **Status:** Concluído
- ✅ **Áreas Verificadas:**
  - Consistência de interface de kernels (assinaturas padronizadas)
  - Padronização de tratamento de erros
  - Otimização de performance AVX2 verificada
  - Cobertura de testes completa
- ✅ **Métricas:** Zero regressões, performance mantida ou melhorada, todos os testes passando

**Documentação:**
- `docs/KERNEL_PORTATION_PLAN.md` - Plano completo seguindo MFR + CoT + Mathematical Proof + TDD (Status: ✅ COMPLETA)
- `docs/KERNEL_IMPLEMENTATION_DETAILS.md` - Guia de implementação com código completo
- `docs/PLANNING_SUMMARY.md` - Resumo executivo do planejamento

---

### ✅ Dívida Técnica de Baixa Prioridade - **COMPLETA**

**Objetivo:** Estabelecer base sólida de testes, benchmarking e documentação antes de avançar para forward pass.

**Implementação:**
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

**Critérios Objetivos de Qualidade:**
- ✅ **Cobertura de Testes:** Todos os utilitários testados (100% pass rate)
- ✅ **Benchmarks:** Ferramenta funcional e validada
- ✅ **Documentação:** Análise assintótica completa para todas as funções críticas

---

### ✅ FASE 3: Model Graph Building (O Corpo) - **PARCIALMENTE COMPLETA**

**Objetivo:** Conectar os kernels na ordem correta usando framework genérico.

**Implementação:**
- ✅ **Passo 3.1:** Definir estruturas genéricas em `qorus_types.h`.  
  **Status:** Estruturas definidas e validadas com `_Static_assert`.

- ✅ **Passo 3.2:** Implementar `q_model_build_graph()`. Configurar ponteiros dos tensores baseados no arquivo mmap.
  **Status:** Implementado e testado (31 testes, 100% pass rate).
  - Zero-copy tensor views
  - Validação completa de configuração
  - Suporte a Q4_0 e FP32
  - Testes adversarial completos

- ✅ **Passo 3.3:** Implementar `llama_forward()`. Orquestrar passagem dos dados pelos kernels usando framework genérico.
  **Status:** ✅ **COMPLETA** (2025-01-02)
  **Dependências:** ✅ Todas resolvidas (FASE 2.5 completa)
    - ✅ MatMul FP32 AVX2 (Q @ K^T, probs @ V, projection layers)
    - ✅ Causal Masking AVX2 (attention mask)
    - ✅ Tensor Add AVX2 (residual connections)
    - ✅ Element-wise Mul AVX2 (SwiGLU activation)
  **Implementação Completa:**
    - ✅ Estrutura do forward pass completa
    - ✅ KV cache helper implementado (`get_kv_cache_ptr`)
    - ✅ MLP forward pass completo (SwiGLU)
    - ✅ Layer forward pass completo (attention + MLP com residuals)
    - ✅ Attention forward pass completo (Q/K/V projections, RoPE, KV cache, causal mask, softmax)
    - ✅ Final RMSNorm implementado
    - ✅ LM Head projection implementado (transposed view)
    - ✅ Token embedding lookup implementado
    - ✅ Validações de segurança implementadas
    - ✅ Estrutura `q_llama_layer` definida e integrada
    - ✅ Correção de alinhamento em softmax (buffers alinhados para cada linha)
    - ✅ Debug aprimorado em validações de alinhamento (`Q_VALIDATE_ALIGNED_OR_RETURN`)
  **Testes:** ✅ Todos passando (14 testes unitários + 19 testes adversariais, 100% pass rate)
    - ✅ Forward pass básico (single token, multiple tokens)
    - ✅ Geração incremental (pos > 0)
    - ✅ Validação de logits (finite, shape correto)
    - ✅ Tratamento de erros (NULL pointers, invalid sizes, invalid positions)
    - ✅ Testes adversariais completos (19 testes, 100% pass rate):
      - ✅ NULL pointer attacks
      - ✅ Edge cases (empty sequences, invalid token IDs)
      - ✅ Memory safety (buffer overflows, double-free)
      - ✅ Large sequences (seq_len = 100)
      - ✅ Misaligned memory attacks
      - ✅ Corrupted model data
      - ✅ Numerical stability attacks

**Critérios Objetivos de Qualidade (FASE 3.3):**
- ✅ **Testes:** 100% pass rate (14 testes unitários + 19 testes adversariais)
- ✅ **Validação:** Forward pass completo validado end-to-end
- ✅ **Zero-Malloc:** Nenhuma alocação dinâmica no hot path (apenas arena)
- ✅ **Validação de Erros:** Todos os códigos de erro testados e documentados
- ✅ **Sanitizers:** AddressSanitizer, MemorySanitizer passam sem erros
- ✅ **Alinhamento:** Todos os buffers alinhados a 32 bytes (verificado)
- ✅ **Performance:** Benchmarks mantidos ou melhorados

**Checkpoint de Refatoração (Após FASE 3.3):**
- ✅ **Status:** Concluído
- ✅ **Áreas Verificadas:**
  - Arquitetura do forward pass revisada
  - Integração de layers padronizada
  - Performance do forward pass otimizada
  - Tratamento de erros melhorado
  - Testes de forward pass completos
- ✅ **Métricas:** Zero regressões, performance mantida, todos os testes passando
- ✅ **Limite de Tempo:** 1-2 dias (concluído)

**Nota:** Framework genérico permite qualquer arquitetura, não apenas Transformers.

---

### ✅ FASE 4: Tokenizer & Loop (A Vida) - **COMPLETA**

**Objetivo:** Texto entra, texto sai.

**Implementação:**
- ✅ **Passo 4.1:** Implementar `src/tokenizer/dummy_tokenizer.c`. Carregar `tokenizer.bin` (extraído do modelo original).
  - **Status:** ✅ **COMPLETA** (2025-01-02) - **ATUALIZADO** (2025-01-02)
  - **Arquivos Implementados:**
    - `src/tokenizer/dummy_tokenizer.c` - Dummy Tokenizer para testes (350+ linhas)
    - **⚠️ IMPORTANTE:** Este é um **Dummy Tokenizer** (NÃO implementa BPE real)
    - **Limitações:**
      - Não implementa algoritmo BPE (Byte Pair Encoding)
      - Mapeia bytes diretamente para token IDs (byte value = token ID se < vocab_size)
      - Não usa regras de merge carregadas do arquivo tokenizer
    - **Casos de Uso:**
      - Testes de infraestrutura com inputs pré-tokenizados
      - Desenvolvimento/debugging com tokens byte-level
      - **NÃO adequado para inferência em produção com modelos Transformer reais**
    - **Para Produção:**
      - Implementar algoritmo BPE completo (aplicação greedy de merges)
      - Ou usar inputs pré-tokenizados de tokenizer externo
    - `include/qorus_types.h` - Estruturas `q_tokenizer` e `q_bpe_merge`
    - `include/qorus.h` - API pública completa
    - `tools/convert_llama.py` - Função `write_tokenizer()` para exportação
    - `tests/test_tokenizer.c` - Testes completos (Release + Debug)
    - `examples/hello_world.c` - Exemplo funcional "Hello World"
  - **Estruturas de Dados:**
    ```c
    typedef struct {
        char** vocab;              // Array de token strings [vocab_size]
        uint32_t vocab_size;       // Tamanho do vocabulário
        q_bpe_merge* merges;       // Array de regras BPE [num_merges]
        uint32_t num_merges;       // Número de merges BPE
        uint32_t bos_token_id;     // Beginning of sequence token ID
        uint32_t eos_token_id;     // End of sequence token ID
        uint32_t pad_token_id;     // Padding token ID
        bool initialized;          // Flag de inicialização
    } q_tokenizer;
    ```
  - **Formato Binário:**
    - **Header (32 bytes):** Magic (4B), Version (4B), vocab_size (4B), num_merges (4B), bos_id (4B), eos_id (4B), pad_id (4B), reserved (4B)
    - **Vocab Section:** Para cada token: length (1B) + token_bytes (N bytes)
    - **Merges Section:** Para cada merge: token_id1 (4B) + token_id2 (4B) + merged_id (4B)
  - **API Pública:**
    - `q_tokenizer_load()` - Carrega tokenizer de arquivo binário
    - `q_tokenizer_encode()` - Converte texto → tokens (com suporte a BOS/EOS)
    - `q_tokenizer_decode()` - Converte tokens → texto
    - `q_tokenizer_free()` - Libera recursos do tokenizer
  - **Funcionalidades:**
    - ✅ Carregamento de tokenizer binário (formato customizado)
    - ✅ Encode: texto → tokens (com suporte a BOS/EOS)
    - ✅ Decode: tokens → texto
    - ✅ Vocabulário base: 256 tokens (bytes 0-255) + 3 tokens especiais (BOS=256, EOS=257, PAD=258)
    - ✅ Validações de segurança implementadas (Q_VALIDATE_PTR_OR_RETURN, etc.)
    - ✅ Gerenciamento de memória seguro (cleanup em caso de erro)
  - **Complexidade:**
    - Load: O(V + M) onde V=vocab_size, M=num_merges
    - Encode: O(T) onde T=text_length (mapeamento direto byte→token, sem BPE merges)
    - Decode: O(N) onde N=num_tokens
  - **⚠️ Limitação Crítica:**
    - O tokenizer atual é um **placeholder** que não implementa BPE real
    - Para produção com modelos Transformer reais, é necessário implementar algoritmo BPE completo
    - Ou usar inputs pré-tokenizados de tokenizer externo (ex: HuggingFace tokenizers)
  - **Testes:** ✅ Todos passando (Release + Debug com sanitizers)
    - Teste de carregamento
    - Teste de encode/decode
    - Teste de BOS/EOS tokens
    - Hello World funcionando: "Hello World" → tokens → "Hello World"
  - **Ferramenta de Exportação:**
    ```bash
    python3 tools/convert_llama.py --tokenizer tokenizer.bin [vocab_size]
    ```
  - **Documentação:** `docs/TOKENIZER_IMPLEMENTATION.md` - Documentação completa

- ✅ **Passo 4.2:** Criar `main.c`. Loop: Tokenize -> Forward -> Sample -> Print -> Update Cache.
  - **Status:** ✅ **COMPLETA** (2025-01-02)
  - **Implementação:**
    - ✅ Interface de linha de comando (CLI)
    - ✅ Loop de geração: Tokenize input → Forward pass → Sample → Print → Update KV Cache
    - ✅ Suporte a prompts interativos
    - ✅ Tratamento de erros robusto (verificar `q_error_code` em todas as chamadas)
    - ✅ Integração com tokenizer (FASE 4.1 completa)
    - ✅ Integração com forward pass (FASE 3.3 completa)
    - ✅ Sampling strategies: Greedy, Temperature, Top-k, Top-p, Combined Top-k+Top-p
    - ✅ Performance benchmarks implementados
  - **Dependências:** 
    - ✅ FASE 4.1 (Tokenizer) - COMPLETA
    - ✅ FASE 3.3 (Forward Pass) - COMPLETA
  - **Nota:** Todas as chamadas de funções matemáticas verificam retorno `q_error_code`.
  
- ✅ **Passo 4.3:** Auditoria de Performance e Otimizações Críticas.
  - **Status:** ✅ **COMPLETA** (2025-01-02)
  - **Problema Crítico Identificado e Corrigido:**
    - 🔴 **Top-p catastrófico:** ~60× mais lento que greedy (~6000 ms/token)
    - **Causa Raiz:** Memcpy repetido no binary search (3.84 MB copiado desnecessariamente)
    - **Solução:** Sort completo UMA VEZ + binary search no cumsum prefixo (sem memcpy repetido)
    - **Resultado:** ~11× melhoria (5985 ms → 532 ms/token)
  - **Status de Performance Atual:**
    - ✅ **Greedy:** ~100 ms/token (perfeito)
    - ✅ **Prefill:** ~26 ms/token (ótimo)
    - ✅ **Top-p=0.9:** ~532 ms/token (corrigido, ~11× melhoria)
    - ⚠️ **Top-k=10:** ~616 ms/token (aceitável, pode melhorar)
  - **Documentação:**
    - `docs/src-docs/AUDIT_PERFORMANCE_TOP_P_CRITICAL.md` - Auditoria detalhada
    - `docs/src-docs/AUDIT_PERFORMANCE_TOP_K.md` - Análise de top-k
    - `docs/AUDITORIA_PERFORMANCE_COMPLETA.md` - Resumo consolidado
    - `docs/CORRECAO_TOP_P_IMPLEMENTADA.md` - Documentação da correção

**Critérios Objetivos de Qualidade (FASE 4.1):**
- ✅ **Testes:** 100% pass rate (Release + Debug com sanitizers)
- ✅ **Validação:** Tokenizer validado end-to-end (encode/decode round-trip)
- ✅ **Validação de Erros:** Todos os códigos de erro testados
- ✅ **Sanitizers:** AddressSanitizer, MemorySanitizer passam sem erros
- ✅ **Exemplo Funcional:** Hello World funcionando corretamente

**Critérios Objetivos de Qualidade (FASE 4.2):**
- ✅ **Testes:** 100% pass rate em testes de main loop
- ✅ **Validação:** Loop de geração validado end-to-end
- ✅ **Validação de Erros:** Todos os códigos de erro tratados corretamente
- ✅ **Performance:** Latência de geração medida e documentada
  - Greedy: ~100 ms/token
  - Top-p: ~532 ms/token (corrigido de ~6000 ms)
  - Top-k: ~616 ms/token
- ✅ **Sanitizers:** AddressSanitizer, MemorySanitizer passam sem erros
- ✅ **Auditoria de Performance:** Completa com correções críticas implementadas

**Checkpoint de Refatoração (Após FASE 4.2):**
- ✅ **Status:** Concluído (2025-01-02)
- ✅ **Áreas Verificadas:**
  - Arquitetura do main loop
  - Integração tokenizer + forward pass
  - Tratamento de erros robusto
  - Performance do loop de geração
  - Otimizações críticas de sampling (top-p corrigido)
- ✅ **Métricas:** Zero regressões, performance otimizada, todos os testes passando

---

## PARTE 2: CAPACIDADE DE TREINAMENTO (Após Inferência Completa)

**Nota:** As fases abaixo devem ser implementadas após a conclusão da FASE 4 (Tokenizer & Loop), quando o sistema de inferência estiver completo e funcional. **Status:** ✅ FASE 4 COMPLETA (2025-01-02)

---

### ⏳ FASE 2.6: Training Kernels (Planejamento Completo) - **PLANEJAMENTO COMPLETO**

**Objetivo:** Adicionar capacidade de treinamento para future-implementations (Code Agent, Customer Behavior Prediction, SEO AI Specialist).

**Status:** 📋 Planejamento completo (2024-12-30). Pronto para implementação após FASE 4.2.

**Dependências:**
- ✅ FASE 4.2 (Main Loop) - Deve estar completa antes de iniciar

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

**Critérios Objetivos de Qualidade (FASE 2.6 - Pendente):**
- ⏳ **Testes:** 100% pass rate em todos os testes de optimizer e loss functions
- ⏳ **Precisão Numérica:** Max absolute difference < 1e-5, Max relative difference < 1e-4 (FP32)
- ⏳ **Validação:** Todos os optimizers validados contra referências PyTorch
- ⏳ **Zero-Malloc:** Nenhuma alocação dinâmica no hot path (apenas arena)
- ⏳ **Performance:** Benchmarks mantidos ou melhorados
- ⏳ **Sanitizers:** AddressSanitizer, MemorySanitizer, UndefinedBehaviorSanitizer passam

**Checkpoint de Refatoração (Após FASE 2.6 - Pendente):**
- ⏳ **Status:** Pendente
- ⏳ **Áreas a Verificar:**
  - Consistência de interface de optimizers
  - Padronização de tratamento de erros
  - Otimização de performance AVX2
  - Cobertura de testes completa
- ⏳ **Limite de Tempo:** 1 dia

**Documentação de Planejamento:**
- `docs/TRAINING_CAPABILITY_PLAN.md` - Plano completo de capacidade de treinamento

---

### ⏳ FASE 2.7: CUDA Support (Google Colab / GPU) - **PLANEJAMENTO COMPLETO**

**Objetivo:** Adicionar suporte CUDA para treinamento acelerado em GPU (Google Colab, NVIDIA GPUs).

**Status:** 📋 Planejamento completo (2025-01-02). Pronto para implementação após abstrações necessárias.

**Análise Crítica Aplicada:**
- ✅ **Problema Identificado:** Falta de abstração de device em `q_tensor`
- ✅ **Problema Identificado:** Conflito potencial com Zero-Malloc no hot path
- ✅ **Problema Identificado:** Estrutura de diretórios incompleta para seleção de kernel
- ✅ **Solução Proposta:** Abstração de device + gerenciamento unificado de memória

**Componentes Planejados:**

- ⏳ **Abstração de Device** (`include/qorus_types.h`)
  - **Estrutura:** `q_device_type` enum (CPU, CUDA)
  - **Tempo Estimado:** 2-3 horas
  - **Características:**
    - Adicionar campo `device` em `q_tensor`
    - Adicionar campos `kv_device` e `scratch_device` em `q_context`
    - Adicionar campo `cuda_context` em `q_context`

- ⏳ **Gerenciamento de Memória Unificado** (`src/core/memory.c`)
  - **Funções:** `q_alloc_kv_cache_ex()`, `q_alloc_arena_ex()` com suporte a device
  - **Tempo Estimado:** 4-6 horas
  - **Características:**
    - CPU: `aligned_alloc` (como antes)
    - CUDA: `cudaMalloc` (zero-malloc mantido no hot path)
    - Pinned memory apenas para buffers persistentes (Tier 2: KV Cache)
    - Scratchpad usa `cudaMalloc` normal (zero-malloc mantido)

- ⏳ **Interface Comum com Seleção Automática** (`src/ops/`)
  - **Funções:** `q_matmul_f32()`, `q_add_f32()`, etc. (wrapper que seleciona kernel)
  - **Tempo Estimado:** 6-8 horas
  - **Características:**
    - Interface pública permanece a mesma (`qorus.h`)
    - Seleção automática de kernel baseada em `device` do tensor
    - Fallback para CPU se CUDA não disponível

- ⏳ **Kernels CUDA** (`src/ops/cuda/`)
  - **Arquivos:** `q_cuda_utils.cu`, `matmul.cu`, `rope.cu`, etc.
  - **Tempo Estimado:** 20-30 horas
  - **Características:**
    - CUDA kernels para operações críticas
    - Integração com cuBLAS para MatMul
    - Custom kernels para RoPE, RMSNorm, etc.

- ⏳ **Resolução do Problema do mmap no Google Drive** (`src/core/memory.c`)
  - **Função:** `q_init_memory_smart()` com detecção de fuse filesystem
  - **Tempo Estimado:** 2-3 horas
  - **Características:**
    - Detecta Google Drive (fuse filesystem)
    - Copia modelo para `/tmp` antes de mmap
    - Mantém compatibilidade com sistemas normais

**Total Estimado (FASE 2.7):** 34-50 horas

**Dependências:**
- ✅ FASE 3.3 (Forward Pass) - COMPLETA (necessária para testar kernels CUDA)
- ✅ FASE 4.2 (Main Loop) - Recomendado estar completa antes de iniciar
- ⏳ Abstrações de device (pré-requisito)

**Nota:** Pode ser implementada em paralelo com FASE 2.6 (Training Kernels) para acelerar treinamento em GPU.

**Critérios Objetivos de Qualidade (FASE 2.7 - Pendente):**
- ⏳ **Testes:** 100% pass rate em todos os testes CUDA
- ⏳ **Precisão Numérica:** Max absolute difference < 1e-5, Max relative difference < 1e-4 (FP32)
- ⏳ **Validação:** Todos os kernels CUDA validados contra referências CPU
- ⏳ **Zero-Malloc:** Nenhuma alocação dinâmica no hot path (apenas `cudaMalloc` na inicialização)
- ⏳ **Performance:** Speedup medido vs CPU (objetivo: >2x para operações grandes)
- ⏳ **Compatibilidade:** Código CPU existente continua funcionando sem mudanças
- ⏳ **Sanitizers:** CUDA-Memcheck passa sem erros

**Checkpoint de Refatoração (Após FASE 2.7 - Pendente):**
- ⏳ **Status:** Pendente
- ⏳ **Áreas a Verificar:**
  - Abstração de device funcionando corretamente
  - Seleção automática de kernel validada
  - Gerenciamento de memória GPU otimizado
  - Compatibilidade CPU mantida
- ⏳ **Limite de Tempo:** 1-2 dias

**Documentação de Planejamento:**
- `docs/CUDA_ADAPTATION_PLAN.md` - Plano completo de adaptação CUDA (a ser criado)

**Análise Crítica Completa (First Principles Thinking + CoT):**

**Problemas Identificados e Soluções:**

1. **Falta de Abstração de Device:**
   - **Problema:** `q_tensor` não distingue entre CPU e GPU, causando crashes se ponteiro GPU for passado para kernel AVX2
   - **Solução:** Adicionar campo `q_device_type device` em `q_tensor`
   - **Impacto:** Permite seleção automática de kernel baseada em device
   - **Prova:** Se `q_tensor.data` aponta para GPU mas kernel AVX2 é chamado → CRASH. Com `device`, seleção automática previne isso.

2. **Conflito com Zero-Malloc:**
   - **Problema:** `cudaHostAlloc` quebra zero-malloc no hot path (é alocação)
   - **Solução:** Usar `cudaMalloc` normal no hot path, `cudaHostAlloc` apenas para buffers persistentes (Tier 2: KV Cache)
   - **Impacto:** Mantém garantia zero-malloc mesmo com CUDA
   - **Prova:** Zero-malloc = zero alocações no hot path. `cudaMalloc` é alocação, mas apenas na inicialização (não no hot path). `cudaHostAlloc` seria alocação no hot path → quebra garantia.

3. **Estrutura de Diretórios:**
   - **Problema:** Falta abstração para seleção de kernel (runtime vs compile-time)
   - **Solução:** Interface comum (`q_matmul_f32()`) que seleciona kernel automaticamente baseado em `device`
   - **Impacto:** Código cliente não precisa mudar, seleção transparente
   - **Prova:** Sem abstração, código cliente precisa saber qual kernel chamar → duplicação. Com abstração, uma função pública seleciona automaticamente.

4. **Problema do mmap no Google Drive:**
   - **Problema:** Fuse filesystem é muito lento para mmap (latência de milissegundos vs nanossegundos)
   - **Solução:** Detectar fuse e copiar modelo para `/tmp` antes de mmap
   - **Impacto:** Performance normal mesmo no Google Colab
   - **Prova:** Fuse filesystem tem overhead de rede → mmap bloqueia. Copiar para `/tmp` (SSD local) → mmap rápido.

**Estrutura de Implementação:**

```c
// Interface pública (não muda) - em qorus.h
q_error_code q_matmul_f32(const q_tensor* A, const q_tensor* B, 
                          const q_tensor* C, q_context* ctx);

// Implementação interna seleciona kernel automaticamente - em src/ops/matmul.c
q_error_code q_matmul_f32(const q_tensor* A, const q_tensor* B,
                          const q_tensor* C, q_context* ctx) {
    // Auto-select kernel based on device
    if (A->device == Q_DEVICE_CUDA || B->device == Q_DEVICE_CUDA) {
        return q_matmul_f32_cuda(A, B, C, ctx);
    } else {
        return q_matmul_f32_avx2(A, B, C, ctx);
    }
}
```

**Gerenciamento de Memória Unificado:**

```c
// Extensão do q_context para suportar GPU - em qorus_types.h
typedef struct {
    // Tier 1: Weights (Read-Only)
    void* weights_mmap;       // CPU: mmap, GPU: NULL (pesos ficam em GPU)
    size_t weights_size;
    
    // Tier 2: KV Cache (Persistent)
    void* kv_buffer;          // CPU: aligned_alloc, GPU: cudaMalloc
    size_t kv_size;
    q_device_type kv_device;  // NEW: Onde está o KV cache
    
    // Tier 3: Scratchpad (Transient)
    void* scratch_buffer;     // CPU: aligned_alloc, GPU: cudaMalloc
    size_t scratch_size;
    size_t scratch_head;
    q_device_type scratch_device;  // NEW: Onde está o scratchpad
    
    // NEW: CUDA context (se disponível)
    void* cuda_context;       // NULL se não usar CUDA
} q_context;
```

**Resolução do Problema do mmap no Google Drive:**

```c
// Função helper para detectar e copiar se necessário - em src/core/memory.c
q_error_code q_init_memory_smart(q_context* ctx, const char* model_path) {
    // Detectar se é Google Drive (fuse filesystem)
    struct statfs fs_info;
    if (statfs(model_path, &fs_info) == 0) {
        if (fs_info.f_type == 0x65735546) {  // FUSE magic number
            // É fuse: copiar para /tmp primeiro
            char tmp_path[PATH_MAX];
            snprintf(tmp_path, sizeof(tmp_path), "/tmp/qorus_model_%d.bin", getpid());
            // Copiar arquivo...
            return q_init_memory(ctx, tmp_path);
        }
    }
    // Não é fuse: usar diretamente
    return q_init_memory(ctx, model_path);
}
```

**Notas Importantes:**
- **Zero-Malloc Mantido:** Usar `cudaMalloc` (não `cudaHostAlloc`) no hot path
- **Compatibilidade:** Código CPU existente continua funcionando sem mudanças (device padrão = CPU)
- **Performance:** Seleção de kernel em runtime tem overhead mínimo (< 1 ciclo, apenas comparação de enum)
- **Abstração:** Interface pública não muda, seleção automática transparente

---

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

**Dependências:**
- ✅ FASE 2.6 (Optimizers, Loss Functions, Gradient Clipping) - Deve estar completa antes de iniciar

**Critérios Objetivos de Qualidade (FASE 3.4 - Pendente):**
- ⏳ **Testes:** 100% pass rate em todos os testes de backward pass
- ⏳ **Validação:** Gradientes validados contra referências PyTorch (gradient checking)
- ⏳ **Zero-Malloc:** Nenhuma alocação dinâmica no hot path (apenas arena)
- ⏳ **Performance:** Benchmarks mantidos ou melhorados
- ⏳ **Sanitizers:** AddressSanitizer, MemorySanitizer passam sem erros

**Checkpoint de Refatoração (Após FASE 3.4 - Pendente):**
- ⏳ **Status:** Pendente
- ⏳ **Áreas a Verificar:**
  - Arquitetura do backward pass
  - Integração com forward cache
  - Propagação de gradientes validada
  - Performance do backward pass
- ⏳ **Limite de Tempo:** 1-2 dias

---

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

**Dependências:**
- ✅ FASE 3.4 (Backward Pass) - Deve estar completa antes de iniciar

**Critérios Objetivos de Qualidade (FASE 3.5 - Pendente):**
- ⏳ **Testes:** 100% pass rate em todos os testes de training loop
- ⏳ **Validação:** Training loop validado end-to-end (converge em dataset pequeno)
- ⏳ **Zero-Malloc:** Nenhuma alocação dinâmica no hot path (apenas arena)
- ⏳ **Performance:** Throughput de treinamento medido e documentado
- ⏳ **Sanitizers:** AddressSanitizer, MemorySanitizer passam sem erros

**Checkpoint de Refatoração (Após FASE 3.5 - Pendente):**
- ⏳ **Status:** Pendente
- ⏳ **Áreas a Verificar:**
  - Arquitetura do training loop
  - Integração de optimizer
  - Integração de loss function
  - Fluxo de gradientes
  - Performance de treinamento
- ⏳ **Limite de Tempo:** 1-2 dias

---

## PARTE 3: EVOLUÇÃO PARA v3.0 - FRAMEWORK GENÉRICO

**Nota:** As fases abaixo devem ser implementadas após a conclusão das PARTES 1 e 2, quando tanto inferência quanto treinamento estiverem completos e funcionais.

### Objetivo v3.0

Transformar QorusIA de engine especializado em **framework genérico** sem limitações arquiteturais, mantendo:
- ✅ Performance máxima (zero-malloc, AVX2)
- ✅ Arquitetura limpa (validações robustas)
- ✅ Flexibilidade total (qualquer arquitetura)

**Dependências:**
- ✅ PARTE 1: Inferência completa (FASE 1-4)
- ✅ PARTE 2: Treinamento completo (FASE 2.6-3.5) - Recomendado estar completa antes de iniciar

---

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

**Critérios Objetivos de Qualidade (FASE 5.0 - Pendente):**
- ⏳ **Testes:** 100% pass rate em todos os testes de framework genérico
- ⏳ **Validação:** Framework genérico validado com modelo Transformer
- ⏳ **Zero-Malloc:** Nenhuma alocação dinâmica no hot path (apenas arena)
- ⏳ **Performance:** Overhead de function pointers < 1% (medido)
- ⏳ **Sanitizers:** AddressSanitizer, MemorySanitizer passam sem erros

**Checkpoint de Refatoração (Após FASE 5.0 - Pendente):**
- ⏳ **Status:** Pendente
- ⏳ **Áreas a Verificar:**
  - Design de interface genérica
  - Overhead de function pointers otimizado
  - Interface de layer padronizada
  - Zero overhead de performance verificado
- ⏳ **Limite de Tempo:** 1-2 dias

**Documentação de Planejamento:**
- `docs/GENERIC_FRAMEWORK_PLAN.md` - Plano completo de framework genérico

---

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

**Critérios Objetivos de Qualidade (FASE 5.1 - Pendente):**
- ⏳ **Testes:** 100% pass rate em todos os testes de layers básicas
- ⏳ **Precisão Numérica:** Max absolute difference < 1e-5, Max relative difference < 1e-4 (FP32)
- ⏳ **Validação:** Todas as layers validadas contra referências PyTorch
- ⏳ **Zero-Malloc:** Nenhuma alocação dinâmica no hot path (apenas arena)
- ⏳ **Performance:** Benchmarks mantidos ou melhorados
- ⏳ **Sanitizers:** AddressSanitizer, MemorySanitizer passam sem erros

**Checkpoint de Refatoração (Após FASE 5.1 - Pendente):**
- ⏳ **Status:** Pendente
- ⏳ **Áreas a Verificar:**
  - Consistência de interface de layers
  - Qualidade de implementação de layers
  - Otimização de performance
  - Cobertura de testes completa
- ⏳ **Limite de Tempo:** 1 dia

---

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

**Critérios Objetivos de Qualidade (FASE 5.2 - Pendente):**
- ⏳ **Testes:** 100% pass rate em todos os testes de layers avançadas
- ⏳ **Precisão Numérica:** Max absolute difference < 1e-5, Max relative difference < 1e-4 (FP32)
- ⏳ **Validação:** Todas as layers validadas contra referências PyTorch
- ⏳ **Zero-Malloc:** Nenhuma alocação dinâmica no hot path (apenas arena)
- ⏳ **Performance:** Benchmarks mantidos ou melhorados
- ⏳ **Sanitizers:** AddressSanitizer, MemorySanitizer passam sem erros

**Checkpoint de Refatoração (Após FASE 5.2 - Pendente):**
- ⏳ **Status:** Pendente
- ⏳ **Áreas a Verificar:**
  - Arquitetura de layers avançadas
  - Composição de layers
  - Otimização de performance
  - Testes de layers complexas
- ⏳ **Limite de Tempo:** 1-2 dias

---

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

**Dependências:**
- ✅ FASE 5.0 (Core Abstraction) - Deve estar completa antes de iniciar
- ✅ FASE 5.1 (Basic Layers) - Deve estar completa antes de iniciar
- ✅ FASE 5.2 (Advanced Layers) - Deve estar completa antes de iniciar

**Critérios Objetivos de Qualidade (FASE 5.3 - Pendente):**
- ⏳ **Testes:** 100% pass rate em todos os testes de exemplo
- ⏳ **Validação:** Modelos exemplo validados end-to-end
- ⏳ **Documentação:** Exemplos de uso completos e funcionais
- ⏳ **Performance:** Benchmarks mantidos ou melhorados

**Checkpoint de Refatoração (Após FASE 5.3 - Pendente):**
- ⏳ **Status:** Pendente
- ⏳ **Áreas a Verificar:**
  - Completude de migração
  - Compatibilidade reversa
  - Validação de performance
  - Limpeza de código
- ⏳ **Limite de Tempo:** 1 dia

---

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

**Critérios Objetivos de Qualidade (FASE 5.4 - Pendente):**
- ⏳ **Testes:** 100% pass rate em todos os testes de arquiteturas adicionais
- ⏳ **Validação:** Arquiteturas adicionais validadas end-to-end
- ⏳ **Documentação:** Exemplos de uso completos

**Checkpoint de Refatoração (Após FASE 5.4 - Pendente):**
- ⏳ **Status:** Pendente
- ⏳ **Áreas a Verificar:**
  - Prontidão para produção
  - Revisão final de qualidade de código
  - Validação final de performance
  - Completude de documentação
- ⏳ **Limite de Tempo:** 2 dias

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
  - **CUDA:** Usar `cudaMalloc` (não `cudaHostAlloc`) no hot path para manter zero-malloc
  - **Pinned Memory:** Apenas para buffers persistentes (Tier 2), não no hot path

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
  - ✅ Debug detalhado em `Q_VALIDATE_ALIGNED_OR_RETURN` para diagnóstico de problemas de alinhamento
  - ✅ Correção de alinhamento em softmax (buffers alinhados para cada linha)
- ✅ **Validação de Contiguidade:** Previne leitura de memória inválida em MatMul
  - ✅ Validação de `nb[0] == expected_stride` em `q_gemv_q4_f32_avx2`
  - ✅ Falha com erro claro se tensor não for contíguo (v1.0 limitation)
  - ✅ Documentação clara de limitação arquitetural
- ✅ **Validação de Tipo:** Previne uso incorreto de dados quantizados
- ✅ **Validação de Dimensões:** Previne acesso fora dos limites

### Testes Adversariais

**Status:** ✅ **COMPLETO** (2025-01-02)

Testes adversariais completos implementados para validar robustez do código:
- ✅ **19 testes adversariais** para `llama_forward()` (100% pass rate)
- ✅ **24 testes adversariais** para tokenizer (100% pass rate)
- ✅ Cobertura completa: NULL pointers, edge cases, memory safety, large sequences, misaligned memory, corrupted data, numerical stability
- ✅ Metodologia Lead SDET: Scenario Map, Acceptance Criteria, Blinded Implementation (AAA pattern)

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

### Performance Benchmarks (2025-01-02)

**Status Atual de Performance:**
- ✅ **Greedy Sampling:** ~100 ms/token (baseline perfeito)
- ✅ **Prefill:** ~26 ms/token (excelente)
- ✅ **Top-p=0.9:** ~532 ms/token (corrigido, ~11× melhoria de ~6000 ms)
- ⚠️ **Top-k=10:** ~616 ms/token (aceitável, complexidade correta O(V + k log k))
- ⚠️ **Top-k+Top-p:** ~1029 ms/token (aceitável, pode melhorar)

**Otimizações Críticas Implementadas:**
- ✅ **Top-p:** Eliminado memcpy repetido no binary search (sort UMA VEZ + cumsum prefixo)
- ✅ **Validação:** Auditoria completa de performance com correções implementadas
- ⚠️ **Top-k:** Otimizações recomendadas (SIMD init, renormalização otimizada)

**Documentação de Performance:**
- `docs/AUDITORIA_PERFORMANCE_COMPLETA.md` - Resumo consolidado
- `docs/src-docs/AUDIT_PERFORMANCE_TOP_P_CRITICAL.md` - Auditoria detalhada de top-p
- `docs/src-docs/AUDIT_PERFORMANCE_TOP_K.md` - Análise de top-k
- `docs/CORRECAO_TOP_P_IMPLEMENTADA.md` - Documentação da correção crítica

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

## 8. PRÓXIMOS PASSOS

### ✅ FASE 4.2 (Main Loop) - **COMPLETA**

**Status:** ✅ **COMPLETA** (2025-01-02)

**Objetivo:** Implementar loop principal de geração de texto.

**Dependências:** 
- ✅ FASE 4.1 (Tokenizer) - COMPLETA
- ✅ FASE 3.3 (Forward Pass) - COMPLETA

**Implementação:**
- ✅ Loop de geração completo (Tokenize → Forward → Sample → Print → Update Cache)
- ✅ Suporte a múltiplas estratégias de sampling (Greedy, Temperature, Top-k, Top-p)
- ✅ Benchmarks de performance implementados
- ✅ Auditoria de performance completa com correções críticas

**Próxima Fase Recomendada:**
- **FASE 2.6:** Training Kernels (Optimizers, Loss Functions, Gradient Clipping)
- **FASE 2.7:** CUDA Support (Google Colab / GPU)

### Implementação Futura: PARTE 2 - Capacidade de Treinamento

**Pré-requisito:** FASE 4.2 (Main Loop) deve estar completa antes de iniciar.

**Ordem de Implementação Recomendada:**

1. **FASE 2.6: Training Kernels**
   > **"Atue como Qorus-Architect. Vamos implementar a FASE 2.6. Comece com os Optimizers seguindo o planejamento completo em `docs/TRAINING_CAPABILITY_PLAN.md`. Use o framework MFR + CoT + Mathematical Proof + TDD conforme `docs/.cursorrules`."**
   - Optimizers (Adam, AdamW) - Base para treinamento
   - Loss Functions (MSE, CrossEntropy) - Necessário para backward
   - Gradient Clipping - Estabilização de treinamento

2. **FASE 2.7: CUDA Support** (Pode ser paralelo a FASE 2.6)
   - Abstração de device
   - Gerenciamento de memória GPU
   - Kernels CUDA

3. **FASE 3.4: Backward Pass**
   - Propagação de gradientes através das camadas

4. **FASE 3.5: Training Loop**
   - Loop completo de treinamento (epochs, mini-batches)

### Implementação Futura: PARTE 3 - Framework Genérico v3.0

**Pré-requisito:** PARTE 1 (Inferência) e PARTE 2 (Treinamento) devem estar completas antes de iniciar.

**Ordem de Implementação Recomendada:**

1. **FASE 5.0: Core Abstraction**
   > **"Atue como Qorus-Architect. Vamos implementar a FASE 5.0. Comece com a Generic Layer Interface seguindo o planejamento completo em `docs/GENERIC_FRAMEWORK_PLAN.md`. Use o framework MFR + CoT + Mathematical Proof + TDD conforme `docs/.cursorrules`."**
   - Generic Layer Interface (polimorfismo via function pointers)
   - Generic Model Container
   - Generic Forward/Backward Pass

2. **FASE 5.1: Basic Layers**
   - Linear Layer
   - Activation Layers (ReLU, GeLU, SiLU, Sigmoid)
   - Normalization Layers (RMSNorm, LayerNorm, BatchNorm)
   - Softmax Layer

3. **FASE 5.2: Advanced Layers**
   - Multi-Head Attention (MHA)
   - Feed-Forward Network (FFN)
   - Transformer Block
   - Embedding Layer

4. **FASE 5.3: Example Model Builders**
   - Transformer Model Builder usando API genérica
   - Exemplos de uso e documentação

5. **FASE 5.4: Additional Architectures** (Futuro)
   - Simple MLP
   - CNN Support
   - RNN/LSTM Support

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

**Documentos de Planejamento (FASE 2.7 - CUDA Support):**
- `docs/CUDA_ADAPTATION_PLAN.md` - **Plano completo de adaptação CUDA para Google Colab / GPU (MFR + CoT + Proof + TDD)** (a ser criado)

**Documentação de Qualidade:**
- `docs/REFACTORING_CHECKPOINTS.md` - **Procedimentos de checkpoint de refatoração e garantia de qualidade**

**Documentação Técnica:**
- `docs/STATUS.md` - Status detalhado do projeto
- `docs/QUICK_REFERENCE.md` - Referência rápida
- `docs/FASE_3.3_ANALYSIS.md` - Análise do forward pass
- `docs/TOKENIZER_IMPLEMENTATION.md` - **Documentação completa do tokenizer (FASE 4.1)**
- `docs/PRECISION_STANDARDS.md` - Padrões de precisão numérica
- `docs/ASYMPTOTIC_ANALYSIS.md` - Análise assintótica
- `docs/.cursorrules` - Metodologia de desenvolvimento (MFR + CoT + Proof + TDD)

**Documentação de Performance e Auditoria:**
- `docs/AUDITORIA_PERFORMANCE_COMPLETA.md` - **Resumo consolidado de auditoria de performance**
- `docs/src-docs/AUDIT_PERFORMANCE_TOP_P_CRITICAL.md` - **Auditoria detalhada de top-p (gargalo crítico corrigido)**
- `docs/src-docs/AUDIT_PERFORMANCE_TOP_K.md` - **Análise de top-k**
- `docs/CORRECAO_TOP_P_IMPLEMENTADA.md` - **Documentação da correção crítica de top-p**
- `docs/src-docs/INDEX_AUDITORIAS.md` - **Índice de todas as auditorias de performance**
