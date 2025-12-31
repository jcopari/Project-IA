# Code Agent - Documento de Implementação | Qorus-IA

**Data**: 2024-12-29  
**Versão**: Qorus-IA v2.0.0 (Reorganizado para Implementação)  
**Status**: 📋 Documento Prático de Implementação  
**Metodologia Core**: TDD + MFR + CoT + Proof (Integrado e Obrigatório)

---

## 📋 ÍNDICE

1. [Visão Geral e Contexto](#visão-geral-e-contexto)
2. [Metodologia Core: TDD + MFR + CoT + Proof](#metodologia-core-tdd--mfr--cot--proof)
3. [O que Já Existe vs O que Precisa ser Feito](#o-que-já-existe-vs-o-que-precisa-ser-feito)
4. [Roadmap Progressivo de Implementação](#roadmap-progressivo-de-implementação)
5. [Especificações Técnicas](#especificações-técnicas)
6. [Integração e Deploy](#integração-e-deploy)
7. [Referências e Checklist](#referências-e-checklist)

---

## 🎯 VISÃO GERAL E CONTEXO

### Propósito

**Qorus-IA Code Agent** é uma **ferramenta interna** de desenvolvimento que utiliza IA para gerar código de qualidade seguindo metodologia rigorosa (TDD + MFR + CoT + Proof).

**Características Principais:**
- ✅ **Ferramenta Interna** - Acesso restrito à equipe de desenvolvimento (não para clientes)
- ✅ **Multi-Linguagem** - Suporta todas linguagens do projeto (C, Python, JavaScript, TypeScript, PHP, SQL, etc)
- ✅ **Latência ultra-baixa** (inferência local no servidor: 10-50ms)
- ✅ **Privacidade total** (código nunca sai do servidor)
- ✅ **Código sempre testado** (TDD automático integrado)
- ✅ **Performance CPU** (157.79 GFLOPS sem GPU)

**Objetivo**: Acelerar desenvolvimento interno gerando código de qualidade que segue padrões do projeto e passa em testes automaticamente.

### Casos de Uso

- **Code Completion**: Autocompletar código enquanto você digita
- **Code Generation**: Gerar código a partir de descrições em linguagem natural
- **Code Refactoring**: Refatorar código existente seguindo instruções
- **Code Explanation**: Explicar código existente
- **Design-to-Code**: Gerar código frontend a partir de imagens de design (mockups, Figma)

### Arquitetura Simplificada

```
┌─────────────────────────────────────────────────────────┐
│              CLIENTE (PhpStorm ou Chat Web)            │
│              - LSP Client ou Interface Web             │
└───────────────────────┬─────────────────────────────────┘
                        │ SSH/HTTP
                        ↓
┌─────────────────────────────────────────────────────────┐
│              SERVIDOR (tempo-main)                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Chat Server (Node.js/TypeScript) - Opcional    │  │
│  └───────────────────┬──────────────────────────────┘  │
│                      ↕ IPC                              │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Qorus-IA Code Agent (C Native)                 │  │
│  │  - Tokenizer Multi-Linguagem                    │  │
│  │  - Transformer Decoder Stack                    │  │
│  │  - TDD + MFR + CoT + Proof integrado           │  │
│  │  - 157.79 GFLOPS                                │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 METODOLOGIA CORE: TDD + MFR + CoT + Proof

### ⚠️ IMPORTANTE: Metodologia Obrigatória

**TDD + MFR + CoT + Proof NÃO são opções** - são parte integrante do cerne do agente. Todo código gerado **DEVE** seguir este fluxo:

1. **MFR primeiro**: Modelo definido antes de qualquer código
2. **Proof obrigatório**: Validação matemática (complexidade, corretude, limites) antes da implementação
3. **TDD sempre**: Testes gerados antes da implementação
4. **CoT explícito**: Raciocínio documentado passo a passo
5. **Colaboração**: Desenvolvedor participa em cada fase

### Fluxo Core Integrado

```
┌─────────────────────────────────────────────────────────┐
│         FASE 0: CHAIN OF THOUGHT (CoT)                │
│         - Passo 1: Entender o problema                │
│         - Passo 2: Decompor em sub-problemas          │
│         - Passo 3: Identificar edge cases              │
│         - Passo 4: Verificar padrões existentes      │
└───────────────────────┬───────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────┐
│    FASE 0.5: MATHEMATICAL PROOF & COMPLEXITY ANALYSIS  │
│         - Time Complexity (Big O)                      │
│         - Space Complexity                             │
│         - Proof of Correctness (termination, bounds)  │
│         - Edge Case Proof (N=0, N=1, N=MAX)           │
│         - Numerical Stability (se aplicável)          │
└───────────────────────┬───────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────┐
│         FASE 1: MODEL-FIRST REASONING (MFR)            │
│         - Definir ENTITIES (estruturas de dados)      │
│         - Definir STATE VARIABLES (layout de memória) │
│         - Definir CONSTRAINTS (invariantes)           │
│         - Definir ACTIONS (protótipos de funções)      │
└───────────────────────┬───────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────┐
│         FASE 2: TEST-DRIVEN DESIGN (TDD)               │
│         - RED: Gerar testes primeiro                   │
│         - Definir casos de teste (unit + integration) │
│         - Validar testes compilam (mas falham)        │
└───────────────────────┬───────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────┐
│         FASE 3: IMPLEMENTAÇÃO (GREEN)                  │
│         - Gerar código baseado no modelo MFR          │
│         - Seguir raciocínio CoT                        │
│         - Respeitar provas matemáticas (Phase 0.5)    │
│         - Implementar para passar nos testes          │
└───────────────────────┬───────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────┐
│         FASE 4: VALIDAÇÃO E TESTES                     │
│         - Executar testes unitários                    │
│         - Executar testes de integração                │
│         - Validar contra modelo MFR                    │
│         - Verificar provas matemáticas                 │
└───────────────────────┬───────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────┐
│         FASE 5: REFINAMENTO (REFACTOR)                 │
│         - Se testes falharem: corrigir código          │
│         - Se modelo violado: ajustar implementação    │
│         - Se prova violada: revisar algoritmo          │
│         - Iterar até aprovação do desenvolvedor       │
└─────────────────────────────────────────────────────────┘
```

### Detalhamento da Fase 0.5: Mathematical Proof & Complexity Analysis

**CRITICAL:** Before defining the model, the agent must mathematically validate the proposed solution. **NO GUESSING ALLOWED.**

The Mathematical Proof phase must include:

1. **TIME COMPLEXITY (Big O):** Calculate theoretical Time Complexity. Is this optimal? Why?
   - Example: "Proposed solution is O(N). Naive is O(N^2). This is optimal because we visit each element once."
   - **Cache Complexity:** For data-oriented operations, analyze cache behavior (spatial/temporal locality).
   - **SIMD Efficiency:** For vectorized operations, prove that SIMD lanes are fully utilized (e.g., 8 floats per AVX2 register).

2. **SPACE COMPLEXITY:** Calculate memory overhead (auxiliary space, not input size).
   - Distinguish between in-place operations (O(1)) and operations requiring temporary buffers.
   - Document peak memory usage for multi-stage algorithms.

3. **PROOF OF CORRECTNESS:**
   - **Termination:** Prove the loop/recursion will finish (decreasing variant, bounded iteration).
   - **Bounds:** Prove that indices `i` will strictly stay within `0 <= i < size` (no buffer overflows).
   - **Arithmetic:** Prove that operations (e.g., `a + b`) will not overflow/underflow for expected types, or how overflow is handled.
   - **Alignment:** Prove that SIMD operations access aligned memory (e.g., `ptr % 32 == 0` for AVX2, `ptr % 64 == 0` for AVX-512).

4. **EDGE CASE PROOF:** Mathematically demonstrate behavior at boundaries:
   - **N=0:** Loop condition fails immediately, returns safe state.
   - **N=1:** Single-element case handled correctly.
   - **N=MAX:** No integer overflow in loop counters or array indices.
   - **Special Values:** NaN, Inf, denormals propagate correctly (for floating-point).

5. **NUMERICAL STABILITY (for floating-point operations):**
   - Prove that operations maintain numerical precision (e.g., Kahan summation for reductions).
   - Document any approximations or trade-offs (e.g., fast approximations vs. exact computation).
   - Prove that rounding errors accumulate within acceptable bounds.

6. **TRIVIAL PROOF SHORTCUT:**
   - For obviously correct operations (e.g., element-wise addition), a brief statement suffices:
     *"Trivial: O(N) time, O(1) space. Bounds: `i < n` guarantees termination and safety. Alignment: Precondition ensures 64-byte alignment."*

### Implementação do Fluxo

```c
// Estrutura para fluxo completo TDD + MFR + CoT + Proof
typedef struct s_tdd_mfr_cot_proof_flow {
    // Fase 0: CoT
    t_cot_reasoning *reasoning;
    bool reasoning_approved;
    
    // Fase 0.5: Mathematical Proof
    t_mathematical_proof *proof;
    bool proof_validated;
    
    // Fase 1: MFR
    t_proposed_model *model;
    bool model_approved;
    
    // Fase 2: TDD
    t_proposed_tests *tests;
    bool tests_approved;
    
    // Fase 3: Implementação
    char *generated_code;
    
    // Fase 4: Validação
    t_test_results *test_results;
    bool all_tests_passing;
    
    // Fase 5: Refinamento
    uint32_t iteration_count;
    bool code_approved;
} t_tdd_mfr_cot_proof_flow;

// Executar fluxo completo
int execute_tdd_mfr_cot_proof_flow(t_tdd_mfr_cot_proof_flow *flow,
                                    const char *requirement,
                                    const char *language,
                                    t_developer_feedback *feedback) {
    // FASE 0: CoT - Raciocínio Passo a Passo
    flow->reasoning = agent_generate_reasoning(requirement, language);
    flow->reasoning_approved = true;
    
    // FASE 0.5: Mathematical Proof - Validação Matemática OBRIGATÓRIA
    flow->proof = agent_generate_proof(requirement,
                                        flow->reasoning,
                                        language);
    if (!flow->proof || !validate_proof(flow->proof)) {
        return -1; // Proof inválido - não prosseguir
    }
    flow->proof_validated = true;
    
    // FASE 1: MFR - Definir Modelo baseado em Proof validado
    flow->model = agent_propose_model(requirement,
                                      language,
                                      flow->proof);
    flow->model_approved = true;
    
    // FASE 2: TDD - Gerar Testes PRIMEIRO
    flow->tests = agent_propose_tests(flow->model,
                                      requirement,
                                      flow->proof);
    flow->tests_approved = true;
    
    // FASE 3: Implementação - Código que respeita Proof e passa nos testes
    flow->generated_code = agent_generate_code(flow->model,
                                               flow->tests,
                                               flow->reasoning,
                                               flow->proof);
    
    // FASE 4: Validação - Executar testes e verificar Proof
    flow->test_results = agent_run_tests(flow->generated_code, flow->tests);
    flow->all_tests_passing = (flow->test_results->failures == 0);
    
    bool proof_respected = verify_code_against_proof(flow->generated_code,
                                                     flow->proof);
    
    // FASE 5: Refinamento iterativo se necessário
    while ((!flow->all_tests_passing || !proof_respected) &&
           flow->iteration_count < MAX_ITERATIONS) {
        if (!flow->all_tests_passing) {
            flow->generated_code = agent_fix_failing_tests(flow->generated_code,
                                                           flow->test_results,
                                                           flow->model,
                                                           flow->proof);
        }
        if (!proof_respected) {
            flow->generated_code = agent_fix_proof_violations(flow->generated_code,
                                                              flow->proof,
                                                              flow->model);
        }
        flow->test_results = agent_run_tests(flow->generated_code, flow->tests);
        flow->all_tests_passing = (flow->test_results->failures == 0);
        proof_respected = verify_code_against_proof(flow->generated_code,
                                                     flow->proof);
        flow->iteration_count++;
    }
    
    return (flow->all_tests_passing && proof_respected) ? 0 : -1;
}
```

### Critérios de "Done"

```c
typedef struct s_done_criteria {
    bool compiles_successfully;    // Código compila sem erros
    bool all_tests_passing;        // Todos os testes passam
    bool proof_validated;          // Proof matemático é respeitado
    bool model_validated;          // Modelo MFR é respeitado
    bool user_approved;            // Usuário aprova explicitamente
    bool syntax_valid;             // Sintaxe válida
    uint32_t iteration_count;      // Número de iterações
} t_done_criteria;

bool is_code_done(t_done_criteria *criteria) {
    return criteria->compiles_successfully &&
           criteria->all_tests_passing &&
           criteria->proof_validated &&
           criteria->model_validated &&
           criteria->user_approved &&
           criteria->syntax_valid &&
           criteria->iteration_count < MAX_ITERATIONS;
}
```

### Recuperação de Falhas (Self-Healing)

```c
typedef enum {
    ERROR_COMPILATION,
    ERROR_TEST_FAILURE,
    ERROR_PROOF_VIOLATION,
    ERROR_MODEL_VIOLATION,
    ERROR_SYNTAX,
    ERROR_TIMEOUT
} t_error_type;

typedef enum {
    RECOVERY_REGENERATE_BLOCK,
    RECOVERY_REGENERATE_FUNCTION,
    RECOVERY_REFINE_INCREMENTAL,
    RECOVERY_FALLBACK_SIMPLE,
    RECOVERY_ASK_USER
} t_recovery_strategy;

// Analisar erro e propor estratégia
t_error_recovery *analyze_error(const char *error_output,
                                 const char *generated_code,
                                 t_test_results *test_results) {
    t_error_recovery *recovery = calloc(1, sizeof(t_error_recovery));
    
    if (strstr(error_output, "error:")) {
        recovery->error_type = ERROR_COMPILATION;
        recovery->strategy = RECOVERY_REGENERATE_BLOCK;
    } else if (test_results && test_results->failures > 0) {
        recovery->error_type = ERROR_TEST_FAILURE;
        recovery->strategy = RECOVERY_REFINE_INCREMENTAL;
    } else {
        recovery->strategy = RECOVERY_ASK_USER;
    }
    
    return recovery;
}
```

---

## ✅ O QUE JÁ EXISTE VS O QUE PRECISA SER FEITO

### ✅ O que Já Existe no Qorus-IA

- ✅ **Transformer Block completo** (MHA + FFN + LayerNorm)
- ✅ **RoPE** (Rotary Positional Embeddings)
- ✅ **Causal Masking** (necessário para geração autoregressiva)
- ✅ **Optimizers** (Adam, AdamW)
- ✅ **Loss functions** (CrossEntropy)
- ✅ **Performance otimizada** (157.79 GFLOPS)
- ✅ **Memory management** (64-byte aligned)
- ✅ **Thread-safe operations** (OpenMP compatible)

### ❌ O que Precisa ser Implementado

#### Fase 1: Base LLM (Crítico - Bloqueador)

| Componente | Horas | Prioridade | Descrição |
|------------|-------|------------|-----------|
| **Tokenizer Multi-Linguagem** | 16-22h | 🔴 Crítica | BPE/SentencePiece para C, Python, JS, TS, PHP, SQL |
| **Embedding Layer** | 3-4h | 🔴 Crítica | Token embeddings + integração RoPE |
| **Decoder Stack** | 4-6h | 🔴 Crítica | Empilhar Transformer Blocks (12-24 layers) |
| **LM Head** | 2-3h | 🔴 Crítica | Projeção final (embed_dim → vocab_size) |
| **Generation Loop** | 8-10h | 🔴 Crítica | Loop autoregressivo + sampling |
| **Subtotal** | **33-45h** | | |

#### Fase 2: Especialização Multi-Linguagem

| Componente | Horas | Prioridade | Descrição |
|------------|-------|------------|-----------|
| **Syntax-Aware Generation** | 4-6h | 🟡 Alta | Validação de sintaxe durante geração |
| **Context Manager** | 2-3h | 🟡 Alta | Extração de contexto do cursor |
| **Multi-file Context** | 2-3h | 🟡 Alta | Suporte a múltiplos arquivos |
| **Subtotal** | **8-12h** | | |

#### Fase 3: TDD + MFR + CoT + Proof Core (Crítico - Diferencial)

| Componente | Horas | Prioridade | Descrição |
|------------|-------|------------|-----------|
| **Templates de Prompt** | 8-12h | 🔴 Crítica | build_mfr_prompt, build_tdd_prompt, build_cot_prompt, build_proof_prompt |
| **Geração Automática de Proofs** | 10-14h | 🔴 Crítica | agent_generate_proof() - complexidade, corretude, limites |
| **Geração Automática de Testes** | 12-16h | 🔴 Crítica | agent_propose_tests() baseado em modelo MFR e proof |
| **Execução e Validação** | 8-10h | 🔴 Crítica | agent_run_tests(), validate_against_model(), verify_proof() |
| **Refinamento Iterativo** | 6-8h | 🔴 Crítica | agent_fix_failing_tests(), agent_fix_proof_violations(), refine_code() |
| **Integração Core** | 4-6h | 🔴 Crítica | execute_tdd_mfr_cot_proof_flow() completo |
| **Subtotal** | **48-66h** | | |

#### Fase 4: Funcionalidades Básicas

| Componente | Horas | Prioridade | Descrição |
|------------|-------|------------|-----------|
| **Code Completion** | 8-10h | 🟡 Média | Autocomplete baseado em contexto |
| **Code Generation** | 6-8h | 🟡 Média | Geração a partir de descrição |
| **Code Refactoring** | 8-10h | 🟡 Média | Refatoração guiada |
| **Code Explanation** | 4-6h | 🟡 Média | Explicação de código |
| **Subtotal** | **26-34h** | | |

#### Fase 5: Design-to-Code (Opcional - Feature Diferencial)

| Componente | Horas | Prioridade | Descrição |
|------------|-------|------------|-----------|
| **Vision Processing** | 20-30h | 🟡 Alta | Carregamento, preprocessamento, ViT/CNN |
| **Design Analysis** | 12-18h | 🟡 Alta | Detecção componentes, layout, cores |
| **Code Generation Frontend** | 8-12h | 🟡 Alta | Geração React/Vue com TDD+MFR+CoT |
| **Visual Validation** | 8-12h | 🟡 Alta | Renderização, comparação visual |
| **Subtotal** | **48-72h** | | |

#### Fase 6: Integração

| Componente | Horas | Prioridade | Descrição |
|------------|-------|------------|-----------|
| **LSP Server** | 12-16h | 🟡 Alta | Language Server Protocol completo |
| **Chat Interno** | 20-30h | 🟡 Média | Interface web simples para equipe |
| **Subtotal** | **32-46h** | | |

### Resumo Executivo

| Fase | Horas | Prioridade | Bloqueador? |
|------|-------|------------|-------------|
| **Base LLM** | 33-45h | 🔴 Crítica | ✅ Sim |
| **Especialização** | 8-12h | 🟡 Alta | ✅ Sim |
| **TDD+MFR+CoT+Proof Core** | 48-66h | 🔴 Crítica | ✅ Sim |
| **Funcionalidades** | 26-34h | 🟡 Média | ❌ Não |
| **Design-to-Code** | 48-72h | 🟡 Alta | ❌ Não |
| **Integração** | 32-46h | 🟡 Alta | ⚠️ Parcial |
| **TOTAL** | **195-275h** | | |

**MVP Mínimo**: 89-123h (Base LLM + Especialização + TDD+MFR+CoT+Proof básico)  
**Produto Completo**: 133-188h (MVP + Funcionalidades + Integração)  
**Produto Premium**: 195-275h (Todos os componentes)

---

## 🛣️ ROADMAP PROGRESSIVO DE IMPLEMENTAÇÃO

### Princípio: Implementar do Mais Fácil para o Mais Difícil

Seguindo a filosofia MFR + Proof + TDD + CoT, implementamos em ordem crescente de complexidade:

### Etapa 1: Base LLM (33-45h) - FUNDAÇÃO

**Objetivo**: Ter um LLM funcional básico capaz de gerar texto/código.

#### 1.1 Tokenizer Multi-Linguagem (16-22h)

**Estratégia Pragmática:**
1. **Referência e Análise** (2-3h): Estudar `llama.cpp`, `tiktoken`, `sentencepiece`
2. **Porte Inicial** (Opcional - 4-6h): Portar implementação de referência para validação
3. **Reimplementação Otimizada** (10-13h): Reescrever do zero adaptado ao Qorus-IA

**Estrutura de Dados:**

```c
typedef struct s_code_tokenizer {
    char **vocab;                    // Vocabulário (~50k-100k tokens)
    uint32_t vocab_size;
    uint32_t *bpe_merges;           // Regras BPE
    char **supported_languages;     // ["c", "python", "javascript", "typescript", "php", "sql"]
    uint32_t num_languages;
    void *lookup_cache;             // Cache 64-byte aligned
    bool use_avx2;
    bool use_avx512;
} t_code_tokenizer;

// API
t_code_tokenizer *code_tokenizer_create_multi(const char *vocab_path,
                                              const char **languages,
                                              uint32_t num_languages);

uint32_t *code_tokenizer_encode(t_code_tokenizer *tok, 
                                 const char *code,
                                 const char *language,
                                 uint32_t *out_len);

char *code_tokenizer_decode(t_code_tokenizer *tok,
                            const uint32_t *tokens,
                            uint32_t len,
                            const char *language);
```

#### 1.2 Embedding Layer (3-4h)

```c
typedef struct s_embedding {
    t_tensor *weight;       // [vocab_size, embed_dim]
    uint32_t vocab_size;
    uint32_t embed_dim;
} t_embedding;

t_embedding *embedding_create(uint32_t vocab_size, uint32_t embed_dim);
t_tensor *embedding_forward(t_embedding *emb, 
                            const uint32_t *token_ids, 
                            uint32_t batch_size,
                            uint32_t seq_len);
```

#### 1.3 Decoder Stack (4-6h)

```c
typedef struct s_decoder_stack {
    t_transformer_block **blocks;  // Array de N blocks
    uint32_t num_layers;
    t_layer_layernorm *final_norm;
    uint32_t embed_dim;
    t_rope_cache *rope_cache;
} t_decoder_stack;

t_decoder_stack *decoder_stack_create(uint32_t num_layers,
                                      uint32_t embed_dim,
                                      uint32_t num_heads,
                                      uint32_t hidden_dim,
                                      t_rope_cache *rope_cache,
                                      float dropout_p);
```

#### 1.4 LM Head (2-3h)

```c
typedef struct s_lm_head {
    t_layer_linear *proj;   // [embed_dim, vocab_size]
    bool weight_tied;
    t_embedding *tied_embedding;
} t_lm_head;

t_lm_head *lm_head_create(uint32_t embed_dim, 
                          uint32_t vocab_size,
                          bool weight_tied);
```

#### 1.5 Generation Loop (8-10h)

```c
typedef struct s_code_generation_config {
    uint32_t max_new_tokens;
    float temperature;
    uint32_t top_k;
    float top_p;
    bool syntax_check;
    const char *language;
} t_code_generation_config;

uint32_t *code_agent_generate(t_model *llm,
                                const char *prompt_code,
                                const char *context_code,
                                t_code_generation_config *config,
                                uint32_t *out_len);
```

### Etapa 2: Especialização Multi-Linguagem (8-12h)

**Objetivo**: Adaptar LLM para código multi-linguagem.

- Syntax-aware generation (4-6h)
- Context Manager (2-3h)
- Multi-file Context (2-3h)

### Etapa 3: TDD + MFR + CoT + Proof Core (48-66h) - DIFERENCIAL ÚNICO

**Objetivo**: Implementar metodologia core que gera código sempre testado e matematicamente validado.

#### 3.1 Templates de Prompt (8-12h)

```c
char *build_cot_prompt(const char *description, const char *language);
char *build_proof_prompt(const char *description, t_cot_reasoning *reasoning);
char *build_mfr_prompt(const char *description, const char *language, t_mathematical_proof *proof);
char *build_tdd_prompt(const char *description, t_proposed_model *model, t_mathematical_proof *proof);
char *build_impl_prompt(const char *description,
                        const char *data_model,
                        const char *reasoning_steps,
                        const char *proof_details,
                        const char *language);
```

#### 3.2 Geração Automática de Proofs (10-14h)

```c
typedef struct s_mathematical_proof {
    char *time_complexity;        // "O(N)", "O(N log N)", etc.
    char *space_complexity;        // "O(1)", "O(N)", etc.
    char *termination_proof;       // Prova de terminação
    char *bounds_proof;            // Prova de limites (0 <= i < n)
    char *edge_cases;              // N=0, N=1, N=MAX
    char *numerical_stability;     // Estabilidade numérica (se aplicável)
    bool is_trivial;              // Se é prova trivial
} t_mathematical_proof;

t_mathematical_proof *agent_generate_proof(const char *description,
                                            t_cot_reasoning *reasoning,
                                            const char *language);

bool validate_proof(t_mathematical_proof *proof);
bool verify_code_against_proof(const char *code, t_mathematical_proof *proof);
```

#### 3.3 Geração Automática de Testes (12-16h)

```c
t_proposed_tests *agent_propose_tests(t_proposed_model *model,
                                      t_mathematical_proof *proof,
                                      const char *description);
```

#### 3.4 Execução e Validação (8-10h)

```c
t_test_results *agent_run_tests(const char *code,
                                 t_proposed_tests *tests);

bool validate_against_model(const char *code, const char *data_model);
bool verify_code_against_proof(const char *code, t_mathematical_proof *proof);
```

#### 3.5 Refinamento Iterativo (6-8h)

```c
char *agent_fix_failing_tests(const char *code,
                               t_test_results *results,
                               t_proposed_model *model,
                               t_mathematical_proof *proof);

char *agent_fix_proof_violations(const char *code,
                                  t_mathematical_proof *proof,
                                  t_proposed_model *model);

char *refine_code(t_code_agent *agent,
                  const char *current_code,
                  const char *data_model,
                  const char *reasoning_steps,
                  const char *proof_details);
```

#### 3.6 Integração Core (4-6h)

```c
int execute_tdd_mfr_cot_proof_flow(t_tdd_mfr_cot_proof_flow *flow,
                                    const char *requirement,
                                    const char *language,
                                    t_developer_feedback *feedback);
```

### Etapa 4: Funcionalidades Básicas (26-34h)

- Code Completion (8-10h)
- Code Generation (6-8h)
- Code Refactoring (8-10h)
- Code Explanation (4-6h)

### Etapa 5: Design-to-Code (48-72h) - Opcional

- Vision Processing (20-30h)
- Design Analysis (12-18h)
- Code Generation Frontend (8-12h)
- Visual Validation (8-12h)

### Etapa 6: Integração (32-46h)

- LSP Server (12-16h)
- Chat Interno (20-30h)

---

## 📐 ESPECIFICAÇÕES TÉCNICAS

### Convenções de Código Qorus-IA

#### Naming Conventions

```c
// Prefixo obrigatório: ft_ para todas funções
int ft_function_name(const t_tensor *input, t_tensor *output);

// Naming: snake_case sempre
typedef struct s_struct_name {
    // campos
} t_struct_name;

// Constantes: UPPER_SNAKE_CASE
#define MAX_DIMS 8
```

#### Error Handling

```c
// Sempre retornar int: 0 = sucesso, negativo = erro
int ft_function(const t_tensor *input, t_tensor *output) {
    if (!input || !output) {
        return -1;
    }
    return 0;
}
```

#### Memory Management

```c
// TODOS tensores devem ser 64-byte aligned
t_tensor *ft_tensor_create(uint32_t *shape, uint32_t ndim) {
    t_tensor *t = calloc(1, sizeof(t_tensor));
    size_t size = calculate_size(shape, ndim);
    void *data = NULL;
    if (posix_memalign(&data, 64, size * sizeof(float)) != 0) {
        free(t);
        return NULL;
    }
    // ...
}
```

#### Thread Safety

```c
// NUNCA usar estado global mutável
// SEMPRE usar parâmetros para estado
int ft_function_with_state(t_state *state, const t_tensor *input);
```

### Padrões de Teste (TDD)

```c
#include "tensor.h"
#include "tests/common/test_utils.h"

static int test_suite_basic(void)
{
    int failures = 0;
    test_suite_start("Basic Functionality");
    
    {
        t_tensor *input = tensor_create((uint32_t[]){4}, 1);
        t_tensor *output = tensor_create((uint32_t[]){4}, 1);
        
        int ret = ft_function(input, output);
        bool passed = (ret == 0) && (output->data[0] == expected_value);
        
        failures += test_result("Basic case", passed, NULL);
        
        tensor_free(input);
        tensor_free(output);
    }
    
    test_suite_end(failures);
    return failures;
}

int main(void)
{
    double start = get_time_ms();
    int failures = 0;
    
    print_test_header("Component Test Suite");
    failures += test_suite_basic();
    
    double elapsed = get_time_ms() - start;
    print_test_footer(0, failures, elapsed);
    
    return failures > 0 ? 1 : 0;
}
```

### Estruturas de Dados Padrão

```c
#define MAX_DIMS 8

typedef struct s_tensor {
    float *data;                    // Dados (64-byte aligned)
    uint32_t shape[MAX_DIMS];
    uint32_t ndim;
    size_t size;
    size_t strides[MAX_DIMS];
    bool is_view;
    struct s_tensor *view_source;
} t_tensor;

typedef struct s_code_agent {
    t_model *llm_model;
    t_code_tokenizer *tokenizer;
    t_embedding *embeddings;
    t_decoder_stack *decoder;
    t_lm_head *lm_head;
    t_tdd_mfr_cot_proof_flow *core_flow;
    t_code_generation_config default_config;
} t_code_agent;
```

### Constraints e Invariantes

**Memory Constraints:**
- ✅ Todos tensors devem ser 64-byte aligned
- ✅ Nenhuma alocação dentro de loops (hot path)
- ✅ Zero-copy quando possível (views)
- ✅ Sempre liberar memória alocada

**Numerical Constraints:**
- ✅ Validação científica contra Python/PyTorch
- ✅ Tolerância híbrida (absolute + relative error)
- ✅ Tratamento de NaN/Inf

**Thread Safety Constraints:**
- ✅ Sem estado global mutável
- ✅ Funções thread-safe (OpenMP compatible)
- ✅ Usar `restrict` quando seguro

---

## 🔌 INTEGRAÇÃO E DEPLOY

### Arquitetura: PhpStorm + LSP Server Direto via SSH

**Decisão**: PhpStorm como editor principal devido a SSH nativo, LSP completo e Remote Development.

**Arquitetura:**

```
PhpStorm (Cliente)
    ↕ SSH Tunnel (automático)
Servidor (tempo-main)
    ↕ IPC (stdio)
Qorus-IA LSP Server (C Native)
```

### LSP Server Implementation

```c
// src/lsp/ft_lsp_server.c
#include "tensor.h"
#include "code_agent.h"

typedef struct s_lsp_server {
    t_code_agent *agent;
    FILE *stdin;
    FILE *stdout;
    bool initialized;
} t_lsp_server;

int lsp_stdio_mode(void) {
    t_code_agent *agent = code_agent_create("models/code_model.mia", 
                                            "vocabs/code_vocab.json");
    t_lsp_server server = {
        .agent = agent,
        .stdin = stdin,
        .stdout = stdout,
        .initialized = false
    };
    
    char buffer[8192];
    while (fgets(buffer, sizeof(buffer), stdin)) {
        json_t *request = json_parse(buffer);
        json_t *response = process_lsp_request(&server, request);
        char *response_str = json_stringify(response);
        fprintf(stdout, "%s\n", response_str);
        fflush(stdout);
        free(response_str);
    }
    
    code_agent_free(agent);
    return 0;
}

int main(int argc, char **argv) {
    return lsp_stdio_mode();
}
```

### Chat Interno Simplificado

**Estrutura no tempo-main:**

```
tempo-main/src/modules/code-agent-nd/
├── controllers/ChatController.ts
├── services/CodeAgentService.ts
├── websocket/ChatWebSocket.ts
└── views/chat.html
```

**Comando simples:**

```bash
# /usr/local/bin/qorus-ia-chat
#!/bin/bash
cd /path/to/tempo-main
npm run start:chat &
sleep 2
xdg-open "http://localhost:3000/code-agent/chat"
```

**Uso:**

```bash
qorus-ia-chat  # Abre interface web automaticamente
```

### Deployment

```bash
# Compilar LSP Server
cd /path/to/qorus-ia
make lsp-server

# Instalar
sudo cp build/qorus-ia-lsp /usr/local/bin/
sudo chmod +x /usr/local/bin/qorus-ia-lsp

# Estrutura recomendada
/opt/qorus-ia/
├── bin/qorus-ia-lsp
├── models/code_model.mia
└── vocabs/code_vocab.json
```

---

## 📚 REFERÊNCIAS E CHECKLIST

### Referências Técnicas

- **GPT-2**: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- **Codex**: "Evaluating Large Language Models Trained on Code" (Chen et al., 2021)
- **llama.cpp**: Implementação C++ otimizada
- **tiktoken**: BPE robusto (OpenAI)
- **LSP**: Language Server Protocol

### Checklist de Implementação

#### Fase 1: Base LLM
- [ ] Tokenizer multi-linguagem (16-22h)
- [ ] Embedding Layer (3-4h)
- [ ] Decoder Stack (4-6h)
- [ ] LM Head (2-3h)
- [ ] Generation Loop (8-10h)

#### Fase 2: Especialização
- [ ] Syntax-aware generation (4-6h)
- [ ] Context Manager (2-3h)
- [ ] Multi-file Context (2-3h)

#### Fase 3: TDD + MFR + CoT + Proof Core
- [ ] Templates de prompt (8-12h)
- [ ] Geração automática de proofs (10-14h)
- [ ] Geração automática de testes (12-16h)
- [ ] Execução e validação (8-10h)
- [ ] Refinamento iterativo (6-8h)
- [ ] Integração core (4-6h)

#### Fase 4: Funcionalidades
- [ ] Code Completion (8-10h)
- [ ] Code Generation (6-8h)
- [ ] Code Refactoring (8-10h)
- [ ] Code Explanation (4-6h)

#### Fase 5: Design-to-Code (Opcional)
- [ ] Vision Processing (20-30h)
- [ ] Design Analysis (12-18h)
- [ ] Code Generation Frontend (8-12h)
- [ ] Visual Validation (8-12h)

#### Fase 6: Integração
- [ ] LSP Server (12-16h)
- [ ] Chat Interno (20-30h)

---

## 🎓 CONCLUSÃO

**Status Atual**: ~30-40% completo (especificação)

**MVP Funcional**: 89-123 horas (~2-3 semanas full-time)
- Base LLM + Especialização + TDD+MFR+CoT+Proof básico

**Produto Completo**: 133-188 horas (~3-4 semanas full-time)
- MVP + Funcionalidades + Integração

**Produto Premium**: 195-275 horas (~5-6 semanas full-time)
- Todos os componentes incluindo Design-to-Code

**Bloqueadores Críticos:**
1. Base LLM (33-45h) - Sem isso, nada funciona
2. TDD+MFR+CoT+Proof Core (48-66h) - Diferencial competitivo único

**Recomendação**: Focar no MVP primeiro (89-123h), depois expandir progressivamente.

---

**Última Atualização**: 2024-12-29  
**Versão**: v2.1.0 (Reorganizado para Implementação Prática + Mathematical Proof)  
**Metodologia Core**: TDD + MFR + CoT + Proof (Integrado e Obrigatório)
