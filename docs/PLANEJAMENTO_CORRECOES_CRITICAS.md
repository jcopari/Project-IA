# 📋 PLANEJAMENTO: Correções Críticas de Performance

**Data:** 2025-01-02  
**Metodologia:** Protocolo de Planejamento de Engenharia (Critical Path)  
**Baseado em:** Auditorias acumuladas + Code Reviewer V2  
**Prioridade:** CRÍTICA - Correções de complexidade algorítmica e otimizações de pipeline

---

## FASE 1: Decomposição por Primeiros Princípios (First Principles)

### 1.1 Restrições Físicas Reais

**BPE Tokenizer (`src/tokenizer/bpe.c`):**
- **Restrição:** Largura de banda de memória para `memmove()` repetido
- **Latência:** `memmove()` de n elementos ≈ 10-50 ciclos/elemento (depende do tamanho)
- **Cache:** Movimentação de dados destrói localidade de cache
- **I/O Bound:** Operações de memória são o gargalo, não CPU

**Arena Allocator (`src/core/memory.c`):**
- **Restrição:** Dependência de dados no pipeline (stall de 4-5 ciclos)
- **Latência:** Load de `ctx->scratch_buffer` + `ctx->scratch_head` + add + modulo = ~5 ciclos
- **Branch Prediction:** Validação de alinhamento ocupa slot na BTB
- **CPU Bound:** Hot path executado milhões de vezes por inferência

**MatMul AVX2 (`src/ops/avx2/matmul_fp32.c`):**
- **Restrição:** Largura de banda de memória L1/L2
- **Latência:** Hardware prefetchers modernos são mais eficientes que prefetch manual
- **Cache:** Prefetch manual pode expulsar dados úteis da L1
- **I/O Bound:** Acesso sequencial já é otimizado por HW prefetchers

**RoPE (`src/ops/avx2/rope.c`):**
- **Restrição:** Contrato implícito de layout de memória
- **Latência:** Zero overhead em RELEASE (validação apenas DEBUG)
- **Segurança:** Bug silencioso se contrato violado (corrupção de inferência)

### 1.2 O que é Matematicamente Necessário

**BPE Tokenizer:**
- **Álgebra:** Aplicar merges greedy sem mover memória
- **Lógica:** Estado de token = VIVO ou MORTO (UINT32_MAX)
- **Invariante:** Array compactado apenas quando densidade de buracos > 50%

**Arena Allocator:**
- **Álgebra:** Invariante matemática: `scratch_head` sempre múltiplo de `Q_ALIGN`
- **Lógica:** `scratch_head = 0` (base) e `scratch_head += Q_ALIGN_SIZE(size)` (indução)
- **Prova:** Se `head` é múltiplo de 64 e `size` é arredondado para múltiplo de 64, então `head + size` é múltiplo de 64

**MatMul AVX2:**
- **Álgebra:** Remover prefetch manual (redundante com HW prefetchers)
- **Lógica:** Hardware prefetchers detectam padrões sequenciais automaticamente

**RoPE:**
- **Álgebra:** Validar layout duplicado: `cos[i] == cos[i+1]` para todo `i` par
- **Lógica:** Contrato explícito via assertions DEBUG

### 1.3 Custo Mínimo Teórico (Lower Bound)

**BPE Tokenizer:**
- **Lower Bound:** O(m × n) onde m = merges, n = tokens
- **Atual:** O(m × n³) (catastrófico)
- **Proposto:** O(m × n) + O(n) compactação = O(m × n) ✅

**Arena Allocator:**
- **Lower Bound:** O(1) com ~2 ciclos (aritmética + load)
- **Atual:** O(1) com ~6.5 ciclos (validações + dependências)
- **Proposto:** O(1) com ~2 ciclos (invariante + `__builtin_assume_aligned`) ✅

**MatMul AVX2:**
- **Lower Bound:** Latência de memória determinada por HW prefetcher
- **Atual:** Latência + overhead de prefetch manual (~1-5%)
- **Proposto:** Latência pura (sem overhead) ✅

**RoPE:**
- **Lower Bound:** Zero overhead em RELEASE
- **Atual:** Zero overhead (sem validação)
- **Proposto:** Zero overhead em RELEASE + validação DEBUG ✅

### 1.4 Critérios de Parada (Threshold)

**Threshold Assintótico:**
- **BPE:** Solução proposta O(m × n) ≤ Lower Bound O(m × n) × 1.1 ✅
- **Arena:** Solução proposta O(1) ≤ Lower Bound O(1) × 1.1 ✅
- **MatMul:** Solução proposta = Lower Bound ✅
- **RoPE:** Solução proposta = Lower Bound ✅

**Threshold Constante:**
- **BPE:** Fatores constantes medidos ≤ 2x do teórico (após implementação)
- **Arena:** Fatores constantes ≤ 2x do teórico (~2 ciclos vs ~1 ciclo teórico)
- **MatMul:** Remoção de overhead de prefetch manual
- **RoPE:** Zero overhead em RELEASE

**Iteração Máxima:** 3 iterações para convergência

---

## FASE 2: Model-First Reasoning (Estrutura do Problema)

### 2.1 Entidades e Estruturas de Dados

**BPE Tokenizer:**
```c
// Estado de token: VIVO (uint32_t válido) ou MORTO (UINT32_MAX)
#define Q_TOKEN_DELETED UINT32_MAX

// Estrutura existente (sem mudanças):
typedef struct {
    uint32_t* token_ids;      // [num_tokens] - Array de tokens (pode conter UINT32_MAX)
    size_t num_tokens;        // Número de tokens válidos (após compactação)
    size_t capacity;          // Capacidade total do array
} token_array_t;  // Implícito no código atual
```

**Arena Allocator:**
```c
// Estrutura existente (sem mudanças):
typedef struct {
    void* scratch_buffer;     // Buffer alinhado a Q_ALIGN
    size_t scratch_size;      // Tamanho total
    size_t scratch_head;      // Offset atual (sempre múltiplo de Q_ALIGN)
    size_t scratch_base_offset; // Offset base (protege estruturas do modelo)
} q_context;  // Já existe

// Invariante matemática:
// scratch_head % Q_ALIGN == 0 (sempre verdadeiro)
```

**MatMul AVX2:**
```c
// Sem mudanças estruturais
// Apenas remoção de código (prefetch manual)
```

**RoPE:**
```c
// Sem mudanças estruturais
// Apenas adição de validação DEBUG
```

### 2.2 Estados e Invariantes

**BPE Tokenizer:**

**Pré-condições:**
- `token_ids != NULL`
- `num_tokens > 0`
- `tok->num_merges > 0`
- Array `token_ids` contém apenas valores válidos (< vocab_size) ou `Q_TOKEN_DELETED`

**Pós-condições:**
- Array `token_ids` contém apenas tokens válidos (sem `Q_TOKEN_DELETED`)
- `num_tokens` atualizado para número de tokens válidos
- Todos os merges aplicáveis foram aplicados

**Invariantes de Loop:**
- `deleted_count <= num_tokens` (nunca excede tamanho)
- `write_idx <= i` durante compactação (nunca escreve além do lido)
- `token_ids[i] == Q_TOKEN_DELETED` ou `token_ids[i] < vocab_size` (estado válido)

**Arena Allocator:**

**Pré-condições:**
- `ctx != NULL`
- `ctx->scratch_buffer != NULL` (alinhado a Q_ALIGN)
- `ctx->scratch_head % Q_ALIGN == 0` (invariante garantida)
- `size > 0`

**Pós-condições:**
- `ptr != NULL` (se sucesso)
- `ptr % Q_ALIGN == 0` (alinhado)
- `ctx->scratch_head % Q_ALIGN == 0` (invariante mantida)
- `ctx->scratch_head <= ctx->scratch_size` (dentro dos limites)

**Invariantes:**
- `scratch_head % Q_ALIGN == 0` (sempre verdadeiro, garantido matematicamente)

**MatMul AVX2:**

**Pré-condições:**
- Tensores válidos e alinhados
- Dimensões compatíveis

**Pós-condições:**
- Resultado correto (sem mudanças)

**Invariantes:**
- Sem prefetch manual (removido)

**RoPE:**

**Pré-condições:**
- `cos` e `sin` arrays válidos e alinhados
- Layout duplicado: `cos[i] == cos[i+1]` para todo `i` par

**Pós-condições:**
- Rotação aplicada corretamente
- Layout validado (DEBUG apenas)

**Invariantes:**
- Layout duplicado mantido (garantido por produtor)

### 2.3 Grafo de Dependência

**BPE Tokenizer:**
```
(q_tokenizer_encode) -> (apply_bpe_merges)
(apply_bpe_merges) -> (lookup_merge_in_tokenizer) [se hash table disponível]
(apply_bpe_merges) -> (compactação lazy) [quando necessário]
```

**Arena Allocator:**
```
(q_arena_alloc) -> (safe_align_size)
(q_arena_alloc) -> (__builtin_assume_aligned) [novo]
(q_arena_alloc) -> (Q_ASSERT_ALIGNED) [DEBUG apenas]
```

**MatMul AVX2:**
```
(q_matmul_f32_avx2) -> (sem prefetch manual) [remoção]
```

**RoPE:**
```
(q_rope_f32_avx2) -> (validação layout DEBUG) [novo]
(q_rope_f32_avx2) -> (cálculo rotação)
```

**Validação:** Nenhum ciclo detectado ✅

---

## FASE 3: Prova e Análise (The "Proof")

### 3.1 Análise Assintótica

**BPE Tokenizer:**

**Tempo:**
- **Pior Caso:** O(m × n) onde m = merges, n = tokens
  - Loop externo `while(changed)`: até n iterações (cada merge reduz 1 token)
  - Loop de merges: m iterações
  - Loop de tokens: n iterações (pula mortos em O(1))
  - Compactação: O(n) quando densidade > 50%
  - **Total:** O(m × n) + O(n) = O(m × n) ✅

- **Caso Médio:** O(m × n) (similar ao pior caso)

**Espaço:**
- **Stack:** O(1) (apenas variáveis locais)
- **Heap:** O(n) (array de tokens, sem mudança)

**Validação:** O(m × n) ≤ Lower Bound O(m × n) × 1.1 ✅

**Arena Allocator:**

**Tempo:**
- **Pior Caso:** O(1) com ~2 ciclos
  - Aritmética: ~1 ciclo
  - Load: ~1 ciclo (cache hit)
  - **Total:** O(1) ✅

- **Caso Médio:** O(1) com ~2 ciclos

**Espaço:**
- **Stack:** O(1)
- **Heap:** O(1) (sem alocações adicionais)

**Validação:** O(1) ≤ Lower Bound O(1) × 1.1 ✅

**MatMul AVX2:**

**Tempo:**
- **Pior Caso:** Sem mudança (remoção de overhead)
- **Caso Médio:** Melhoria de ~1-5% (sem overhead de prefetch)

**Espaço:**
- Sem mudança

**Validação:** Solução proposta = Lower Bound ✅

**RoPE:**

**Tempo:**
- **Pior Caso:** O(1) em RELEASE (validação removida)
- **Caso Médio:** O(1) em RELEASE

**Espaço:**
- Sem mudança

**Validação:** Solução proposta = Lower Bound ✅

### 3.2 Demonstração Lógica

**BPE Tokenizer - Soft-Delete:**

**Prova de Correção:**
```
Invariante: Array contém tokens VIVOS ou MORTO (UINT32_MAX)

1. Inicialização: Todos tokens são VIVOS ✅

2. Merge: Se token_ids[i] == id1 && token_ids[next] == id2:
   - token_ids[i] = merged (VIVO)
   - token_ids[next] = Q_TOKEN_DELETED (MORTO)
   - Invariante mantida ✅

3. Compactação: Remove todos Q_TOKEN_DELETED
   - Array contém apenas tokens VIVOS ✅
   - Invariante mantida ✅

Conclusão: Algoritmo preserva invariante e produz resultado correto
```

**Prova de Complexidade:**
```
Sem memmove: Loop sobre tokens é O(n) (pula mortos em O(1))
Compactação: O(n) apenas quando necessário (densidade > 50%)
Total: O(m × n) + O(n) = O(m × n) ✅
```

**Arena Allocator - Invariante de Alinhamento:**

**Prova Matemática da Invariante:**
```
Base: scratch_head = 0 (0 % 64 == 0) ✅

Indução: Se scratch_head % 64 == 0, então:
- aligned_size = Q_ALIGN_SIZE(size) = múltiplo de 64
- new_head = scratch_head + aligned_size
- new_head % 64 = (scratch_head % 64 + aligned_size % 64) % 64
- new_head % 64 = (0 + 0) % 64 = 0 ✅

Conclusão: Invariante mantida por indução matemática
```

**Prova de Segurança:**
```
Se invariante garantida matematicamente:
- __builtin_assume_aligned é seguro
- Compilador pode gerar VMOVAPS sem verificação
- Segurança mantida sem overhead de runtime
```

### 3.3 Simulação de Falha (Failure Mode Analysis)

**BPE Tokenizer:**

**Resultado Correto (Target):**
- Array de tokens válidos após aplicação de todos os merges
- Complexidade O(m × n) no pior caso
- Sem movimentação de memória desnecessária

**Exemplos de Resultado Ruim/Errado (Anti-Patterns):**
- ❌ **Memmove repetido:** O(m × n³) - Catastrófico para prompts grandes
- ❌ **Re-scanning desnecessário:** `j--` causa re-processamento
- ❌ **Compactação muito frequente:** Overhead de O(n) a cada iteração
- ❌ **Array não compactado no final:** Tokens mortos deixados no array

**Arena Allocator:**

**Resultado Correto (Target):**
- Ponteiro alinhado retornado
- Overhead mínimo (~2 ciclos)
- Segurança mantida via invariante

**Exemplos de Resultado Ruim/Errado (Anti-Patterns):**
- ❌ **Validação de alinhamento em runtime:** Overhead de ~5 ciclos
- ❌ **Remover validação sem invariante:** Crash em AVX2 (segfault)
- ❌ **Dependência de dados:** Stall no pipeline de 4-5 ciclos

**MatMul AVX2:**

**Resultado Correto (Target):**
- Performance determinada apenas por HW prefetcher
- Sem overhead de prefetch manual

**Exemplos de Resultado Ruim/Errado (Anti-Patterns):**
- ❌ **Prefetch manual hardcoded:** Compete com HW prefetcher
- ❌ **Prefetch em CPUs modernas:** Overhead de 1-5%

**RoPE:**

**Resultado Correto (Target):**
- Rotação aplicada corretamente
- Layout validado em DEBUG

**Exemplos de Resultado Ruim/Errado (Anti-Patterns):**
- ❌ **Sem validação de layout:** Corrupção silenciosa se contrato violado
- ❌ **Validação em RELEASE:** Overhead desnecessário

### 3.4 Especificação Testável

**BPE Tokenizer:**

**Assinatura da Função:**
```c
static q_error_code apply_bpe_merges(
    const q_tokenizer* restrict tok,
    uint32_t* restrict token_ids,
    size_t* restrict num_tokens,
    size_t max_tokens
);
```

**Pré-condições:**
- `tok != NULL && tok->num_merges > 0`
- `token_ids != NULL && num_tokens != NULL`
- `*num_tokens > 0 && *num_tokens <= max_tokens`
- `token_ids[i] < tok->vocab_size` para todo `i < *num_tokens`

**Pós-condições:**
- `token_ids[i] != Q_TOKEN_DELETED` para todo `i < *num_tokens`
- `token_ids[i] < tok->vocab_size` para todo `i < *num_tokens`
- Todos os merges aplicáveis foram aplicados
- `*num_tokens` atualizado para número de tokens válidos

**Teste de Especificação:**
```
Input: token_ids = [0, 0, 0, 0], num_tokens = 4
Merge: "0,0 -> 1"
Output esperado: token_ids = [1, 1], num_tokens = 2
Validação: Complexidade O(m × n) onde m=1, n=4
```

**Arena Allocator:**

**Assinatura da Função:**
```c
void* q_arena_alloc(q_context* restrict ctx, size_t size);
```

**Pré-condições:**
- `ctx != NULL && ctx->scratch_buffer != NULL`
- `ctx->scratch_head % Q_ALIGN == 0` (invariante)
- `size > 0`

**Pós-condições:**
- `ptr != NULL` (se sucesso) ou `NULL` (se erro)
- `ptr % Q_ALIGN == 0` (se sucesso)
- `ctx->scratch_head % Q_ALIGN == 0` (invariante mantida)

**Teste de Especificação:**
```
Input: ctx com scratch_head = 0, size = 100
Output esperado: ptr alinhado a 64 bytes, scratch_head = 128
Validação: Overhead ≤ 2 ciclos (medido via benchmark)
```

**MatMul AVX2:**

**Assinatura da Função:**
```c
q_error_code q_matmul_f32_avx2(
    const q_tensor* restrict A,
    const q_tensor* restrict B,
    q_tensor* C,
    q_context* restrict ctx
);
```

**Pré-condições:**
- Tensores válidos e alinhados
- Dimensões compatíveis

**Pós-condições:**
- Resultado correto
- Sem prefetch manual no código

**Teste de Especificação:**
```
Input: Matrizes A[32,32], B[32,32]
Output esperado: C[32,32] = A @ B
Validação: Performance ≥ baseline (sem prefetch manual)
```

**RoPE:**

**Assinatura da Função:**
```c
q_error_code q_rope_f32_avx2(
    const float* restrict x,
    const float* restrict cos,
    const float* restrict sin,
    float* restrict output,
    uint32_t N
);
```

**Pré-condições:**
- Arrays válidos e alinhados
- `cos[i] == cos[i+1]` para todo `i` par (layout duplicado)
- `sin[i] == sin[i+1]` para todo `i` par

**Pós-condições:**
- Rotação aplicada corretamente
- Layout validado em DEBUG (se violado, abort)

**Teste de Especificação:**
```
Input: x = [1, 0, 0, 1], cos = [c, c, c, c], sin = [s, s, s, s]
Output esperado: Rotação aplicada corretamente
Validação: Layout validado em DEBUG (teste adversarial)
```

---

## FASE 4: Chain-of-Thought e Execução (Passo a Passo)

### 4.1 Definir Interface (Header)

**BPE Tokenizer:**
- Sem mudanças em `include/qorus.h` (função `static`)
- Adicionar `#define Q_TOKEN_DELETED UINT32_MAX` em `src/tokenizer/bpe.c`

**Arena Allocator:**
- Sem mudanças em `include/qorus.h` (assinatura mantida)
- Implementação interna muda apenas

**MatMul AVX2:**
- Sem mudanças em `include/qorus.h`

**RoPE:**
- Sem mudanças em `include/qorus.h`

### 4.2 Implementar Teste de Unidade (TDD)

**BPE Tokenizer:**
- Criar `tests/test_bpe_soft_delete.c`
- Testes:
  1. Caso básico: "aaaa" com merge "aa -> A" → "AA"
  2. Caso múltiplos merges: Aplicar várias regras
  3. Caso compactação: Validar que tokens mortos são removidos
  4. Caso performance: Medir complexidade O(m × n)

**Arena Allocator:**
- Estender `tests/test_main.c` ou criar `tests/test_arena_optimized.c`
- Testes:
  1. Validação de invariante de alinhamento
  2. Benchmark de overhead (deve ser ≤ 2 ciclos)
  3. Validação de segurança (não crasha em AVX2)

**MatMul AVX2:**
- Estender testes existentes
- Validação: Performance não degrada após remoção de prefetch

**RoPE:**
- Criar `tests/test_rope_layout.c`
- Testes:
  1. Validação de layout duplicado (DEBUG)
  2. Teste adversarial: Layout incorreto deve abortar em DEBUG

### 4.3 Implementar Kernel/Lógica (Draft)

**BPE Tokenizer:**
- Reescrever `apply_bpe_merges` com soft-delete
- Implementar compactação lazy (densidade > 50%)
- Remover `j--` (evitar re-scanning)

**Arena Allocator:**
- Remover validação de alinhamento em runtime
- Adicionar `__builtin_assume_aligned`
- Manter validação DEBUG

**MatMul AVX2:**
- Remover `_mm_prefetch` manual
- Remover `PREFETCH_DISTANCE` macro

**RoPE:**
- Adicionar validação DEBUG de layout no início da função

### 4.4 Otimização (Vectorização/Memory Access)

**BPE Tokenizer:**
- Loop de compactação pode ser otimizado com SIMD (futuro)
- Por enquanto, manter simples e correto

**Arena Allocator:**
- `__builtin_assume_aligned` permite otimizações do compilador
- Compilador pode gerar instruções alinhadas diretamente

**MatMul AVX2:**
- HW prefetcher já otimiza acesso sequencial
- Sem otimizações adicionais necessárias

**RoPE:**
- Validação apenas em DEBUG (zero overhead em RELEASE)

### 4.5 Verificação de Limites e Erros

**BPE Tokenizer:**
- Validar que `deleted_count` não excede `num_tokens`
- Validar que `write_idx` não excede `num_tokens` durante compactação
- Validar que tokens mortos são removidos no final

**Arena Allocator:**
- Validar invariante de alinhamento em DEBUG
- Validar overflow de `scratch_head`
- Validar bounds de `scratch_size`

**MatMul AVX2:**
- Validar que remoção de prefetch não quebra funcionalidade

**RoPE:**
- Validar que layout incorreto é detectado em DEBUG

---

## FASE 5: Checkpoints e Fatoração

### Checkpoint 1: Compilação Limpa
- [ ] Compilar sem warnings (`-Wall -Wextra -Werror`)
- [ ] Validar que todas as mudanças compilam

### Checkpoint 2: Teste Básico Passa
- [ ] Teste BPE soft-delete passa
- [ ] Teste arena otimizada passa
- [ ] Teste matmul sem prefetch passa
- [ ] Teste rope layout validation passa

### Checkpoint 3: Análise Estática Limpa
- [ ] `cppcheck` sem erros críticos
- [ ] `clang-tidy` sem warnings críticos
- [ ] Zero race conditions detectáveis

### Checkpoint 4: Métricas Quantitativas Validadas
- [ ] Complexidade BPE: O(m × n) ≤ Lower Bound × 1.1 ✅
- [ ] Overhead arena: ≤ 2 ciclos (medido via benchmark)
- [ ] Performance matmul: ≥ baseline (sem degradação)
- [ ] Cobertura de testes: ≥ 90% branches

### Fatoração (Complexidade Ciclomática)

**BPE Tokenizer:**
- V(G) estimado: ~8-10 (loops aninhados controlados)
- Linhas: ~80-100
- Níveis de indentação: 3-4
- **Veredito:** Aceitável (V(G) ≤ 10)

**Arena Allocator:**
- V(G) estimado: ~5-6 (validações simples)
- Linhas: ~30-40
- Níveis de indentação: 2-3
- **Veredito:** Aceitável (V(G) ≤ 10)

**MatMul AVX2:**
- V(G) estimado: Sem mudança (remoção de código)
- **Veredito:** Aceitável

**RoPE:**
- V(G) estimado: Sem mudança (adição de validação DEBUG)
- **Veredito:** Aceitável

---

## FASE 6: O Artefato de Execução (Machine-Readable Output)

### Contexto Ancorado

**Arquivos que Serão Modificados:**
1. `src/tokenizer/bpe.c` - Reescrever `apply_bpe_merges` com soft-delete
2. `src/core/memory.c` - Otimizar `q_arena_alloc` com `__builtin_assume_aligned`
3. `src/ops/avx2/matmul_fp32.c` - Remover prefetch manual
4. `src/ops/avx2/rope.c` - Adicionar validação DEBUG de layout

**Arquivos que Serão Criados:**
1. `tests/test_bpe_soft_delete.c` - Testes para BPE soft-delete
2. `tests/test_arena_optimized.c` - Testes para arena otimizada
3. `tests/test_rope_layout.c` - Testes para validação de layout RoPE

### Validação de Thresholds

**BPE Tokenizer:**
- ✅ Complexidade: O(m × n) ≤ Lower Bound O(m × n) × 1.1
- ✅ Fatores constantes: Medir após implementação (target: ≤ 2x teórico)

**Arena Allocator:**
- ✅ Complexidade: O(1) ≤ Lower Bound O(1) × 1.1
- ✅ Fatores constantes: ~2 ciclos ≤ 2x teórico (~1 ciclo)

**MatMul AVX2:**
- ✅ Performance: ≥ baseline (sem degradação)

**RoPE:**
- ✅ Overhead: Zero em RELEASE

### Checklist de Implementação

#### BPE Tokenizer - Soft-Delete

- [ ] **PASSO 1:** Adicionar `#define Q_TOKEN_DELETED UINT32_MAX` em `bpe.c`
- [ ] **PASSO 2:** Reescrever `apply_bpe_merges` com soft-delete
  - [ ] Substituir `memmove` por marcação `Q_TOKEN_DELETED`
  - [ ] Implementar loop que pula tokens mortos
  - [ ] Implementar compactação lazy (densidade > 50%)
  - [ ] Remover `j--` (evitar re-scanning)
  - [ ] Adicionar compactação final obrigatória
- [ ] **PASSO 3:** Criar `tests/test_bpe_soft_delete.c`
  - [ ] Teste caso básico: "aaaa" → "AA"
  - [ ] Teste múltiplos merges
  - [ ] Teste compactação lazy
  - [ ] Teste performance (complexidade O(m × n))
- [ ] **PASSO 4:** Validar compilação sem warnings
- [ ] **PASSO 5:** Executar testes e validar especificação
- [ ] **PASSO 6:** Benchmark de performance (antes/depois)

#### Arena Allocator - Otimização com Invariante

- [ ] **PASSO 1:** Remover validação de alinhamento em runtime (linha 222)
- [ ] **PASSO 2:** Adicionar `__builtin_assume_aligned` (linha ~252)
- [ ] **PASSO 3:** Manter validação DEBUG (linha ~258)
- [ ] **PASSO 4:** Criar `tests/test_arena_optimized.c`
  - [ ] Teste invariante de alinhamento
  - [ ] Benchmark de overhead (target: ≤ 2 ciclos)
  - [ ] Teste segurança (não crasha em AVX2)
- [ ] **PASSO 5:** Validar compilação sem warnings
- [ ] **PASSO 6:** Executar testes e validar especificação
- [ ] **PASSO 7:** Benchmark de performance (antes/depois)

#### MatMul AVX2 - Remover Prefetch Manual

- [ ] **PASSO 1:** Remover `#define PREFETCH_DISTANCE 192` (linha 11)
- [ ] **PASSO 2:** Remover `_mm_prefetch` calls (linhas 375-377)
- [ ] **PASSO 3:** Validar compilação sem warnings
- [ ] **PASSO 4:** Executar testes existentes (não deve quebrar)
- [ ] **PASSO 5:** Benchmark de performance (não deve degradar)

#### RoPE - Validação DEBUG de Layout

- [ ] **PASSO 1:** Adicionar validação DEBUG no início de `q_rope_f32_avx2`
  - [ ] Loop sobre `cos` array validando `cos[i] == cos[i+1]`
  - [ ] Loop sobre `sin` array validando `sin[i] == sin[i+1]`
  - [ ] Abort com mensagem clara se violado
- [ ] **PASSO 2:** Criar `tests/test_rope_layout.c`
  - [ ] Teste layout correto (não deve abortar)
  - [ ] Teste adversarial: layout incorreto deve abortar em DEBUG
- [ ] **PASSO 3:** Validar compilação sem warnings
- [ ] **PASSO 4:** Executar testes e validar especificação

### Pseudo-Code/Spec

#### BPE Tokenizer - Soft-Delete

```c
static q_error_code apply_bpe_merges(...) {
    bool changed = true;
    size_t deleted_count = 0;
    const size_t COMPACT_THRESHOLD = (*num_tokens) / 2;
    
    while (changed) {
        changed = false;
        
        for (uint32_t m = 0; m < tok->num_merges; m++) {
            // Obter regra de merge
            uint32_t id1 = tok->merges[m].token_id1;
            uint32_t id2 = tok->merges[m].token_id2;
            uint32_t merged = lookup_merge(...);
            
            // Escanear tokens (pular mortos)
            for (size_t i = 0; i < *num_tokens; i++) {
                if (token_ids[i] == Q_TOKEN_DELETED) continue;
                
                // Encontrar próximo token vivo
                size_t next = i + 1;
                while (next < *num_tokens && token_ids[next] == Q_TOKEN_DELETED) {
                    next++;
                }
                if (next >= *num_tokens) break;
                
                // Verificar merge
                if (token_ids[i] == id1 && token_ids[next] == id2) {
                    token_ids[i] = merged;
                    token_ids[next] = Q_TOKEN_DELETED;
                    deleted_count++;
                    changed = true;
                    // NÃO fazer i-- ou recuo
                }
            }
        }
        
        // Compactação lazy
        if (deleted_count > COMPACT_THRESHOLD) {
            size_t write_idx = 0;
            for (size_t i = 0; i < *num_tokens; i++) {
                if (token_ids[i] != Q_TOKEN_DELETED) {
                    token_ids[write_idx++] = token_ids[i];
                }
            }
            *num_tokens = write_idx;
            deleted_count = 0;
        }
    }
    
    // Compactação final obrigatória
    size_t write_idx = 0;
    for (size_t i = 0; i < *num_tokens; i++) {
        if (token_ids[i] != Q_TOKEN_DELETED) {
            token_ids[write_idx++] = token_ids[i];
        }
    }
    *num_tokens = write_idx;
    
    return Q_OK;
}
```

#### Arena Allocator - Otimização

```c
void* q_arena_alloc(q_context* restrict ctx, size_t size) {
    // Validações críticas (sempre ativas)
    Q_HOT_PATH_VALIDATE(ctx != NULL, Q_ERR_INVALID_ARG);
    Q_HOT_PATH_VALIDATE(ctx->scratch_buffer != NULL, Q_ERR_INVALID_ARG);
    
    // Cálculo de alinhamento
    size_t aligned_size = safe_align_size(size);
    if (aligned_size == 0) return NULL;
    
    // Overflow check
    if (__builtin_expect(ctx->scratch_head > SIZE_MAX - aligned_size, 0)) {
        return NULL;
    }
    
    size_t new_head = ctx->scratch_head + aligned_size;
    
    // Bounds check
    if (__builtin_expect(new_head > ctx->scratch_size, 0)) {
        return NULL;
    }
    
    // Usar __builtin_assume_aligned (invariante garantida matematicamente)
    void* base_ptr = __builtin_assume_aligned(ctx->scratch_buffer, Q_ALIGN);
    void* ptr = (uint8_t*)base_ptr + ctx->scratch_head;
    
    ctx->scratch_head = new_head; // Invariante mantida
    
    #ifdef DEBUG
    // Validação apenas em DEBUG
    if (new_head % Q_ALIGN != 0) {
        fprintf(stderr, "ERROR: Invariante violada!\n");
        abort();
    }
    Q_ASSERT_ALIGNED(ptr);
    #endif
    
    return ptr;
}
```

#### MatMul AVX2 - Remover Prefetch

```c
// REMOVIDO:
// #define PREFETCH_DISTANCE 192
// _mm_prefetch((const char*)(A_row + k + PREFETCH_DISTANCE), _MM_HINT_T0);
```

#### RoPE - Validação DEBUG

```c
q_error_code q_rope_f32_avx2(...) {
    // ... validações existentes ...
    
    #ifdef DEBUG
    // Validação de contrato de layout
    for (uint32_t i = 0; i < N; i += 2) {
        if (cos[i] != cos[i+1] || sin[i] != sin[i+1]) {
            fprintf(stderr, "FATAL: RoPE table corrupted/invalid layout at index %u\n", i);
            abort();
        }
    }
    #endif
    
    // ... resto da função ...
}
```

---

## Resumo Executivo

**Prioridade:** CRÍTICA - Correções de complexidade algorítmica e otimizações de pipeline

**Impacto Esperado:**
- **BPE:** Redução de O(m × n³) para O(m × n) - **Melhoria de 1000× para prompts grandes**
- **Arena:** Redução de overhead de ~6.5 para ~2 ciclos - **Melhoria de 3.25×**
- **MatMul:** Remoção de overhead de prefetch manual - **Melhoria de 1-5%**
- **RoPE:** Zero overhead em RELEASE + segurança em DEBUG

**Riscos:**
- BPE: Implementação complexa, requer testes extensivos
- Arena: Invariante deve ser garantida matematicamente
- MatMul: Remoção de prefetch pode degradar em CPUs antigas (mitigado por flag condicional)
- RoPE: Validação DEBUG pode ser lenta para arrays grandes (aceitável, apenas DEBUG)

**Próximos Passos:**
1. Implementar BPE soft-delete (URGENTE)
2. Implementar arena otimizada (ALTO)
3. Remover prefetch manual (MÉDIO)
4. Adicionar validação RoPE (MÉDIO)

---

**Status:** ✅ Planejamento completo e pronto para execução

