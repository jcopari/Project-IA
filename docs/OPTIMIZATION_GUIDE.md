# Guia de Otimização - Geração de Texto

**Data:** 2025-01-XX  
**Versão:** Qorus-IA v2.0

## Resumo Executivo

Este documento identifica oportunidades de otimização baseadas na análise do código e arquitetura do sistema de geração de texto.

---

## 1. Análise de Gargalos (Baseada em Arquitetura)

### 1.1 Forward Pass (Modelo) - ~80-90% do Tempo Total

**Funções Críticas:**
- `llama_forward()` - Loop principal através das camadas
- `llama_layer_forward()` - Processamento de cada camada
- `llama_attention_forward()` - Cálculo de atenção (maior parte do tempo)

**Otimizações Já Aplicadas:**
- ✅ AVX2 para MatMul
- ✅ AVX2 para RMSNorm
- ✅ AVX2 para RoPE
- ✅ AVX2 para SiLU
- ✅ AVX2 para Softmax

**Oportunidades de Otimização:**

#### 1.1.1 Causal Mask (Otimização Crítica)

**Status Atual:** `q_causal_mask_f32_avx2()` já otimizado com AVX2

**Análise:**
- Complexidade: O(N²) onde N = seq_len
- Para geração incremental (seq_len = 1), mask é trivial
- Para prefill (seq_len > 1), mask é aplicado uma vez

**Otimizações Possíveis:**
1. **Lazy Masking:** Aplicar mask apenas quando necessário (não em cada forward)
2. **Incremental Masking:** Para geração incremental, apenas adicionar nova linha/coluna
3. **SIMD Blocking:** Melhorar cache locality com blocking

**Impacto Esperado:** 5-10% de redução no tempo de prefill

#### 1.1.2 KV Cache Update

**Status Atual:** KV cache é atualizado durante `llama_attention_forward()`

**Análise:**
- Para geração incremental, apenas novo token precisa ser adicionado ao KV cache
- Operação é O(L × D) onde L = n_layers, D = head_dim

**Otimizações Possíveis:**
1. **SIMD Copy:** Usar `memcpy` otimizado ou AVX2 para copiar KV cache
2. **In-place Update:** Evitar cópias desnecessárias
3. **Prefetch:** Prefetch de dados do KV cache antes de uso

**Impacto Esperado:** 2-5% de redução no tempo incremental

### 1.2 Sampling - ~5-10% do Tempo Total

**Status Atual:**
- ✅ SIMD Softmax implementado
- ✅ Partial Sort O(V + k log k) para top-k
- ✅ Partial Sort O(V + k log k) para top-p (corrigido)
- ✅ Zero-malloc com arena

**Otimizações Adicionais Possíveis:**

#### 1.2.1 SIMD para Aplicação de Temperatura

**Status:** Atualmente escalar

**Otimização:**
```c
// Aplicar temperatura com AVX2
__m256 temp_vec = _mm256_set1_ps(temperature);
for (uint32_t i = 0; i < vocab_size; i += 8) {
    __m256 logits_vec = _mm256_load_ps(logits + i);
    __m256 scaled = _mm256_div_ps(logits_vec, temp_vec);
    _mm256_store_ps(scaled_logits + i, scaled);
}
```

**Impacto Esperado:** 1-2% de redução no tempo de sampling

#### 1.2.2 Cache-Friendly Top-k/Top-p

**Status:** Já otimizado com partial sort

**Otimização Adicional:**
- Usar heap mínimo em vez de quickselect para top-p (melhor cache locality)
- Complexidade: O(V log k) em vez de O(V + k log k), mas melhor constante

**Impacto Esperado:** Marginal (já otimizado)

### 1.3 Tokenizer - ~1-2% do Tempo Total

**Status:** Não crítico para performance

**Otimizações Possíveis:**
- SIMD para comparação de strings
- Cache de tokens frequentes
- **Impacto:** Baixo (não é gargalo)

---

## 2. Batch Processing

### 2.1 Implementação de Batch Generation

**Objetivo:** Processar múltiplas sequências em paralelo para melhorar throughput

**Estrutura Proposta:**

```c
typedef struct {
    q_generation_state* states;  // Array de estados
    uint32_t batch_size;         // Número de sequências
    uint32_t* shared_logits;    // Logits compartilhados (se mesmo modelo)
} q_batch_generation_state;

q_error_code q_generate_batch(
    q_batch_generation_state* restrict batch_state,
    uint32_t* restrict tokens_out,      // [batch_size, max_tokens]
    uint32_t* restrict num_tokens_out   // [batch_size]
);
```

**Estratégia:**
1. **Prefill Paralelo:** Processar todos os prompts em paralelo
2. **Sampling Paralelo:** Sample de todas as sequências simultaneamente
3. **Forward Paralelo:** Usar SIMD para processar múltiplas sequências

**Complexidade:**
- Tempo: O(B × T × F) onde B = batch_size, T = tokens, F = forward time
- Espaço: O(B × KV_cache_size)

**Benefícios:**
- Melhor utilização de SIMD (processar 8 sequências por vez)
- Throughput aumentado: ~4-8x para batch_size = 8
- Melhor cache locality

**Desafios:**
- Gerenciamento de memória (KV cache para cada sequência)
- Sincronização (todas as sequências devem avançar juntas)
- Padding para sequências de tamanhos diferentes

**Prioridade:** Média (melhora throughput, mas aumenta complexidade)

---

## 3. Otimizações de Memória

### 3.1 KV Cache Layout

**Status Atual:** KV cache é alocado como buffer contínuo

**Otimização Proposta:**
- **Interleaved Layout:** Intercalar K e V para melhor cache locality
- **Block Layout:** Organizar por camadas para melhor prefetch

**Impacto Esperado:** 2-5% de redução em cache misses

### 3.2 Arena Allocation Strategy

**Status Atual:** Arena resetada após cada token

**Otimização Proposta:**
- **Incremental Reset:** Resetar apenas buffers temporários, manter estruturas persistentes
- **Memory Pool:** Pool de buffers pré-alocados para reduzir fragmentação

**Impacto Esperado:** Marginal (já otimizado)

---

## 4. Otimizações de Algoritmo

### 4.1 Attention Optimization

**Status Atual:** Attention calculada completamente a cada forward

**Otimizações Possíveis:**

#### 4.1.1 Flash Attention (Algoritmo)

**Descrição:** Dividir atenção em blocos para reduzir memória e melhorar cache

**Complexidade:**
- Memória: O(N) em vez de O(N²)
- Tempo: O(N²) (mesmo, mas melhor constante)

**Implementação:** Requer refatoração significativa

**Prioridade:** Baixa (complexidade alta, benefício moderado)

#### 4.1.2 Sparse Attention

**Descrição:** Aplicar atenção apenas a tokens relevantes

**Complexidade:** O(kN) onde k << N

**Implementação:** Requer análise de padrões de atenção

**Prioridade:** Baixa (requer pesquisa)

### 4.2 Quantization

**Status Atual:** Modelo usa Q4_0 quantization para pesos

**Otimizações Possíveis:**
- **INT8 Quantization:** Reduzir precisão de ativações
- **Dynamic Quantization:** Quantizar apenas camadas não-críticas

**Impacto Esperado:** 2-4x speedup, mas reduz qualidade

**Prioridade:** Média (trade-off qualidade/performance)

---

## 5. Priorização de Otimizações

### Prioridade Alta (Implementar Primeiro)

1. ✅ **SIMD Softmax** - Implementado
2. ✅ **Partial Sort Top-p** - Implementado e corrigido
3. ⏳ **SIMD Temperature Scaling** - Fácil, impacto moderado
4. ⏳ **Causal Mask Optimization** - Impacto significativo no prefill

### Prioridade Média

1. ⏳ **KV Cache Update Optimization** - Impacto moderado
2. ⏳ **Batch Processing** - Alto impacto, mas alta complexidade
3. ⏳ **Memory Layout Optimization** - Impacto baixo-moderado

### Prioridade Baixa

1. ⏳ **Flash Attention** - Alta complexidade, benefício moderado
2. ⏳ **Sparse Attention** - Requer pesquisa
3. ⏳ **Quantization Avançada** - Trade-off qualidade/performance

---

## 6. Métricas de Sucesso

### Benchmarks Alvo

**Atual (Modelo Dummy 2 layers):**
- Prefill: ~26 ms/token
- Incremental: ~53 ms/token
- Throughput: ~18.9 tokens/s

**Alvo (Após Otimizações):**
- Prefill: < 20 ms/token (redução de ~25%)
- Incremental: < 40 ms/token (redução de ~25%)
- Throughput: > 25 tokens/s (aumento de ~30%)

### Validação

- ✅ Testes unitários passando
- ✅ Testes end-to-end passando
- ✅ Performance dentro dos thresholds planejados
- ✅ Zero regressões de qualidade

---

## 7. Próximos Passos

1. **Implementar SIMD Temperature Scaling** (1-2 horas)
2. **Otimizar Causal Mask** (2-4 horas)
3. **Otimizar KV Cache Update** (2-4 horas)
4. **Implementar Batch Processing** (1-2 semanas)
5. **Validar com Benchmarks** (contínuo)

---

**Documento atualizado:** 2025-01-XX

