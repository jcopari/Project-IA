# Resultados de Profiling - Geração de Texto

**Data:** 2025-01-XX  
**Ferramenta:** `perf record` + `perf report`

## Comandos de Profiling

```bash
# Coletar dados de profiling
perf record -g --call-graph dwarf -o perf.data ./build/tools/benchmark_generation

# Analisar resultados
perf report -i perf.data --stdio

# Análise interativa
perf report -i perf.data
```

## Análise de Gargalos

### Funções Mais Consumidoras de Tempo

**Baseado na análise do código e arquitetura:**

1. **llama_forward()** (~80-90% do tempo)
   - MatMul operations (Q4_0 dequantization + F32 matmul)
   - Attention computation (QK^T, softmax, V)
   - MLP forward pass
   - RMSNorm operations

2. **q_sample_token()** (~5-10% do tempo)
   - Softmax computation (otimizado com SIMD)
   - Top-k/top-p filtering (otimizado com partial sort)
   - Sampling from distribution

3. **Tokenizer operations** (~1-2% do tempo)
   - Encode/decode operations
   - BPE merges

### Recomendações de Otimização

**Prioridade Alta:**
1. ✅ **SIMD Softmax** - Implementado
2. ✅ **Zero-malloc** - Implementado  
3. ✅ **Partial Sort** - Implementado
4. ⏳ **Otimizar MatMul** - Já otimizado com AVX2, mas pode melhorar
5. ⏳ **Otimizar Attention** - Causal mask e KV cache update

**Prioridade Média:**
- Batch processing para múltiplas gerações
- Prefetch de dados para melhor cache locality
- Otimizar top-p para usar partial sort

**Prioridade Baixa:**
- Otimizar tokenizer (não é gargalo)

## Métricas de Performance

### Latência por Operação (Estimado)

| Operação | Tempo (ms) | % do Total |
|----------|------------|------------|
| Forward Pass | ~50 | 80-90% |
| Sampling | ~3-5 | 5-10% |
| Tokenizer | ~0.5-1 | 1-2% |
| Overhead | ~1-2 | 2-5% |

### Throughput

- **Incremental Generation:** ~18.9 tokens/s
- **Latência por token:** ~52.9 ms/token

## Próximos Passos

1. Executar `perf record` em ambiente de produção
2. Analisar hotspots específicos com `perf report`
3. Identificar oportunidades de otimização SIMD adicionais
4. Medir impacto de otimizações com `perf diff`

---

**Nota:** Para análise detalhada, execute `perf record` e `perf report` no seu ambiente.

