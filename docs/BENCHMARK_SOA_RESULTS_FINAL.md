# Benchmark Results - SoA Optimizations Impact (Final)

**Date:** 2025-01-XX  
**Ferramenta:** `benchmark_sampling` + `perf stat`  
**Foco:** Impacto das otimizações SoA e `qsort_soa()` no sampling

---

## Configuração do Benchmark

- **Vocab Size:** 32,000 (tamanho típico)
- **Warmup Iterations:** 10
- **Benchmark Iterations:** 1,000
- **Test Cases:**
  1. Greedy Sampling (temperature=0.0)
  2. Top-k Sampling (k=10)
  3. Top-p Sampling (p=0.9)
  4. Combined Top-k + Top-p (k=10, p=0.9)

---

## Resultados de Performance

### Tempo por Chamada `q_sample_token()`

| Test Case | Tempo (ms) | Throughput (calls/sec) | Observação |
|-----------|------------|------------------------|------------|
| Greedy | [A ser preenchido] | [A ser preenchido] | Sem softmax, mais rápido |
| Top-k (k=10) | [A ser preenchido] | [A ser preenchido] | Usa quickselect + qsort_soa |
| Top-p (p=0.9) | [A ser preenchido] | [A ser preenchido] | Usa binary search + qsort_soa |
| Combined | [A ser preenchido] | [A ser preenchido] | Mais complexo |

### Métricas de Cache (perf stat)

| Métrica | Valor | Observação |
|---------|-------|------------|
| **Cache Misses** | [A ser preenchido] | Redução esperada: ~40% |
| **Cache References** | [A ser preenchido] | Total de acessos |
| **LLC Loads** | [A ser preenchido] | Acessos ao último nível |
| **LLC Load Misses** | [A ser preenchido] | Redução esperada: ~38% |
| **Cycles** | [A ser preenchido] | Ciclos totais |
| **Instructions** | [A ser preenchido] | Instruções executadas |

---

## Comparação: Antes vs Depois

### Antes das Otimizações SoA

**Problemas Identificados:**
- Layout AoS (Array of Structures) causava 50% waste de cache line
- Conversão AoS → qsort → AoS desperdiçava 16k bytes por sort
- Prefetch incorreto acessava memória fora do intervalo

**Métricas Esperadas (Baseado em Análise Teórica):**
- Cache Miss Rate: ~53% (cache references)
- LLC Miss Rate: ~96% (LLC loads)
- Bandwidth: 16k bytes copiados por sort (k ≥ 64)

### Depois das Otimizações SoA

**Melhorias Implementadas:**
- ✅ Layout SoA (Structure of Arrays) - 100% cache line utilization
- ✅ `qsort_soa()` - elimina conversão AoS (0 bytes copiados)
- ✅ Prefetch bounds check corrigido
- ✅ Insertion sort para arrays pequenos (n < 16)

**Métricas Esperadas (Baseado em Análise Teórica):**
- Cache Miss Rate: < 30% (redução de ~44%)
- LLC Miss Rate: < 60% (redução de ~38%)
- Bandwidth: 0 bytes copiados (100% redução)

**Melhorias Esperadas:**
- **Memória:** Elimina 8KB alocação temporária por sort (k=1000)
- **Bandwidth:** Elimina 16k bytes copiados por sort
- **Cache Locality:** Mantém SoA durante todo o sort
- **Performance:** ≥ 20% mais rápido (target)

---

## Análise Teórica vs Prática

### Análise Teórica

**Cache Complexity:**
- **AoS:** 8 elementos por cache line, apenas prob usado → 50% waste
- **SoA:** 16 floats por cache line, todos usados → 100% utilization
- **Melhoria:** 50% redução em cache misses (V/16 vs V/8)

**Bandwidth:**
- **Antes:** 16k bytes copiados (SoA → AoS → qsort → AoS → SoA)
- **Depois:** 0 bytes copiados (sort direto em SoA)
- **Melhoria:** 100% redução em bandwidth de conversão

**Complexidade Assintótica:**
- **Antes:** O(V + k log k) - mesmo algoritmo
- **Depois:** O(V + k log k) - mesmo algoritmo
- **Melhoria:** Mesma complexidade, mas melhores fatores constantes

### Validação Prática

**Resultados Reais:** (Será preenchido após execução do benchmark)

**Comando para Executar:**
```bash
make benchmark-sampling
perf stat -e cache-misses,cache-references,LLC-loads,LLC-load-misses,cycles,instructions -r 1 ./build/tools/benchmark_sampling
```

---

## Próximos Passos

1. ✅ **Benchmark Configurado:** `benchmark_sampling` criado
2. ⏳ **Executar Benchmark:** Medir resultados reais
3. ⏳ **Comparar com Teoria:** Validar melhorias esperadas
4. ⏳ **Profiling Detalhado:** `perf annotate` para identificar hotspots restantes
5. ⏳ **Otimizações Adicionais:** Baseado em resultados do profiling

---

**Status:** Benchmark configurado e pronto para execução. Aguardando resultados reais para documentação final.

