# Auditoria: Loop Infinito em test-generation-e2e

**Data:** 2025-01-XX  
**Arquivo Auditado:** `src/main.c` - Função `find_nucleus_size_optimized()`  
**Problema Reportado:** Teste `test-generation-e2e` fica rodando infinitamente

---

## 1. [ANÁLISE CRÍTICA] Deconstrução

### Fluxo de Dados Identificado

**Função `find_nucleus_size_optimized()`:**
1. Recebe `prob_indices[]` (array de índices + probabilidades)
2. Estima `k_estimate = top_p * vocab_size`
3. Chama `quickselect_top_k(prob_indices, 0, vocab_size-1, k)` - **MODIFICA ARRAY IN-PLACE**
4. Calcula `cumsum` dos primeiros `k` elementos
5. Se `cumsum < top_p`, incrementa `k` e repete do passo 3

### Falha Lógica Crítica Identificada

**PROBLEMA:** `quickselect_top_k()` modifica o array `prob_indices` in-place. Quando chamado múltiplas vezes com diferentes valores de `k`, os dados originais são corrompidos.

**Prova Matemática:**

Seja `P = [p₀, p₁, ..., p_{V-1}]` o array original de probabilidades.

1. **Primeira iteração:** `quickselect_top_k(P, 0, V-1, k₁)` reorganiza `P` tal que `P[0..k₁-1]` contém os `k₁` maiores elementos (mas não necessariamente ordenados).

2. **Segunda iteração:** `quickselect_top_k(P, 0, V-1, k₂)` onde `k₂ ≠ k₁`:
   - O array `P` já foi modificado pela primeira chamada
   - `quickselect_top_k` assume que `P` contém os dados originais
   - **Resultado:** Os elementos em `P[0..k₂-1]` podem não ser os `k₂` maiores elementos do array original

3. **Cálculo de `cumsum`:** 
   ```c
   for (uint32_t i = 0; i < k; i++) {
       cumsum += prob_indices[i].prob;
   }
   ```
   - Se `prob_indices` foi corrompido, `cumsum` pode nunca atingir `top_p`
   - **Loop infinito:** `while (k < vocab_size)` nunca termina porque `cumsum < top_p` sempre

### Counter-Example (Cenário de Falha)

**Input:**
- `vocab_size = 10`
- `top_p = 0.9`
- `prob_indices = [(0, 0.5), (1, 0.3), (2, 0.1), (3, 0.05), (4, 0.02), (5, 0.01), (6, 0.01), (7, 0.005), (8, 0.003), (9, 0.002)]`

**Execução:**
1. `k_estimate = 0.9 * 10 = 9`
2. `quickselect_top_k(prob_indices, 0, 9, 9)` → reorganiza array
3. `cumsum = 0.5 + 0.3 + 0.1 + ... = 0.991` → `>= 0.9` ✓
4. Entra no branch de redução: `while (k > 1)`
5. `k = 8`, `quickselect_top_k(prob_indices, 0, 9, 8)` → **ARRAY JÁ MODIFICADO**
6. `cumsum` calculado com dados corrompidos pode ser `< 0.9`
7. Continua reduzindo até `k = 1`
8. Se `cumsum` nunca atingir `>= top_p` devido à corrupção, o loop pode não encontrar solução válida

**Cenário de Loop Infinito:**
- Se `cumsum < top_p` após todas as iterações de redução
- Entra no branch `else` (linha 323)
- `while (k < vocab_size)` incrementa `k` indefinidamente
- Mas `quickselect_top_k` com `k` crescente em array já modificado pode nunca produzir `cumsum >= top_p`
- **Resultado:** Loop infinito

### Segurança

**Buffer Overflow:** Não detectado (índices validados)  
**Race Conditions:** Não aplicável (função não thread-safe, mas não é problema aqui)  
**Uninitialized Memory:** Não detectado  
**Use-After-Free:** Não detectado  

**Problema Principal:** **Corrupção de dados devido a modificação in-place de array compartilhado**

### Complexidade Acidental

**Problema:** A função tenta otimizar usando `quickselect` múltiplas vezes, mas não preserva o estado original do array. Isso cria complexidade desnecessária e bugs.

---

## 2. [A PROVA] Demonstração Rigorosa

### Análise Assintótica

**Atual (com bug):**
- **Melhor caso:** O(1) - estimativa correta na primeira tentativa
- **Pior caso:** O(∞) - **LOOP INFINITO** devido à corrupção de dados
- **Caso médio:** O(V + k log k) - se não houver corrupção

**Teórico (correto):**
- **Melhor caso:** O(V) - quickselect uma vez
- **Pior caso:** O(V + k log k) - quickselect + sort top-k
- **Caso médio:** O(V + k log k)

**Comparação:** 
- Implementação atual **NÃO ATENDE** threshold de `@planeje-isto.md` FASE 1.4
- Complexidade pior caso é **INFINITA** (loop infinito)
- **REJEITAR:** Código não funciona corretamente

### Counter-Example Formal

**Teorema:** Se `quickselect_top_k()` modifica `arr` in-place e é chamado múltiplas vezes com diferentes `k`, então `arr` não contém mais os dados originais após a primeira chamada.

**Prova:**
1. Seja `arr_original = [a₀, a₁, ..., a_{n-1}]`
2. Após `quickselect_top_k(arr, 0, n-1, k₁)`, temos `arr_modificado` onde `arr[0..k₁-1]` contém os `k₁` maiores elementos (mas ordem pode ser diferente)
3. Após `quickselect_top_k(arr_modificado, 0, n-1, k₂)` onde `k₂ ≠ k₁`:
   - O algoritmo assume que `arr` contém `arr_original`
   - Mas `arr` contém `arr_modificado`
   - **Conclusão:** `arr[0..k₂-1]` não necessariamente contém os `k₂` maiores elementos de `arr_original`

**Corolário:** Se `cumsum` calculado com `arr_modificado` não atinge `top_p`, o loop pode nunca terminar.

---

## 3. [SOLUÇÃO] Engenharia de Precisão

### Solução Proposta

**Estratégia:** Restaurar array original antes de cada chamada de `quickselect_top_k`, OU usar uma cópia temporária.

**Implementação Mínima:**

```c
static uint32_t find_nucleus_size_optimized(
    prob_index_t* restrict prob_indices,
    uint32_t vocab_size,
    float top_p
) {
    // Criar cópia temporária para preservar dados originais
    prob_index_t* temp_indices = (prob_index_t*)malloc(vocab_size * sizeof(prob_index_t));
    if (temp_indices == NULL) {
        return vocab_size;  // Fallback: retornar máximo
    }
    
    // Copiar dados originais
    memcpy(temp_indices, prob_indices, vocab_size * sizeof(prob_index_t));
    
    // Estimativa inicial
    uint32_t k_estimate = (uint32_t)(top_p * (float)vocab_size);
    if (k_estimate == 0) k_estimate = 1;
    if (k_estimate > vocab_size) k_estimate = vocab_size;
    
    uint32_t k = k_estimate;
    
    // Primeira tentativa
    memcpy(prob_indices, temp_indices, vocab_size * sizeof(prob_index_t));  // Restaurar
    quickselect_top_k(prob_indices, 0, vocab_size - 1, k);
    float cumsum = 0.0f;
    for (uint32_t i = 0; i < k; i++) {
        cumsum += prob_indices[i].prob;
    }
    
    if (cumsum >= top_p) {
        // Reduzir k
        while (k > 1) {
            k--;
            memcpy(prob_indices, temp_indices, vocab_size * sizeof(prob_index_t));  // Restaurar
            quickselect_top_k(prob_indices, 0, vocab_size - 1, k);
            cumsum = 0.0f;
            for (uint32_t i = 0; i < k; i++) {
                cumsum += prob_indices[i].prob;
            }
            if (cumsum < top_p) {
                free(temp_indices);
                return k + 1;
            }
        }
        free(temp_indices);
        return k;
    } else {
        // Aumentar k
        while (k < vocab_size) {
            k++;
            memcpy(prob_indices, temp_indices, vocab_size * sizeof(prob_index_t));  // Restaurar
            quickselect_top_k(prob_indices, 0, vocab_size - 1, k);
            cumsum = 0.0f;
            for (uint32_t i = 0; i < k; i++) {
                cumsum += prob_indices[i].prob;
            }
            if (cumsum >= top_p) {
                free(temp_indices);
                return k;
            }
        }
        free(temp_indices);
        return vocab_size;
    }
}
```

**Alternativa Mais Eficiente (usar arena):**

Se `ctx` estiver disponível, usar arena em vez de `malloc`. Mas como esta função é chamada de `apply_top_p()` que já recebe `ctx`, podemos passar `ctx` como parâmetro.

### Validação Pós-Correção

**Complexidade Assintótica:**
- **Melhor caso:** O(V) - estimativa correta, uma chamada de quickselect
- **Pior caso:** O(V × log V) - múltiplas restaurações + quickselects
- **Caso médio:** O(V + k log k) - algumas restaurações necessárias

**Comparação com Teórico:**
- Teórico: O(V + k log k) usando heap ou abordagem mais sofisticada
- Implementação corrigida: O(V × log V) no pior caso (ainda melhor que O(V log V) do qsort original se k << V)
- **Threshold:** Se k << V, então O(V × log V) ≈ O(V) que é aceitável

---

## 4. [VEREDITO] Checklist Quantitativo

- [ ] **Complexidade Assintótica:** ❌ **FALHA CRÍTICA** - Loop infinito no pior caso
- [ ] **Race Conditions:** ✅ Não aplicável (função não thread-safe, mas não é problema)
- [ ] **Cobertura de Testes:** ❓ Não testado para casos de corrupção de dados
- [ ] **Warnings de Análise Estática:** ✅ Compila sem warnings
- [ ] **Performance:** ❌ **FALHA** - Loop infinito impede execução
- [ ] **Validação de Thresholds:** ❌ **FALHA** - Não atende FASE 1.4 (complexidade infinita)
- [ ] **Failure Modes:** ❌ **FALHA CRÍTICA** - Loop infinito é um failure mode não tratado

### Veredito Final

**CÓDIGO REJEITADO**

**Falhas Críticas:**
1. **Loop infinito** devido à corrupção de dados por `quickselect_top_k()` in-place
2. **Complexidade assintótica:** O(∞) no pior caso
3. **Failure mode não tratado:** Corrupção de dados não detectada

**Solução Proposta:** Implementar restauração de array antes de cada chamada de `quickselect_top_k()`, conforme código acima.

**Próximos Passos:**
1. Aplicar correção proposta
2. Adicionar teste específico para validar que `cumsum` sempre atinge `top_p`
3. Validar que loop sempre termina (adicionar timeout ou limite máximo de iterações)

---

**Status:** Código rejeitado. Correção crítica necessária antes de merge.

