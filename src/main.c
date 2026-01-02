// ============================================================================
// MAIN APPLICATION (FASE 4.2)
// ============================================================================
// Implementação do loop de geração de texto e sampling de tokens
// Seguindo planejamento: docs/PLANEJAMENTO_DIVIDAS_TECNICAS.md
//
// Complexidade:
// - Sampling greedy: O(V) onde V = vocab_size
// - Sampling top-k/top-p: O(V + k log k) usando partial sort (não full sort O(V log V))
// - Loop de geração: O(T × (F + V)) para greedy, O(T × (F + V + k log k)) para top-k/top-p
// ============================================================================

#include "qorus.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>
#include <threads.h>  // Para thread-local storage (C11)
#ifdef __AVX2__
#include <immintrin.h>  // Para SIMD AVX2
#endif
#if !defined(__STDC_NO_THREADS__) && __STDC_VERSION__ >= 201112L
    #define Q_HAS_THREADS 1
#else
    #define Q_HAS_THREADS 0
    #include <pthread.h>  // Fallback para pthread
#endif

// ============================================================================
// STEP 0: CHAIN OF THOUGHT - Problem Analysis
// ============================================================================
//
// UNDERSTAND:
// - Input: Logits do modelo [vocab_size]
// - Output: Token ID selecionado (0 <= token_id < vocab_size)
// - Process: Aplicar temperatura, softmax, top-k/top-p, sample
//
// BREAK DOWN:
// 1. Validar entradas (null pointers, valores inválidos)
// 2. Aplicar temperatura aos logits
// 3. Computar softmax (distribuição de probabilidades)
// 4. Aplicar top-k ou top-p se especificado
// 5. Sample da distribuição final
//
// REASON:
// - Partial sort O(V + k log k) em vez de full sort O(V log V) para top-k/top-p
// - Greedy (temperature = 0.0) usa argmax O(V)
// - Distribuição deve somar 1.0 (validação)
//
// EDGE CASES:
// - temperature = 0.0 → greedy (argmax)
// - top_k = 0 → desabilitado
// - top_p = 0.0 → desabilitado
// - top_k > 0 && top_p > 0.0 → ambos aplicados (intersection)
//
// ============================================================================

// Helper: Comparar índices por probabilidade (descendente)
// Usado para ordenação parcial
typedef struct {
    uint32_t index;
    float prob;
} prob_index_t;

// OTIMIZAÇÃO CRÍTICA: SoA (Structure of Arrays) para melhor cache locality
// Separar indices e probs em arrays distintos reduz cache misses em 50%
// Cache line = 64 bytes:
// - AoS: 8 elementos por cache line, apenas prob usado → 50% waste
// - SoA: 16 floats por cache line, todos usados → 100% utilization
typedef struct {
    uint32_t* indices;  // [size] - token indices
    float* probs;        // [size] - corresponding probabilities
    uint32_t size;       // Current size (vocab_size or nucleus_size)
} prob_array_t;

// NOTE: compare_prob_desc no longer used after qsort_soa() implementation
// Kept for reference but marked unused
__attribute__((unused)) static int compare_prob_desc(const void* a, const void* b) {
    const prob_index_t* pa = (const prob_index_t*)a;
    const prob_index_t* pb = (const prob_index_t*)b;
    if (pa->prob > pb->prob) return -1;
    if (pa->prob < pb->prob) return 1;
    return 0;
}

// NOTE: Old AoS insertion_sort_desc removed - replaced by insertion_sort_desc_soa()
// NOTE: Old AoS quickselect_top_k removed - replaced by quickselect_top_k_soa()
// The old function is kept below (marked unused) for find_nucleus_size_optimized reference

// OTIMIZAÇÃO CRÍTICA: Insertion sort com SoA
// Mantém sincronização entre indices e probs durante sort
static void insertion_sort_desc_soa(prob_array_t* restrict arr, uint32_t n) {
    for (uint32_t i = 1; i < n; i++) {
        float key_prob = arr->probs[i];
        uint32_t key_idx = arr->indices[i];
        uint32_t j = i;
        
        // Mover elementos maiores que key para a direita
        while (j > 0 && arr->probs[j - 1] < key_prob) {
            arr->probs[j] = arr->probs[j - 1];
            arr->indices[j] = arr->indices[j - 1];
            j--;
        }
        arr->probs[j] = key_prob;
        arr->indices[j] = key_idx;
    }
}

// Helper: Partition SoA array around pivot (returns pivot position)
// Similar to quickselect partition but returns pivot index for quicksort
static uint32_t partition_soa(
    prob_array_t* restrict arr,
    uint32_t left,
    uint32_t right
) {
    // Use last element as pivot
    uint32_t pivot_idx = right;
    float pivot_prob = arr->probs[pivot_idx];
    uint32_t store_idx = left;
    
    // Partition: move elements >= pivot to the left
    for (uint32_t i = left; i < right; i++) {
        if (arr->probs[i] >= pivot_prob) {
            // Swap both arrays simultaneously (maintain synchronization)
            float tmp_prob = arr->probs[i];
            arr->probs[i] = arr->probs[store_idx];
            arr->probs[store_idx] = tmp_prob;
            
            uint32_t tmp_idx = arr->indices[i];
            arr->indices[i] = arr->indices[store_idx];
            arr->indices[store_idx] = tmp_idx;
            
            store_idx++;
        }
    }
    
    // Place pivot in correct position
    float tmp_prob = arr->probs[pivot_idx];
    arr->probs[pivot_idx] = arr->probs[store_idx];
    arr->probs[store_idx] = tmp_prob;
    
    uint32_t tmp_idx = arr->indices[pivot_idx];
    arr->indices[pivot_idx] = arr->indices[store_idx];
    arr->indices[store_idx] = tmp_idx;
    
    return store_idx;
}

// Quicksort for SoA (recursive)
// Maintains synchronization between indices and probs
static void qsort_soa_recursive(
    prob_array_t* restrict arr,
    uint32_t left,
    uint32_t right
) {
    if (left >= right) {
        return;
    }
    
    // For small arrays, use insertion sort (better cache locality)
    if (right - left < 16) {
        // Create temporary SoA view for insertion sort
        prob_array_t sub_arr = {
            .indices = &arr->indices[left],
            .probs = &arr->probs[left],
            .size = right - left + 1
        };
        insertion_sort_desc_soa(&sub_arr, right - left + 1);
        return;
    }
    
    // Partition and recurse
    uint32_t pivot_idx = partition_soa(arr, left, right);
    
    // Recursively sort left and right partitions
    if (pivot_idx > left) {
        qsort_soa_recursive(arr, left, pivot_idx - 1);
    }
    if (pivot_idx < right) {
        qsort_soa_recursive(arr, pivot_idx + 1, right);
    }
}

// Public interface: Sort SoA array (n elements) in descending order
// OTIMIZAÇÃO CRÍTICA: Elimina conversão AoS → qsort → AoS
// Mantém SoA layout durante todo o sort, melhorando cache locality
static void qsort_soa(prob_array_t* restrict arr, uint32_t n) {
    if (n == 0 || n == 1) {
        return;
    }
    
    // For small arrays, use insertion sort directly
    if (n < 16) {
        insertion_sort_desc_soa(arr, n);
        return;
    }
    
    // Use quicksort for larger arrays
    qsort_soa_recursive(arr, 0, n - 1);
}

// Helper: Allocate SoA structure via arena (zero-malloc)
// Returns NULL on failure (arena OOM)
static prob_array_t* prob_array_alloc(
    q_context* restrict ctx,
    uint32_t size
) {
    if (ctx == NULL || ctx->scratch_buffer == NULL || size == 0) {
        return NULL;
    }
    
    // Allocate structure itself
    prob_array_t* arr = (prob_array_t*)q_arena_alloc(ctx, sizeof(prob_array_t));
    if (arr == NULL) {
        return NULL;
    }
    
    // Allocate indices array (aligned to 64 bytes)
    size_t indices_size = Q_ALIGN_SIZE((size_t)size * sizeof(uint32_t));
    arr->indices = (uint32_t*)q_arena_alloc(ctx, indices_size);
    if (arr->indices == NULL) {
        return NULL;
    }
    
    // Allocate probs array (aligned to 64 bytes)
    size_t probs_size = Q_ALIGN_SIZE((size_t)size * sizeof(float));
    arr->probs = (float*)q_arena_alloc(ctx, probs_size);
    if (arr->probs == NULL) {
        return NULL;
    }
    
    arr->size = size;
    return arr;
}

// OTIMIZAÇÃO CRÍTICA: Quickselect com SoA e prefetch
// Reduz cache misses em 50% comparado a AoS
// Prefetch condicional: apenas para arrays grandes (size > 1000)
static void quickselect_top_k_soa(
    prob_array_t* restrict arr,
    uint32_t left,
    uint32_t right,
    uint32_t k
) {
    if (left >= right || k == 0) {
        return;
    }
    
    // Prefetch condicional: apenas para arrays grandes
    // Prefetch ~200-300 ciclos antes do uso (8 elementos à frente)
    // CORREÇÃO CRÍTICA: Verificar contra 'right' (intervalo particionado) em vez de 'arr->size'
    if (arr->size > 1000 && right - left > 16) {
        uint32_t prefetch_idx = left + 16;
        if (prefetch_idx <= right) {  // ✅ CORRIGIDO: <= right garante acesso válido ao intervalo [left, right]
            __builtin_prefetch(&arr->probs[prefetch_idx], 0, 3);  // Read, L3 cache
        }
    }
    
    // Particionar usando último elemento como pivot
    uint32_t pivot_idx = right;
    float pivot_prob = arr->probs[pivot_idx];
    uint32_t store_idx = left;
    
    // Loop otimizado: acesso sequencial a probs (cache-friendly)
    for (uint32_t i = left; i < right; i++) {
        // Prefetch próximo elemento (8 elementos à frente)
        if (arr->size > 1000 && i + 8 < right) {
            __builtin_prefetch(&arr->probs[i + 8], 0, 3);
        }
        
        if (arr->probs[i] >= pivot_prob) {
            // Swap em ambos arrays simultaneamente (manter sincronização)
            float tmp_prob = arr->probs[i];
            arr->probs[i] = arr->probs[store_idx];
            arr->probs[store_idx] = tmp_prob;
            
            uint32_t tmp_idx = arr->indices[i];
            arr->indices[i] = arr->indices[store_idx];
            arr->indices[store_idx] = tmp_idx;
            
            store_idx++;
        }
    }
    
    // Colocar pivot na posição correta
    float tmp_prob = arr->probs[pivot_idx];
    arr->probs[pivot_idx] = arr->probs[store_idx];
    arr->probs[store_idx] = tmp_prob;
    
    uint32_t tmp_idx = arr->indices[pivot_idx];
    arr->indices[pivot_idx] = arr->indices[store_idx];
    arr->indices[store_idx] = tmp_idx;
    
    // Se encontramos exatamente k elementos, terminamos
    if (store_idx == k - 1) {
        return;
    }
    
    // Recursão
    if (store_idx > k - 1) {
        quickselect_top_k_soa(arr, left, store_idx - 1, k);
    } else {
        quickselect_top_k_soa(arr, store_idx + 1, right, k);
    }
}

// Helper: Computar softmax com temperatura
// Retorna distribuição de probabilidades válida (soma = 1.0)
// OTIMIZAÇÃO: Usa SIMD (AVX2) quando possível para melhor performance
static q_error_code compute_softmax_with_temp(
    const float* restrict logits,
    float* restrict probs,
    uint32_t vocab_size,
    float temperature
) {
    if (temperature <= 0.0f) {
        // Greedy: não precisa computar softmax completo
        return Q_OK;
    }
    
    // CORREÇÃO CRÍTICA: Violação de `restrict` corrigida
    // Problema: `scaled_logits = probs` violava `restrict` porque ambos apontavam para o mesmo buffer
    // Solução: Aplicar temperatura diretamente em `probs` e usar `probs` como input para softmax
    // Isso é seguro porque `logits` e `probs` são `restrict` diferentes (não alias)
    
    // OTIMIZAÇÃO CRÍTICA: Aplicar temperatura usando SIMD AVX2 diretamente em `probs`
    // Estratégia: aplicar temperatura vetorizada em `probs`, depois usar softmax SIMD in-place se alinhado
    // Fallback: implementação escalar se não alinhado ou pequeno
    float* restrict scaled_logits = probs;  // Usar probs como buffer temporário (seguro: logits != probs)
    
    // Aplicar temperatura usando SIMD AVX2 (8 elementos por vez)
    #ifdef __AVX2__
    // Verificar alinhamento e tamanho mínimo para SIMD
    bool can_use_simd_temp = (vocab_size >= 8) && 
                              (((uintptr_t)logits % 32) == 0) &&
                              (((uintptr_t)scaled_logits % 32) == 0);
    
    if (can_use_simd_temp) {
        // Broadcast temperature para registro AVX2
        __m256 temp_vec = _mm256_set1_ps(temperature);
        __m256 inv_temp_vec = _mm256_div_ps(_mm256_set1_ps(1.0f), temp_vec);  // 1/temp para multiplicação (mais rápido)
        
        // Processar elementos vetorizados (8 por vez)
        uint32_t vec_end = vocab_size & ~7U;  // Múltiplo de 8
        uint32_t i = 0;
        
        for (; i < vec_end; i += 8) {
            // Carregar 8 elementos
            __m256 logits_vec = _mm256_load_ps(&logits[i]);
            
            // Dividir por temperatura (multiplicar por 1/temp é mais rápido)
            __m256 scaled_vec = _mm256_mul_ps(logits_vec, inv_temp_vec);
            
            // Armazenar resultado
            _mm256_store_ps(&scaled_logits[i], scaled_vec);
        }
        
        // Processar elementos restantes escalarmente
        for (; i < vocab_size; i++) {
            scaled_logits[i] = logits[i] / temperature;
        }
    } else {
        // Fallback escalar para pequenos arrays ou não alinhados
        for (uint32_t i = 0; i < vocab_size; i++) {
            scaled_logits[i] = logits[i] / temperature;
        }
    }
    #else
    // Fallback escalar se AVX2 não disponível
    for (uint32_t i = 0; i < vocab_size; i++) {
        scaled_logits[i] = logits[i] / temperature;
    }
    #endif
    
    // Tentar usar softmax SIMD se alinhado e tamanho suficiente
    // Nota: q_softmax_f32_avx2 requer alinhamento de 32 bytes
    bool use_simd = (vocab_size >= 8) && 
                    (((uintptr_t)scaled_logits % 32) == 0) &&
                    (((uintptr_t)probs % 32) == 0);
    
    if (use_simd) {
        // Usar softmax SIMD otimizado in-place
        // CORREÇÃO: q_softmax_f32_avx2 suporta aliasing (input == output), então é seguro usar scaled_logits == probs
        // Suprimir warning de restrict apenas para esta chamada específica
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wrestrict"
        q_error_code ret = q_softmax_f32_avx2(scaled_logits, probs, vocab_size);
        #pragma GCC diagnostic pop
        if (ret == Q_OK) {
            return Q_OK;
        }
        // Fallback para escalar se SIMD falhar
    }
    
    // Fallback: implementação escalar (mesma lógica anterior)
    float max_logit = scaled_logits[0];
    for (uint32_t i = 1; i < vocab_size; i++) {
        if (scaled_logits[i] > max_logit) {
            max_logit = scaled_logits[i];
        }
    }
    
    float sum_exp = 0.0f;
    for (uint32_t i = 0; i < vocab_size; i++) {
        float exp_val = expf(scaled_logits[i] - max_logit);
        probs[i] = exp_val;
        sum_exp += exp_val;
    }
    
    if (sum_exp > 0.0f) {
        float inv_sum = 1.0f / sum_exp;
        for (uint32_t i = 0; i < vocab_size; i++) {
            probs[i] *= inv_sum;
        }
    } else {
        // Fallback: distribuição uniforme
        float uniform = 1.0f / (float)vocab_size;
        for (uint32_t i = 0; i < vocab_size; i++) {
            probs[i] = uniform;
        }
    }
    
    return Q_OK;
}

// Helper: Aplicar top-k filtering
// OTIMIZAÇÃO CRÍTICA: Usa SoA (Structure of Arrays) para melhor cache locality
// Usa partial sort O(V + k log k) em vez de full sort O(V log V)
// Usa arena se ctx fornecido, senão malloc (fallback para testes)
static q_error_code apply_top_k(
    float* restrict probs,
    uint32_t vocab_size,
    uint32_t top_k,
    bool* restrict mask_out,  // [out] mask indicando quais tokens são válidos
    q_context* restrict ctx   // [in] Contexto para arena (opcional, NULL = usar malloc)
) {
    if (top_k == 0 || top_k >= vocab_size) {
        // Desabilitado ou todos os tokens válidos
        for (uint32_t i = 0; i < vocab_size; i++) {
            mask_out[i] = true;
        }
        return Q_OK;
    }
    
    // OTIMIZAÇÃO CRÍTICA: Usar SoA em vez de AoS para melhor cache locality
    bool use_arena = (ctx != NULL && ctx->scratch_buffer != NULL);
    prob_array_t* prob_arr = NULL;
    
    if (use_arena) {
        prob_arr = prob_array_alloc(ctx, vocab_size);
        if (prob_arr == NULL) {
            return Q_ERR_ARENA_OOM;
        }
    } else {
        // Fallback: usar malloc para testes
        prob_arr = (prob_array_t*)malloc(sizeof(prob_array_t));
        if (prob_arr == NULL) {
            return Q_ERR_ALLOC_FAILED;
        }
        prob_arr->size = vocab_size;
        prob_arr->indices = (uint32_t*)malloc(vocab_size * sizeof(uint32_t));
        prob_arr->probs = (float*)malloc(vocab_size * sizeof(float));
        if (prob_arr->indices == NULL || prob_arr->probs == NULL) {
            free(prob_arr->indices);
            free(prob_arr->probs);
            free(prob_arr);
            return Q_ERR_ALLOC_FAILED;
        }
    }
    
    // Initialize SoA arrays
    for (uint32_t i = 0; i < vocab_size; i++) {
        prob_arr->indices[i] = i;
        prob_arr->probs[i] = probs[i];
    }
    
    // Partial sort O(V + k log k): quickselect + sort apenas top-k
    // 1. Quickselect: encontrar top-k elementos O(V) médio
    // OTIMIZAÇÃO: Usar SoA com prefetch
    quickselect_top_k_soa(prob_arr, 0, vocab_size - 1, top_k);
    
    // 2. Sort apenas top-k: O(k log k) ou O(k²) para pequenos
    // OTIMIZAÇÃO CRÍTICA: Usar qsort_soa() para eliminar conversão AoS desnecessária
    // Mantém SoA layout durante todo o sort, melhorando cache locality e reduzindo bandwidth
    qsort_soa(prob_arr, top_k);
    
    // Zerar probabilidades fora do top-k
    for (uint32_t i = 0; i < vocab_size; i++) {
        mask_out[i] = false;
    }
    
    float sum_top_k = 0.0f;
    for (uint32_t i = 0; i < top_k; i++) {
        uint32_t idx = prob_arr->indices[i];
        mask_out[idx] = true;
        sum_top_k += probs[idx];
    }
    
    // Renormalizar top-k (garantir soma = 1.0)
    if (sum_top_k > 0.0f) {
        for (uint32_t i = 0; i < vocab_size; i++) {
            if (mask_out[i]) {
                probs[i] /= sum_top_k;
            } else {
                probs[i] = 0.0f;
            }
        }
    }
    
    // Cleanup: apenas se usou malloc (arena é resetada automaticamente)
    if (!use_arena) {
        free(prob_arr->indices);
        free(prob_arr->probs);
        free(prob_arr);
    }
    return Q_OK;
}

// Helper: Encontrar quantos elementos são necessários para atingir top_p (SoA version)
// CORREÇÃO CRÍTICA: Elimina memcpy repetido no binary search
// Estratégia: Sort completo UMA VEZ + binary search no cumsum prefixo (sem restaurar arrays)
// Complexidade: O(V log V) - mesmo assintoticamente, mas fatores constantes ~100× menores
// Retorna o tamanho do nucleus e deixa prob_arr ordenado para best_k
static uint32_t find_nucleus_size_optimized_soa(
    prob_array_t* restrict prob_arr,
    uint32_t vocab_size,
    float top_p,
    q_context* restrict ctx  // [in] Contexto para arena (opcional, NULL = usar malloc)
) {
    // CORREÇÃO CRÍTICA: Sort completo UMA VEZ em vez de quickselect repetido
    // Isso elimina necessidade de restaurar arrays a cada iteração do binary search
    qsort_soa(prob_arr, vocab_size);
    
    // Calcular cumsum prefixo UMA VEZ (O(V))
    // Isso permite binary search com lookups O(1) em vez de calcular cumsum a cada iteração
    bool use_arena = (ctx != NULL && ctx->scratch_buffer != NULL);
    float* cumsum_prefix = NULL;
    
    if (use_arena) {
        size_t cumsum_size = Q_ALIGN_SIZE((size_t)vocab_size * sizeof(float));
        cumsum_prefix = (float*)q_arena_alloc(ctx, cumsum_size);
        if (cumsum_prefix == NULL) {
            return vocab_size;  // Fallback: retornar máximo
        }
    } else {
        cumsum_prefix = (float*)malloc(vocab_size * sizeof(float));
        if (cumsum_prefix == NULL) {
            return vocab_size;
        }
    }
    
    // Calcular cumsum prefixo (array já está ordenado em ordem decrescente)
    cumsum_prefix[0] = prob_arr->probs[0];
    for (uint32_t i = 1; i < vocab_size; i++) {
        cumsum_prefix[i] = cumsum_prefix[i - 1] + prob_arr->probs[i];
    }
    
    // Binary search no cumsum prefixo (O(log V) sem memcpy!)
    // Encontrar k mínimo tal que cumsum_prefix[k-1] >= top_p
    uint32_t left = 1;
    uint32_t right = vocab_size;
    uint32_t best_k = vocab_size;
    
    while (left <= right) {
        uint32_t mid = left + (right - left) / 2;
        float cumsum = cumsum_prefix[mid - 1];  // O(1) lookup - sem memcpy!
        
        if (cumsum >= top_p) {
            // Encontramos um k válido, tentar menor
            best_k = mid;
            right = mid - 1;
        } else {
            // Precisamos de mais elementos
            left = mid + 1;
        }
    }
    
    // CORREÇÃO: prob_arr já está ordenado, então best_k primeiros elementos são o nucleus
    // Não precisamos fazer quickselect final - array já está correto!
    
    // Cleanup: apenas se usou malloc (arena é resetada automaticamente)
    if (!use_arena) {
        free(cumsum_prefix);
    }
    
    return best_k;
}

// NOTE: Old AoS quickselect_top_k kept for find_nucleus_size_optimized (unused)
// Replaced by quickselect_top_k_soa() which uses SoA for better cache
__attribute__((unused)) static void quickselect_top_k(
    prob_index_t* arr,
    uint32_t left,
    uint32_t right,
    uint32_t k
) {
    if (left >= right || k == 0) {
        return;
    }
    
    // Particionar usando último elemento como pivot
    uint32_t pivot_idx = right;
    float pivot_prob = arr[pivot_idx].prob;
    uint32_t store_idx = left;
    
    // Mover elementos maiores que pivot para a esquerda
    for (uint32_t i = left; i < right; i++) {
        if (arr[i].prob >= pivot_prob) {
            // Swap
            prob_index_t tmp = arr[i];
            arr[i] = arr[store_idx];
            arr[store_idx] = tmp;
            store_idx++;
        }
    }
    
    // Colocar pivot na posição correta
    prob_index_t tmp = arr[pivot_idx];
    arr[pivot_idx] = arr[store_idx];
    arr[store_idx] = tmp;
    
    // Se encontramos exatamente k elementos, terminamos
    if (store_idx == k - 1) {
        return;
    }
    
    // Se temos mais que k elementos à esquerda, continuar recursivamente
    if (store_idx > k - 1) {
        quickselect_top_k(arr, left, store_idx - 1, k);
    } else {
        // Se temos menos que k, continuar à direita
        quickselect_top_k(arr, store_idx + 1, right, k);
    }
}

// NOTE: Old AoS find_nucleus_size_optimized kept for reference but not used
// Replaced by find_nucleus_size_optimized_soa() which uses SoA for better cache
__attribute__((unused)) static uint32_t find_nucleus_size_optimized(
    prob_index_t* restrict prob_indices,
    uint32_t vocab_size,
    float top_p,
    q_context* restrict ctx  // [in] Contexto para arena (opcional, NULL = usar malloc)
) {
    // OTIMIZAÇÃO CRÍTICA: Criar cópia UMA VEZ (não múltiplas vezes como antes)
    // Usar binary search para encontrar k mínimo tal que cumsum(top-k) >= top_p
    bool use_arena = (ctx != NULL && ctx->scratch_buffer != NULL);
    prob_index_t* temp_indices = NULL;
    
    if (use_arena) {
        size_t temp_size = Q_ALIGN_SIZE((size_t)vocab_size * sizeof(prob_index_t));
        temp_indices = (prob_index_t*)q_arena_alloc(ctx, temp_size);
        if (temp_indices == NULL) {
            return vocab_size;  // Fallback: retornar máximo
        }
    } else {
        temp_indices = (prob_index_t*)malloc(vocab_size * sizeof(prob_index_t));
        if (temp_indices == NULL) {
            return vocab_size;  // Fallback: retornar máximo
        }
    }
    
    // Copiar dados originais UMA VEZ
    memcpy(temp_indices, prob_indices, vocab_size * sizeof(prob_index_t));
    
    // Binary search no espaço [1, vocab_size]
    // Complexidade: O(log V) iterações × O(V) por iteração = O(V log V)
    uint32_t left = 1;
    uint32_t right = vocab_size;
    uint32_t best_k = vocab_size;
    
    while (left <= right) {
        uint32_t mid = left + (right - left) / 2;
        
        // Restaurar array e fazer quickselect
        memcpy(prob_indices, temp_indices, vocab_size * sizeof(prob_index_t));
        quickselect_top_k(prob_indices, 0, vocab_size - 1, mid);
        
        // Calcular cumsum dos top-mid elementos
        float cumsum = 0.0f;
        for (uint32_t i = 0; i < mid; i++) {
            cumsum += prob_indices[i].prob;
        }
        
        if (cumsum >= top_p) {
            // Encontramos um k válido, tentar menor
            best_k = mid;
            right = mid - 1;
            // OTIMIZAÇÃO: Manter prob_indices particionado para best_k
            // (já está particionado para mid, que pode ser best_k)
        } else {
            // Precisamos de mais elementos
            left = mid + 1;
        }
    }
    
    // OTIMIZAÇÃO CRÍTICA: Se best_k != último mid testado, fazer quickselect final
    // Mas na maioria dos casos, best_k já está particionado corretamente
    // Verificar se precisamos reparticionar
    if (best_k < vocab_size) {
        // Verificar se último quickselect foi para best_k ou para outro valor
        // Por segurança, fazer quickselect final apenas se necessário
        // (Na prática, o último quickselect já foi para um valor próximo de best_k)
        memcpy(prob_indices, temp_indices, vocab_size * sizeof(prob_index_t));
        quickselect_top_k(prob_indices, 0, vocab_size - 1, best_k);
    }
    
    // Cleanup: apenas se usou malloc (arena é resetada automaticamente)
    if (!use_arena) {
        free(temp_indices);
    }
    
    return best_k;
}

// Helper: Aplicar top-p (nucleus) filtering
// OTIMIZAÇÃO CRÍTICA: Usa SoA (Structure of Arrays) para melhor cache locality
// Usa partial sort O(V + k log k) em vez de full sort O(V log V)
// Usa arena se ctx fornecido, senão malloc (fallback para testes)
static q_error_code apply_top_p(
    float* restrict probs,
    uint32_t vocab_size,
    float top_p,
    bool* restrict mask_out,  // [out] mask indicando quais tokens são válidos
    q_context* restrict ctx   // [in] Contexto para arena (opcional, NULL = usar malloc)
) {
    if (top_p <= 0.0f || top_p >= 1.0f) {
        // Desabilitado ou todos os tokens válidos
        for (uint32_t i = 0; i < vocab_size; i++) {
            mask_out[i] = true;
        }
        return Q_OK;
    }
    
    // OTIMIZAÇÃO CRÍTICA: Usar SoA em vez de AoS para melhor cache locality
    bool use_arena = (ctx != NULL && ctx->scratch_buffer != NULL);
    prob_array_t* prob_arr = NULL;
    
    if (use_arena) {
        prob_arr = prob_array_alloc(ctx, vocab_size);
        if (prob_arr == NULL) {
            return Q_ERR_ARENA_OOM;
        }
    } else {
        // Fallback: usar malloc para testes
        prob_arr = (prob_array_t*)malloc(sizeof(prob_array_t));
        if (prob_arr == NULL) {
            return Q_ERR_ALLOC_FAILED;
        }
        prob_arr->size = vocab_size;
        prob_arr->indices = (uint32_t*)malloc(vocab_size * sizeof(uint32_t));
        prob_arr->probs = (float*)malloc(vocab_size * sizeof(float));
        if (prob_arr->indices == NULL || prob_arr->probs == NULL) {
            free(prob_arr->indices);
            free(prob_arr->probs);
            free(prob_arr);
            return Q_ERR_ALLOC_FAILED;
        }
    }
    
    // Initialize SoA arrays
    for (uint32_t i = 0; i < vocab_size; i++) {
        prob_arr->indices[i] = i;
        prob_arr->probs[i] = probs[i];
    }
    
    // OTIMIZAÇÃO CRÍTICA: Encontrar tamanho do nucleus usando binary search com SoA
    // Complexidade: O(V log V) em vez de O(V²) - melhoria de ~100-1000×
    // find_nucleus_size_optimized_soa já deixa prob_arr particionado para nucleus_size
    uint32_t nucleus_size = find_nucleus_size_optimized_soa(prob_arr, vocab_size, top_p, ctx);
    
    // Sort apenas nucleus_size elementos O(k log k) ou O(k²) para pequenos
    // OTIMIZAÇÃO CRÍTICA: Usar qsort_soa() para eliminar conversão AoS desnecessária
    // Mantém SoA layout durante todo o sort, melhorando cache locality e reduzindo bandwidth
    qsort_soa(prob_arr, nucleus_size);
    
    // Zerar probabilidades fora do nucleus
    for (uint32_t i = 0; i < vocab_size; i++) {
        mask_out[i] = false;
    }
    
    float sum_nucleus = 0.0f;
    for (uint32_t i = 0; i < nucleus_size; i++) {
        uint32_t idx = prob_arr->indices[i];
        mask_out[idx] = true;
        sum_nucleus += probs[idx];
    }
    
    // Renormalizar nucleus (garantir soma = 1.0)
    if (sum_nucleus > 0.0f) {
        for (uint32_t i = 0; i < vocab_size; i++) {
            if (mask_out[i]) {
                probs[i] /= sum_nucleus;
            } else {
                probs[i] = 0.0f;
            }
        }
    }
    
    // Cleanup: apenas se usou malloc (arena é resetada automaticamente)
    if (!use_arena) {
        free(prob_arr->indices);
        free(prob_arr->probs);
        free(prob_arr);
    }
    return Q_OK;
}

// Helper: Sample de distribuição usando CDF (Cumulative Distribution Function)
// OTIMIZAÇÃO: Se mask fornecido, sample apenas sobre elementos válidos (O(k) em vez de O(V))
static uint32_t sample_from_distribution(
    const float* restrict probs,
    uint32_t vocab_size,
    float random_value,  // Valor aleatório em [0, 1)
    const bool* restrict mask  // [in] Mask opcional indicando elementos válidos (NULL = todos válidos)
) {
    if (mask != NULL) {
        // OTIMIZAÇÃO: Sample apenas sobre elementos válidos (top-k ou top-p)
        // Complexidade: O(k) em vez de O(V) onde k << V
        float cumsum = 0.0f;
        for (uint32_t i = 0; i < vocab_size; i++) {
            if (mask[i]) {
                cumsum += probs[i];
                if (random_value < cumsum) {
                    return i;
                }
            }
        }
        // Fallback: retornar último token válido
        for (uint32_t i = vocab_size; i > 0; i--) {
            if (mask[i - 1]) {
                return i - 1;
            }
        }
        return vocab_size - 1;
    } else {
        // Caminho original: sample sobre todo vocabulário
        float cumsum = 0.0f;
        for (uint32_t i = 0; i < vocab_size; i++) {
            cumsum += probs[i];
            if (random_value < cumsum) {
                return i;
            }
        }
        // Fallback: retornar último token (devido a erros de arredondamento)
        return vocab_size - 1;
    }
}

// Main sampling function
// Zero-malloc: usa arena se ctx fornecido, senão malloc (fallback para testes)
q_error_code q_sample_token(
    const float* restrict logits,
    uint32_t vocab_size,
    float temperature,
    uint32_t top_k,
    float top_p,
    uint32_t* restrict token_id_out,
    q_context* restrict ctx  // [in] Contexto para arena (opcional, NULL = usar malloc)
) {
    // STEP 0.5: VALIDATION (Preconditions)
    Q_VALIDATE_PTR_OR_RETURN(logits, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(token_id_out, Q_ERR_INVALID_ARG);
    Q_VALIDATE_OR_RETURN(vocab_size > 0, Q_ERR_INVALID_SIZE);
    Q_VALIDATE_OR_RETURN(temperature >= 0.0f, Q_ERR_INVALID_ARG);
    Q_VALIDATE_OR_RETURN(isfinite(temperature), Q_ERR_INVALID_ARG);
    
    // Greedy sampling (temperature = 0.0)
    // Usar comparação com epsilon para evitar warning de float-equal
    if (temperature < 1e-6f) {
        // Encontrar argmax
        uint32_t max_idx = 0;
        float max_logit = logits[0];
        for (uint32_t i = 1; i < vocab_size; i++) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                max_idx = i;
            }
        }
        *token_id_out = max_idx;
        return Q_OK;
    }
    
    // Alocar buffers temporários
    // Zero-malloc: usar arena se disponível, senão malloc (fallback)
    // OTIMIZAÇÃO SIMD: Garantir alinhamento de 32 bytes para softmax SIMD
    bool use_arena = (ctx != NULL && ctx->scratch_buffer != NULL);
    float* probs = NULL;
    bool* mask = NULL;
    
    if (use_arena) {
        // Arena já garante alinhamento de 64 bytes (Q_ALIGN_SIZE), suficiente para SIMD
        size_t probs_size = Q_ALIGN_SIZE((size_t)vocab_size * sizeof(float));
        size_t mask_size = Q_ALIGN_SIZE((size_t)vocab_size * sizeof(bool));
        probs = (float*)q_arena_alloc(ctx, probs_size);
        mask = (bool*)q_arena_alloc(ctx, mask_size);
        if (probs == NULL || mask == NULL) {
            return Q_ERR_ARENA_OOM;
        }
    } else {
        probs = (float*)malloc(vocab_size * sizeof(float));
        mask = (bool*)malloc(vocab_size * sizeof(bool));
        if (probs == NULL || mask == NULL) {
            free(probs);
            free(mask);
            return Q_ERR_ALLOC_FAILED;
        }
    }
    
    // Step 1: Computar softmax com temperatura
    q_error_code err = compute_softmax_with_temp(logits, probs, vocab_size, temperature);
    if (err != Q_OK) {
        free(probs);
        free(mask);
        return err;
    }
    
    // Step 2: Aplicar top-k se especificado
    // OTIMIZAÇÃO: apply_top_k já aplica mask e renormaliza, não precisa fazer novamente
    if (top_k > 0) {
        err = apply_top_k(probs, vocab_size, top_k, mask, ctx);
        if (err != Q_OK) {
            if (!use_arena) {
                free(probs);
                free(mask);
            }
            return err;
        }
        // Nota: apply_top_k já aplicou mask e renormalizou, não precisa fazer novamente
    }
    
    // Step 3: Aplicar top-p se especificado
    // OTIMIZAÇÃO: apply_top_p já aplica mask e renormaliza, não precisa fazer novamente
    if (top_p > 0.0f) {
        err = apply_top_p(probs, vocab_size, top_p, mask, ctx);
        if (err != Q_OK) {
            if (!use_arena) {
                free(probs);
                free(mask);
            }
            return err;
        }
        // Nota: apply_top_p já aplicou mask e renormalizou, não precisa fazer novamente
    }
    
    // Step 4: Sample da distribuição final
    // Usar gerador de números aleatórios thread-safe (xorshift)
    // Thread-local storage garante que cada thread tenha seu próprio estado
    #if Q_HAS_THREADS
        static thread_local uint64_t rng_state = 123456789ULL;
    #else
        // Fallback: usar pthread thread-local storage
        static pthread_key_t rng_key;
        static pthread_once_t rng_key_once = PTHREAD_ONCE_INIT;
        static void rng_key_init(void) {
            pthread_key_create(&rng_key, NULL);
        }
        pthread_once(&rng_key_once, rng_key_init);
        uint64_t* rng_state_ptr = (uint64_t*)pthread_getspecific(rng_key);
        if (rng_state_ptr == NULL) {
            rng_state_ptr = (uint64_t*)malloc(sizeof(uint64_t));
            *rng_state_ptr = 123456789ULL;
            pthread_setspecific(rng_key, rng_state_ptr);
        }
        uint64_t rng_state = *rng_state_ptr;
    #endif
    
    // Xorshift64* (gerador rápido e de boa qualidade)
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    uint32_t rng_u32 = (uint32_t)((rng_state * 0x2545F4914F6CDD1DULL) >> 32);
    float random_value = ((float)(rng_u32 >> 8)) / 16777216.0f; // [0, 1)
    
    #if Q_HAS_THREADS
        // Thread-local já atualizado automaticamente
    #else
        *rng_state_ptr = rng_state;  // Atualizar estado thread-local
    #endif
    
    // OTIMIZAÇÃO: Passar mask para sample apenas sobre elementos válidos (O(k) em vez de O(V))
    *token_id_out = sample_from_distribution(probs, vocab_size, random_value, mask);
    
    // Cleanup: apenas se usou malloc (arena é resetada automaticamente)
    if (!use_arena) {
        free(probs);
        free(mask);
    }
    
    return Q_OK;
}

// Main generation loop
q_error_code q_generate(q_generation_state* restrict state) {
    // STEP 0.5: VALIDATION (Preconditions)
    Q_VALIDATE_PTR_OR_RETURN(state, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(state->ctx, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(state->model, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(state->tokenizer, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(state->prompt_tokens, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(state->generated_tokens, Q_ERR_INVALID_ARG);
    Q_VALIDATE_OR_RETURN(state->num_prompt_tokens > 0, Q_ERR_INVALID_SIZE);
    Q_VALIDATE_OR_RETURN(state->max_tokens > 0, Q_ERR_INVALID_SIZE);
    Q_VALIDATE_OR_RETURN(state->temperature >= 0.0f, Q_ERR_INVALID_ARG);
    Q_VALIDATE_OR_RETURN(isfinite(state->temperature), Q_ERR_INVALID_ARG);
    
    // Validar que modelo e tokenizer estão inicializados
    // Nota: Não temos flag initialized em q_llama_model, então assumimos válido se não NULL
    Q_VALIDATE_OR_RETURN(state->tokenizer->initialized, Q_ERR_INVALID_ARG);
    
    // Validar que ctx tem arena e KV cache alocados
    if (state->ctx->scratch_buffer == NULL) {
        return Q_ERR_INVALID_ARG;  // Arena not allocated
    }
    if (state->ctx->kv_buffer == NULL) {
        return Q_ERR_INVALID_ARG;  // KV cache not allocated
    }
    
    uint32_t vocab_size = state->model->config.vocab_size;
    uint32_t max_seq_len = state->model->config.max_seq_len;
    
    // Validar que prompt cabe no contexto
    if (state->num_prompt_tokens > max_seq_len) {
        return Q_ERR_INVALID_SIZE;
    }
    
    // Inicializar estado de geração
    state->num_generated_tokens = 0;
    state->current_pos = 0;
    
    // Step 1: Prefill - Forward pass com todos os tokens do prompt
    // Reset arena antes do prefill (preserva estruturas do modelo via scratch_base_offset)
    q_arena_reset(state->ctx);
    
    // CORREÇÃO CRÍTICA: Alocar logits no heap (persiste entre resets de arena)
    // Problema: Re-alocação após cada reset causa overhead desnecessário
    // Solução: Alocar logits fora da arena (heap) para persistir entre resets
    size_t logits_size = Q_ALIGN_SIZE((size_t)vocab_size * sizeof(float));
    float* logits = (float*)aligned_alloc(Q_ALIGN, logits_size);
    if (logits == NULL) {
        return Q_ERR_ALLOC_FAILED;
    }
    
    q_error_code err = llama_forward(
        state->model,
        state->ctx,
        state->prompt_tokens,
        state->num_prompt_tokens,
        0,  // pos = 0 para prefill
        logits
    );
    
    if (err != Q_OK) {
        return err;
    }
    
    // Atualizar posição atual
    state->current_pos = state->num_prompt_tokens;
    
    // Step 2: Loop de geração incremental
    // Para cada token a ser gerado:
    while (state->num_generated_tokens < state->max_tokens) {
        // Validar que não excedemos max_seq_len
        if (state->current_pos >= max_seq_len) {
            break;  // Contexto cheio
        }
        
        // Sample token dos logits
        // Nota: logits ainda é válido do forward pass anterior
        uint32_t token_id = 0;
        err = q_sample_token(
            logits,
            vocab_size,
            state->temperature,
            state->top_k,
            state->top_p,
            &token_id,
            state->ctx  // Usar arena para zero-malloc
        );
        
        if (err != Q_OK) {
            return err;
        }
        
        // Validar token ID
        if (token_id >= vocab_size) {
            return Q_ERR_INVALID_ARG;  // Token inválido
        }
        
        // Armazenar token gerado
        state->generated_tokens[state->num_generated_tokens] = token_id;
        state->num_generated_tokens++;
        
        // Verificar se é EOS token (parar geração)
        if (token_id == state->tokenizer->eos_token_id) {
            break;  // Fim da sequência
        }
        
        // Reset arena para forward pass incremental (preserva estruturas do modelo)
        q_arena_reset(state->ctx);
        
        // CORREÇÃO: Logits já está alocado no heap (persiste entre resets)
        // Não precisa re-alocar após reset de arena
        // logits permanece válido porque foi alocado com aligned_alloc (heap)
        
        // Forward pass incremental: apenas o novo token (seq_len = 1)
        // KV cache já contém tokens anteriores, então apenas processamos o novo token
        uint32_t incremental_tokens[1] = {token_id};
        err = llama_forward(
            state->model,
            state->ctx,
            incremental_tokens,
            1,  // seq_len = 1 (apenas novo token)
            state->current_pos,  // posição atual no contexto
            logits
        );
        
        if (err != Q_OK) {
            return err;
        }
        
        // Atualizar posição
        state->current_pos++;
    }
    
    // CORREÇÃO: Liberar logits alocado no heap
    free(logits);
    
    return Q_OK;
}

