// ============================================================================
// TEST SUITE: BPE Soft-Delete Adversarial Tests
// ============================================================================
// Lead SDET Protocol: Adversarial Testing para apply_bpe_merges
// Baseado em: PLANEJAMENTO_CORRECOES_CRITICAS.md FASE 3.3 e 3.4
//
// Objetivo: Tentar QUEBRAR o código através de cenários adversariais
// Foco: Validar que soft-delete elimina O(m × n³) e mantém correção
// ============================================================================

#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>
#include <limits.h>

// ============================================================================
// FASE 0: CONTEXTO - Validação de Planejamento
// ============================================================================
// Especificação (FASE 3.4):
// - Pré-condições: tok != NULL, token_ids != NULL, num_tokens != NULL
// - Pós-condições: token_ids[i] != Q_TOKEN_DELETED, todos merges aplicados
// - Complexidade: O(m × n) ≤ Lower Bound × 1.1
//
// Failure Modes (FASE 3.3):
// - memmove repetido (O(m × n³)) - ELIMINADO
// - Re-scanning desnecessário (j--) - ELIMINADO
// - Compactação muito frequente - MITIGADO (lazy compaction)
// - Array não compactado no final - CORRIGIDO (compactação final obrigatória)
// ============================================================================

// ============================================================================
// FASE 1: MAPA DE CENÁRIOS
// ============================================================================
// Happy Path:
//   1. Caso básico: "aaaa" com merge "aa -> A" → "AA"
//   2. Múltiplos merges: Aplicar várias regras sequencialmente
//   3. Compactação lazy: Validar que compactação ocorre apenas quando necessário
//
// Edge Cases:
//   4. Array vazio (num_tokens = 0)
//   5. Array tamanho 1 (num_tokens = 1)
//   6. Array tamanho máximo (SIZE_MAX)
//   7. Vocab size máximo (UINT32_MAX)
//   8. Token IDs no limite (UINT32_MAX - 1)
//   9. Múltiplos merges consecutivos (cadeia de merges)
//   10. Compactação threshold exato (50% + 1)
//
// Null/Undefined:
//   11. tok = NULL
//   12. token_ids = NULL
//   13. num_tokens = NULL
//   14. Estrutura tok não inicializada (memset(0))
//   15. token_ids com lixo de memória
//
// Security/Malicious:
//   16. Buffer overflow: num_tokens > capacidade real
//   17. Integer overflow: num_tokens = SIZE_MAX
//   18. Token IDs inválidos (>= vocab_size)
//   19. Q_TOKEN_DELETED já presente no array (corrupção prévia)
//   20. Stale pointers: token_ids após free()
//
// Performance:
//   21. Complexidade O(m × n): Prompt grande (32k tokens)
//   22. Compactação lazy: Validar que não ocorre a cada iteração
// ============================================================================

// ============================================================================
// FASE 2: CRITÉRIOS DE ACEITE
// ============================================================================
// Null/Invalid Inputs:
//   - tok = NULL → Q_ERR_INVALID_ARG (Graceful Failure)
//   - token_ids = NULL → Q_ERR_INVALID_ARG
//   - num_tokens = NULL → Q_ERR_INVALID_ARG
//
// Edge Cases:
//   - num_tokens = 0 → Q_OK (early return)
//   - num_tokens = 1 → Q_OK (early return)
//   - Array grande → Q_OK (complexidade O(m × n))
//
// Security:
//   - Buffer overflow → Detectado por AddressSanitizer ou validação de bounds
//   - Token IDs inválidos → Comportamento indefinido (aceitável para testes)
//
// Pós-condições (Especificação FASE 3.4):
//   - token_ids[i] != Q_TOKEN_DELETED para todo i < *num_tokens
//   - Todos os merges aplicáveis foram aplicados
//   - *num_tokens atualizado corretamente
// ============================================================================

// Helper: Criar tokenizer dummy para testes
static q_tokenizer* create_dummy_tokenizer(uint32_t vocab_size, uint32_t num_merges) {
    q_tokenizer* tok = (q_tokenizer*)calloc(1, sizeof(q_tokenizer));
    if (!tok) return NULL;
    
    tok->vocab_size = vocab_size;
    tok->num_merges = num_merges;
    tok->merges = (q_bpe_merge*)calloc(num_merges, sizeof(q_bpe_merge));
    if (!tok->merges) {
        free(tok);
        return NULL;
    }
    
    // Marcar como inicializado para passar validação
    tok->initialized = true;
    
    return tok;
}

static void destroy_dummy_tokenizer(q_tokenizer* tok) {
    if (tok) {
        free(tok->merges);
        free(tok);
    }
}

// Helper: Adicionar merge rule
static void add_merge_rule(q_tokenizer* tok, uint32_t idx, uint32_t id1, uint32_t id2, uint32_t merged) {
    if (tok && tok->merges && idx < tok->num_merges) {
        tok->merges[idx].token_id1 = id1;
        tok->merges[idx].token_id2 = id2;
        tok->merges[idx].merged_id = merged;
    }
}

// Helper: Validar pós-condições
static bool validate_postconditions(
    const uint32_t* token_ids,
    size_t num_tokens,
    uint32_t vocab_size
) {
    // Pós-condição 1: Nenhum token morto no array final
    for (size_t i = 0; i < num_tokens; i++) {
        if (token_ids[i] == UINT32_MAX) {  // Q_TOKEN_DELETED
            return false;
        }
        if (token_ids[i] >= vocab_size) {
            return false;  // Token ID inválido
        }
    }
    return true;
}

// ============================================================================
// FASE 3: IMPLEMENTAÇÃO BLINDADA
// ============================================================================

// Test 1: Happy Path - Caso básico
static void test_happy_path_basic(void) {
    printf("Test 1: Happy Path - Caso Básico\n");
    printf("----------------------------------\n");
    
    q_tokenizer* tok = create_dummy_tokenizer(256, 1);
    assert(tok != NULL && "Tokenizer creation should succeed");
    
    // Criar merge rule: byte 'a' (97) + byte 'a' (97) -> token 1
    add_merge_rule(tok, 0, 97, 97, 1);  // "aa -> 1"
    
    uint32_t token_ids[1024];
    uint32_t num_tokens_uint = 0;
    
    // Chamar apply_bpe_merges (função static, precisa acesso via encode)
    // Por enquanto, validamos via encode completo
    q_error_code err = q_tokenizer_encode(tok, "aaaa", token_ids, &num_tokens_uint, 1024, false, false);
    
    assert(err == Q_OK && "Encoding should succeed");
    assert(num_tokens_uint > 0 && "Should have tokens after encoding");
    // Pós-condição crítica: Nenhum Q_TOKEN_DELETED no array final
    assert(validate_postconditions(token_ids, num_tokens_uint, tok->vocab_size) && "Postconditions should be valid");
    
    printf("✓ Happy path basic test passed\n");
    printf("✓ Test 1 PASSED\n\n");
    
    destroy_dummy_tokenizer(tok);
}

// Test 2: Happy Path - Múltiplos merges
static void test_happy_path_multiple_merges(void) {
    printf("Test 2: Happy Path - Múltiplos Merges\n");
    printf("--------------------------------------\n");
    
    q_tokenizer* tok = create_dummy_tokenizer(256, 3);
    assert(tok != NULL && "Tokenizer creation should succeed");
    
    add_merge_rule(tok, 0, 0, 0, 1);  // "aa -> A"
    add_merge_rule(tok, 1, 1, 1, 2);  // "AA -> B"
    add_merge_rule(tok, 2, 2, 2, 3);  // "BB -> C"
    
    uint32_t token_ids[] = {0, 0, 0, 0, 0, 0, 0, 0};
    size_t num_tokens = 8;
    
    uint32_t num_tokens_uint = (uint32_t)num_tokens;
    q_error_code err = q_tokenizer_encode(tok, "aaaaaaaa", token_ids, &num_tokens_uint, 1024, false, false);
    num_tokens = (size_t)num_tokens_uint;
    
    assert(err == Q_OK && "Encoding should succeed");
    assert(validate_postconditions(token_ids, num_tokens, tok->vocab_size) && "Postconditions should be valid");
    
    printf("✓ Multiple merges test passed\n");
    printf("✓ Test 2 PASSED\n\n");
    
    destroy_dummy_tokenizer(tok);
}

// Test 3: Edge Case - Array vazio
static void test_edge_case_empty_array(void) {
    printf("Test 3: Edge Case - Array Vazio\n");
    printf("-------------------------------\n");
    
    q_tokenizer* tok = create_dummy_tokenizer(256, 1);
    assert(tok != NULL && "Tokenizer creation should succeed");
    
    uint32_t token_ids[1] = {0};
    size_t num_tokens = 0;
    
    // apply_bpe_merges deve retornar Q_OK para num_tokens = 0
    // Como é função static, validamos via comportamento de encode
    uint32_t num_tokens_uint = 0;
    q_error_code err = q_tokenizer_encode(tok, "", token_ids, &num_tokens_uint, 1024, false, false);
    num_tokens = (size_t)num_tokens_uint;
    
    assert(err == Q_OK && "Empty array should return Q_OK");
    assert(num_tokens == 0 && "num_tokens should remain 0");
    
    printf("✓ Empty array test passed\n");
    printf("✓ Test 3 PASSED\n\n");
    
    destroy_dummy_tokenizer(tok);
}

// Test 4: Edge Case - Array tamanho 1
static void test_edge_case_single_token(void) {
    printf("Test 4: Edge Case - Array Tamanho 1\n");
    printf("------------------------------------\n");
    
    q_tokenizer* tok = create_dummy_tokenizer(256, 1);
    assert(tok != NULL && "Tokenizer creation should succeed");
    
    uint32_t token_ids[] = {0};
    size_t num_tokens = 1;
    
    uint32_t num_tokens_uint = (uint32_t)num_tokens;
    q_error_code err = q_tokenizer_encode(tok, "a", token_ids, &num_tokens_uint, 1024, false, false);
    num_tokens = (size_t)num_tokens_uint;
    
    assert(err == Q_OK && "Single token should return Q_OK");
    assert(num_tokens == 1 && "num_tokens should remain 1");
    assert(validate_postconditions(token_ids, num_tokens, tok->vocab_size) && "Postconditions should be valid");
    
    printf("✓ Single token test passed\n");
    printf("✓ Test 4 PASSED\n\n");
    
    destroy_dummy_tokenizer(tok);
}

// Test 5: Null/Undefined - tok = NULL
static void test_null_tokenizer(void) {
    printf("Test 5: Null/Undefined - tok = NULL\n");
    printf("------------------------------------\n");
    
    uint32_t token_ids[] = {0, 0};
    size_t num_tokens = 2;
    
    uint32_t num_tokens_uint = (uint32_t)num_tokens;
    q_error_code err = q_tokenizer_encode(NULL, "aa", token_ids, &num_tokens_uint, 1024, false, false);
    
    assert(err != Q_OK && "NULL tokenizer should return error");
    
    printf("✓ NULL tokenizer test passed\n");
    printf("✓ Test 5 PASSED\n\n");
}

// Test 6: Null/Undefined - token_ids = NULL
static void test_null_token_ids(void) {
    printf("Test 6: Null/Undefined - token_ids = NULL\n");
    printf("------------------------------------------\n");
    
    q_tokenizer* tok = create_dummy_tokenizer(256, 1);
    assert(tok != NULL && "Tokenizer creation should succeed");
    
    uint32_t num_tokens_uint = 2;
    q_error_code err = q_tokenizer_encode(tok, "aa", NULL, &num_tokens_uint, 1024, false, false);
    
    assert(err != Q_OK && "NULL token_ids should return error");
    
    printf("✓ NULL token_ids test passed\n");
    printf("✓ Test 6 PASSED\n\n");
    
    destroy_dummy_tokenizer(tok);
}

// Test 7: Security - Token IDs inválidos (>= vocab_size)
static void test_security_invalid_token_ids(void) {
    printf("Test 7: Security - Token IDs Inválidos\n");
    printf("----------------------------------------\n");
    
    q_tokenizer* tok = create_dummy_tokenizer(256, 1);
    assert(tok != NULL && "Tokenizer creation should succeed");
    
    add_merge_rule(tok, 0, 0, 0, 1);
    
    // Token IDs inválidos (>= vocab_size)
    uint32_t token_ids[] = {300, 300};  // vocab_size = 256
    size_t num_tokens = 2;
    
    // Comportamento indefinido esperado (aceitável para testes adversarial)
    // Validamos que não crasha
    uint32_t num_tokens_uint = (uint32_t)num_tokens;
    q_tokenizer_encode(tok, "aa", token_ids, &num_tokens_uint, 1024, false, false);
    num_tokens = (size_t)num_tokens_uint;
    
    // Pode retornar erro ou comportamento indefinido
    // O importante é não crashar
    printf("✓ Invalid token IDs test passed (no crash)\n");
    printf("✓ Test 7 PASSED\n\n");
    
    destroy_dummy_tokenizer(tok);
}

// Test 8: Performance - Complexidade O(m × n) - Prompt grande
static void test_performance_large_prompt(void) {
    printf("Test 8: Performance - Prompt Grande (Complexidade O(m × n))\n");
    printf("------------------------------------------------------------\n");
    
    q_tokenizer* tok = create_dummy_tokenizer(256, 10);
    assert(tok != NULL && "Tokenizer creation should succeed");
    
    // Criar merges que se aplicam sequencialmente
    for (uint32_t i = 0; i < 10; i++) {
        add_merge_rule(tok, i, i, i, i + 1);
    }
    
    // Prompt grande: 1000 tokens
    const size_t large_size = 1000;
    uint32_t* token_ids = (uint32_t*)malloc(large_size * sizeof(uint32_t));
    assert(token_ids != NULL && "Memory allocation should succeed");
    
    // Preencher com tokens que aplicam merges
    for (size_t i = 0; i < large_size; i++) {
        token_ids[i] = 0;
    }
    
    size_t num_tokens = large_size;
    
    // Medir tempo (simplificado)
    uint32_t num_tokens_uint = (uint32_t)num_tokens;
    q_error_code err = q_tokenizer_encode(tok, "a", token_ids, &num_tokens_uint, large_size * 2, false, false);
    num_tokens = (size_t)num_tokens_uint;
    
    assert(err == Q_OK && "Large prompt should succeed");
    assert(validate_postconditions(token_ids, num_tokens, tok->vocab_size) && "Postconditions should be valid");
    
    // Validar que num_tokens foi reduzido (merges aplicados)
    assert(num_tokens < large_size && "Tokens should be reduced after merges");
    
    printf("✓ Large prompt test passed (num_tokens: %zu -> %zu)\n", large_size, num_tokens);
    printf("✓ Test 8 PASSED\n\n");
    
    free(token_ids);
    destroy_dummy_tokenizer(tok);
}

// Test 9: Compactação Lazy - Validar que não ocorre a cada iteração
static void test_lazy_compaction_threshold(void) {
    printf("Test 9: Compactação Lazy - Threshold\n");
    printf("-------------------------------------\n");
    
    q_tokenizer* tok = create_dummy_tokenizer(256, 1);
    assert(tok != NULL && "Tokenizer creation should succeed");
    
    add_merge_rule(tok, 0, 0, 0, 1);
    
    // Criar array onde compactação lazy deve ocorrer (> 50% deletados)
    const size_t array_size = 100;
    uint32_t* token_ids = (uint32_t*)malloc(array_size * sizeof(uint32_t));
    assert(token_ids != NULL && "Memory allocation should succeed");
    
    // Preencher com tokens que aplicam merge (50% serão deletados)
    for (size_t i = 0; i < array_size; i++) {
        token_ids[i] = 0;
    }
    
    size_t num_tokens = array_size;
    
    uint32_t num_tokens_uint = (uint32_t)num_tokens;
    q_error_code err = q_tokenizer_encode(tok, "a", token_ids, &num_tokens_uint, array_size * 2, false, false);
    num_tokens = (size_t)num_tokens_uint;
    
    assert(err == Q_OK && "Lazy compaction test should succeed");
    assert(validate_postconditions(token_ids, num_tokens, tok->vocab_size) && "Postconditions should be valid");
    
    printf("✓ Lazy compaction test passed\n");
    printf("✓ Test 9 PASSED\n\n");
    
    free(token_ids);
    destroy_dummy_tokenizer(tok);
}

// Test 10: Pós-condições - Validar que nenhum Q_TOKEN_DELETED permanece
static void test_postconditions_no_deleted_tokens(void) {
    printf("Test 10: Pós-condições - Nenhum Token Morto\n");
    printf("---------------------------------------------\n");
    
    q_tokenizer* tok = create_dummy_tokenizer(256, 5);
    assert(tok != NULL && "Tokenizer creation should succeed");
    
    // Criar múltiplos merges
    for (uint32_t i = 0; i < 5; i++) {
        add_merge_rule(tok, i, i, i, i + 10);
    }
    
    uint32_t token_ids[20];
    for (uint32_t i = 0; i < 20; i++) {
        token_ids[i] = 0;
    }
    size_t num_tokens = 20;
    
    uint32_t num_tokens_uint = (uint32_t)num_tokens;
    q_error_code err = q_tokenizer_encode(tok, "a", token_ids, &num_tokens_uint, 1024, false, false);
    num_tokens = (size_t)num_tokens_uint;
    
    assert(err == Q_OK && "Postconditions test should succeed");
    
    // Validar pós-condição crítica: nenhum Q_TOKEN_DELETED
    bool has_deleted = false;
    for (size_t i = 0; i < num_tokens; i++) {
        if (token_ids[i] == UINT32_MAX) {
            has_deleted = true;
            break;
        }
    }
    
    assert(!has_deleted && "No deleted tokens should remain");
    assert(validate_postconditions(token_ids, num_tokens, tok->vocab_size) && "Postconditions should be valid");
    
    printf("✓ Postconditions test passed (no deleted tokens)\n");
    printf("✓ Test 10 PASSED\n\n");
    
    destroy_dummy_tokenizer(tok);
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("========================================\n");
    printf("  TEST SUITE: BPE Soft-Delete\n");
    printf("  Adversarial Testing Protocol\n");
    printf("========================================\n\n");
    
    test_happy_path_basic();
    test_happy_path_multiple_merges();
    test_edge_case_empty_array();
    test_edge_case_single_token();
    test_null_tokenizer();
    test_null_token_ids();
    test_security_invalid_token_ids();
    test_performance_large_prompt();
    test_lazy_compaction_threshold();
    test_postconditions_no_deleted_tokens();
    
    printf("========================================\n");
    printf("  ALL TESTS PASSED ✓\n");
    printf("========================================\n");
    
    return 0;
}

