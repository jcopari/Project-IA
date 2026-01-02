// ============================================================================
// TEST SUITE: Arena Allocator Optimized (__builtin_assume_aligned)
// ============================================================================
// Lead SDET Protocol: Adversarial Testing para q_arena_alloc
// Baseado em: PLANEJAMENTO_CORRECOES_CRITICAS.md FASE 3.3 e 3.4
//
// Objetivo: Validar que __builtin_assume_aligned funciona corretamente
// Foco: Invariante matemática de alinhamento e segurança AVX2
// ============================================================================

#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>
#include <limits.h>
#include <immintrin.h>  // Para AVX2 validation

// ============================================================================
// FASE 0: CONTEXTO - Validação de Planejamento
// ============================================================================
// Especificação (FASE 3.4):
// - Pré-condições: ctx != NULL, ctx->scratch_buffer != NULL, size > 0
// - Invariante: scratch_head % Q_ALIGN == 0 (sempre verdadeiro)
// - Pós-condições: ptr % Q_ALIGN == 0, invariante mantida
// - Overhead: ≤ 2 ciclos (medido via benchmark)
//
// Failure Modes (FASE 3.3):
// - Validação de alinhamento em runtime (~5 ciclos) - ELIMINADO
// - Dependência de dados no pipeline - MITIGADO (__builtin_assume_aligned)
// - Remover validação sem invariante (crash AVX2) - CORRIGIDO (invariante garantida)
// ============================================================================

// ============================================================================
// FASE 1: MAPA DE CENÁRIOS
// ============================================================================
// Happy Path:
//   1. Alocação básica: size = 100 → ptr alinhado
//   2. Múltiplas alocações: Sequência de alocações mantém invariante
//   3. Tamanhos variados: Validar alinhamento para diferentes sizes
//
// Edge Cases:
//   4. size = 1 (mínimo)
//   5. size = Q_ALIGN (exatamente alinhado)
//   6. size = Q_ALIGN - 1 (arredondado para cima)
//   7. size = SIZE_MAX (overflow)
//   8. size = SIZE_MAX - scratch_head (overflow na adição)
//
// Null/Undefined:
//   9. ctx = NULL
//   10. ctx->scratch_buffer = NULL (arena não inicializada)
//   11. Estrutura ctx não inicializada (memset(0))
//
// Security/Malicious:
//   12. Integer overflow: size que causa wraparound
//   13. Buffer overflow: new_head > scratch_size
//   14. Invariante violada: scratch_head não múltiplo de Q_ALIGN (DEBUG)
//   15. Stale pointers: Acesso após q_arena_reset()
//
// Performance:
//   16. Overhead ≤ 2 ciclos: Benchmark de latência
//   17. AVX2 safety: Validar que ptr pode ser usado com VMOVAPS
// ============================================================================

// ============================================================================
// FASE 2: CRITÉRIOS DE ACEITE
// ============================================================================
// Null/Invalid Inputs:
//   - ctx = NULL → NULL (Graceful Failure)
//   - ctx->scratch_buffer = NULL → NULL
//   - size = 0 → NULL (overflow em safe_align_size)
//
// Edge Cases:
//   - size = 1 → ptr alinhado a Q_ALIGN
//   - size = SIZE_MAX → NULL (overflow)
//
// Security:
//   - Integer overflow → NULL (detectado antes da alocação)
//   - Buffer overflow → NULL (bounds check)
//   - Invariante violada → Abort em DEBUG
//
// Pós-condições (Especificação FASE 3.4):
//   - ptr % Q_ALIGN == 0 (se sucesso)
//   - ctx->scratch_head % Q_ALIGN == 0 (invariante mantida)
//   - Overhead ≤ 2 ciclos
// ============================================================================

// Helper: Validar alinhamento
static bool is_aligned(const void* ptr, size_t alignment) {
    return ((uintptr_t)ptr % alignment) == 0;
}

// Helper: Validar invariante
static bool validate_invariant(const q_context* ctx) {
    return (ctx->scratch_head % Q_ALIGN) == 0;
}

// Helper: Teste AVX2 safety (validar que ptr pode ser usado com VMOVAPS)
static bool test_avx2_safety(const float* ptr) {
    if (!is_aligned(ptr, 32)) {
        return false;  // Não alinhado, não pode usar VMOVAPS
    }
    
    // Tentar usar VMOVAPS (aligned load)
    // Se não estiver alinhado, isso causaria segfault
    __m256 vec = _mm256_load_ps(ptr);
    
    // Se chegou aqui, está seguro
    (void)vec;  // Evitar warning de variável não usada
    return true;
}

// ============================================================================
// FASE 3: IMPLEMENTAÇÃO BLINDADA
// ============================================================================

// Test 1: Happy Path - Alocação básica
static void test_happy_path_basic_allocation(void) {
    printf("Test 1: Happy Path - Alocação Básica\n");
    printf("-------------------------------------\n");
    
    q_context ctx = {0};
    q_error_code err = q_alloc_arena(&ctx, 1024 * 1024);  // 1MB
    assert(err == Q_OK && "Arena allocation should succeed");
    
    void* ptr = q_arena_alloc(&ctx, 100);
    
    assert(ptr != NULL && "Allocation should succeed");
    assert(is_aligned(ptr, Q_ALIGN) && "Pointer should be aligned");
    assert(validate_invariant(&ctx) && "Invariant should be maintained");
    
    printf("✓ Basic allocation test passed\n");
    printf("✓ Test 1 PASSED\n\n");
    
    q_free_memory(&ctx);
}

// Test 2: Happy Path - Múltiplas alocações
static void test_happy_path_multiple_allocations(void) {
    printf("Test 2: Happy Path - Múltiplas Alocações\n");
    printf("-----------------------------------------\n");
    
    q_context ctx = {0};
    q_error_code err = q_alloc_arena(&ctx, 1024 * 1024);
    assert(err == Q_OK && "Arena allocation should succeed");
    
    void* ptrs[10];
    for (int i = 0; i < 10; i++) {
        ptrs[i] = q_arena_alloc(&ctx, 100 + i * 10);
        assert(ptrs[i] != NULL && "Allocation should succeed");
        assert(is_aligned(ptrs[i], Q_ALIGN) && "Pointer should be aligned");
        assert(validate_invariant(&ctx) && "Invariant should be maintained");
    }
    
    printf("✓ Multiple allocations test passed\n");
    printf("✓ Test 2 PASSED\n\n");
    
    q_free_memory(&ctx);
}

// Test 3: Edge Case - size = 1 (mínimo)
static void test_edge_case_minimum_size(void) {
    printf("Test 3: Edge Case - size = 1\n");
    printf("-----------------------------\n");
    
    q_context ctx = {0};
    q_error_code err = q_alloc_arena(&ctx, 1024 * 1024);
    assert(err == Q_OK && "Arena allocation should succeed");
    
    void* ptr = q_arena_alloc(&ctx, 1);
    
    assert(ptr != NULL && "Allocation should succeed");
    assert(is_aligned(ptr, Q_ALIGN) && "Pointer should be aligned to Q_ALIGN");
    assert(validate_invariant(&ctx) && "Invariant should be maintained");
    
    printf("✓ Minimum size test passed\n");
    printf("✓ Test 3 PASSED\n\n");
    
    q_free_memory(&ctx);
}

// Test 4: Edge Case - size = Q_ALIGN (exatamente alinhado)
static void test_edge_case_aligned_size(void) {
    printf("Test 4: Edge Case - size = Q_ALIGN\n");
    printf("-----------------------------------\n");
    
    q_context ctx = {0};
    q_error_code err = q_alloc_arena(&ctx, 1024 * 1024);
    assert(err == Q_OK && "Arena allocation should succeed");
    
    void* ptr = q_arena_alloc(&ctx, Q_ALIGN);
    
    assert(ptr != NULL && "Allocation should succeed");
    assert(is_aligned(ptr, Q_ALIGN) && "Pointer should be aligned");
    assert(validate_invariant(&ctx) && "Invariant should be maintained");
    
    printf("✓ Aligned size test passed\n");
    printf("✓ Test 4 PASSED\n\n");
    
    q_free_memory(&ctx);
}

// Test 5: Edge Case - size = Q_ALIGN - 1 (arredondado para cima)
static void test_edge_case_round_up(void) {
    printf("Test 5: Edge Case - size = Q_ALIGN - 1\n");
    printf("--------------------------------------\n");
    
    q_context ctx = {0};
    q_error_code err = q_alloc_arena(&ctx, 1024 * 1024);
    assert(err == Q_OK && "Arena allocation should succeed");
    
    size_t initial_head = ctx.scratch_head;
    void* ptr = q_arena_alloc(&ctx, Q_ALIGN - 1);
    
    assert(ptr != NULL && "Allocation should succeed");
    assert(is_aligned(ptr, Q_ALIGN) && "Pointer should be aligned");
    assert(validate_invariant(&ctx) && "Invariant should be maintained");
    
    // Validar que foi arredondado para cima
    assert(ctx.scratch_head == initial_head + Q_ALIGN && "Size should be rounded up");
    
    printf("✓ Round up test passed\n");
    printf("✓ Test 5 PASSED\n\n");
    
    q_free_memory(&ctx);
}

// Test 6: Null/Undefined - ctx = NULL
static void test_null_context(void) {
    printf("Test 6: Null/Undefined - ctx = NULL\n");
    printf("------------------------------------\n");
    
    void* ptr = q_arena_alloc(NULL, 100);
    
    assert(ptr == NULL && "NULL context should return NULL");
    
    printf("✓ NULL context test passed\n");
    printf("✓ Test 6 PASSED\n\n");
}

// Test 7: Null/Undefined - Arena não inicializada
static void test_uninitialized_arena(void) {
    printf("Test 7: Null/Undefined - Arena Não Inicializada\n");
    printf("------------------------------------------------\n");
    
    q_context ctx = {0};  // Não inicializado (scratch_buffer = NULL)
    
    void* ptr = q_arena_alloc(&ctx, 100);
    
    assert(ptr == NULL && "Uninitialized arena should return NULL");
    
    printf("✓ Uninitialized arena test passed\n");
    printf("✓ Test 7 PASSED\n\n");
}

// Test 8: Security - Integer overflow
static void test_security_integer_overflow(void) {
    printf("Test 8: Security - Integer Overflow\n");
    printf("------------------------------------\n");
    
    q_context ctx = {0};
    q_error_code err = q_alloc_arena(&ctx, 1024 * 1024);
    assert(err == Q_OK && "Arena allocation should succeed");
    
    // Tentar causar overflow: SIZE_MAX
    ctx.scratch_head = SIZE_MAX - Q_ALIGN + 1;
    void* ptr = q_arena_alloc(&ctx, Q_ALIGN);
    
    assert(ptr == NULL && "Integer overflow should return NULL");
    
    printf("✓ Integer overflow test passed\n");
    printf("✓ Test 8 PASSED\n\n");
    
    q_free_memory(&ctx);
}

// Test 9: Security - Buffer overflow
static void test_security_buffer_overflow(void) {
    printf("Test 9: Security - Buffer Overflow\n");
    printf("----------------------------------\n");
    
    q_context ctx = {0};
    q_error_code err = q_alloc_arena(&ctx, 1024);  // Arena pequena
    assert(err == Q_OK && "Arena allocation should succeed");
    
    // Tentar alocar mais do que a arena suporta
    ctx.scratch_head = ctx.scratch_size - Q_ALIGN + 1;
    void* ptr = q_arena_alloc(&ctx, Q_ALIGN);
    
    assert(ptr == NULL && "Buffer overflow should return NULL");
    
    printf("✓ Buffer overflow test passed\n");
    printf("✓ Test 9 PASSED\n\n");
    
    q_free_memory(&ctx);
}

// Test 10: Performance - AVX2 Safety (validar que ptr pode ser usado com VMOVAPS)
static void test_performance_avx2_safety(void) {
    printf("Test 10: Performance - AVX2 Safety\n");
    printf("----------------------------------\n");
    
    q_context ctx = {0};
    q_error_code err = q_alloc_arena(&ctx, 1024 * 1024);
    assert(err == Q_OK && "Arena allocation should succeed");
    
    // Alocar array de floats para AVX2
    float* ptr = (float*)q_arena_alloc(&ctx, 32 * sizeof(float));  // 32 floats = 128 bytes
    
    assert(ptr != NULL && "Allocation should succeed");
    assert(is_aligned(ptr, 32) && "Pointer should be aligned to 32 bytes for AVX2");
    
    // Inicializar com valores de teste
    for (int i = 0; i < 32; i++) {
        ptr[i] = (float)i;
    }
    
    // Validar que pode ser usado com VMOVAPS (aligned load)
    bool avx2_safe = test_avx2_safety(ptr);
    assert(avx2_safe && "Pointer should be safe for AVX2 aligned operations");
    
    printf("✓ AVX2 safety test passed\n");
    printf("✓ Test 10 PASSED\n\n");
    
    q_free_memory(&ctx);
}

// Test 11: Invariante - Validar que invariante é mantida após múltiplas alocações
static void test_invariant_maintained(void) {
    printf("Test 11: Invariante - Mantida Após Múltiplas Alocações\n");
    printf("------------------------------------------------------\n");
    
    q_context ctx = {0};
    q_error_code err = q_alloc_arena(&ctx, 1024 * 1024);
    assert(err == Q_OK && "Arena allocation should succeed");
    
    // Validar invariante inicial
    assert(validate_invariant(&ctx) && "Initial invariant should be valid");
    
    // Fazer múltiplas alocações com tamanhos variados
    for (int i = 0; i < 100; i++) {
        size_t size = 1 + (i % 100) * 7;  // Tamanhos variados
        void* ptr = q_arena_alloc(&ctx, size);
        
        assert(ptr != NULL && "Allocation should succeed");
        assert(is_aligned(ptr, Q_ALIGN) && "Pointer should be aligned");
        assert(validate_invariant(&ctx) && "Invariant should be maintained");
    }
    
    printf("✓ Invariant maintained test passed (100 allocations)\n");
    printf("✓ Test 11 PASSED\n\n");
    
    q_free_memory(&ctx);
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("========================================\n");
    printf("  TEST SUITE: Arena Optimized\n");
    printf("  Adversarial Testing Protocol\n");
    printf("========================================\n\n");
    
    test_happy_path_basic_allocation();
    test_happy_path_multiple_allocations();
    test_edge_case_minimum_size();
    test_edge_case_aligned_size();
    test_edge_case_round_up();
    test_null_context();
    test_uninitialized_arena();
    test_security_integer_overflow();
    test_security_buffer_overflow();
    test_performance_avx2_safety();
    test_invariant_maintained();
    
    printf("========================================\n");
    printf("  ALL TESTS PASSED ✓\n");
    printf("========================================\n");
    
    return 0;
}

