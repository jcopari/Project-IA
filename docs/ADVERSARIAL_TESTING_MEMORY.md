# Adversarial Testing: Memory Management

## Visão Geral

Este documento descreve a suíte completa de testes adversarial para as funções de gerenciamento de memória do Qorus-IA, focando em segurança, robustez e prevenção de bugs críticos.

**Arquivo de Teste:** `tests/test_memory_adversarial.c`  
**Total de Testes:** 22  
**Metodologia:** AAA Pattern (Arrange, Act, Assert) + Crash Detection

## Objetivo

Tentar **QUEBRAR** o código através de:
- Inputs maliciosos e edge cases extremos
- Tentativas de corrupção de memória
- Exploração de condições de corrida
- Testes de estresse e fadiga

## Categorias de Testes

### 1. Null Pointer Tests (4 testes)

#### Test 1: `q_alloc_kv_cache` - Null context pointer
- **Objetivo:** Verificar que a função retorna erro ao receber NULL
- **Critério de Aceite:** Deve retornar `Q_ERR_INVALID_ARG`
- **Previne:** Segfault em produção

#### Test 2: `q_alloc_arena` - Null context pointer
- **Objetivo:** Verificar que a função retorna erro ao receber NULL
- **Critério de Aceite:** Deve retornar `Q_ERR_INVALID_ARG`
- **Previne:** Segfault em produção

#### Test 3: `q_arena_alloc` - Null context pointer
- **Objetivo:** Verificar que a função retorna NULL ao receber NULL
- **Critério de Aceite:** Deve retornar NULL sem crash
- **Previne:** Segfault em produção

#### Test 4: `q_arena_reset` - Null context pointer
- **Objetivo:** Verificar que a função não crasha com NULL
- **Critério de Aceite:** Deve retornar silenciosamente sem crash
- **Previne:** Segfault em produção

### 2. Memory Leak Prevention (2 testes)

#### Test 5: `q_alloc_kv_cache` - Double allocation
- **Objetivo:** Verificar que double allocation é detectada
- **Critério de Aceite:** Segunda chamada deve retornar `Q_ERR_INVALID_ARG`
- **Previne:** Memory leak silencioso

#### Test 6: `q_alloc_arena` - Double allocation
- **Objetivo:** Verificar que double allocation é detectada
- **Critério de Aceite:** Segunda chamada deve retornar `Q_ERR_INVALID_ARG`
- **Previne:** Memory leak silencioso

### 3. Overflow Protection (5 testes)

#### Test 7: `q_alloc_kv_cache` - Size overflow
- **Objetivo:** Verificar proteção contra overflow em `safe_align_size`
- **Critério de Aceite:** Deve retornar `Q_ERR_OVERFLOW` para `SIZE_MAX - 50`
- **Previne:** Wraparound e comportamento indefinido

#### Test 8: `q_alloc_arena` - Size overflow
- **Objetivo:** Verificar proteção contra overflow em `safe_align_size`
- **Critério de Aceite:** Deve retornar `Q_ERR_OVERFLOW` para `SIZE_MAX - 50`
- **Previne:** Wraparound e comportamento indefinido

#### Test 9: `q_arena_alloc` - Overflow in alignment calculation
- **Objetivo:** Verificar que `safe_align_size` detecta overflow
- **Critério de Aceite:** Deve retornar NULL para tamanhos próximos de `SIZE_MAX`
- **Previne:** Wraparound silencioso

#### Test 10: `q_arena_alloc` - Overflow in addition (head + size)
- **Objetivo:** Verificar proteção contra overflow na adição
- **Critério de Aceite:** Deve retornar NULL quando `head + size` overflow
- **Previne:** Wraparound e corrupção de memória

#### Test 11: `q_alloc_kv_cache` - Zero size
- **Objetivo:** Verificar comportamento com tamanho zero
- **Critério de Aceite:** Deve alinhar para `Q_ALIGN` bytes
- **Previne:** Comportamento indefinido

### 4. Out of Memory (OOM) Tests (1 teste)

#### Test 12: `q_arena_alloc` - Out of memory
- **Objetivo:** Verificar que OOM é detectado corretamente
- **Critério de Aceite:** Deve retornar NULL quando não há espaço suficiente
- **Previne:** Buffer overflow

### 5. Alignment Tests (2 testes)

#### Test 13: `q_arena_alloc` - Misalignment detection
- **Objetivo:** Verificar que desalinhamento é detectado
- **Critério de Aceite:** Deve retornar NULL quando `scratch_head` está desalinhado
- **Previne:** Crash em instruções AVX2

#### Test 14: `q_arena_alloc` - Zero size alignment
- **Objetivo:** Verificar que tamanho zero é alinhado corretamente
- **Critério de Aceite:** Ponteiro retornado deve estar alinhado a `Q_ALIGN`
- **Previne:** Desalinhamento em alocações subsequentes

### 6. Reset Tests (2 testes)

#### Test 15: `q_arena_reset` - Null context pointer
- **Objetivo:** Verificar que reset não crasha com NULL
- **Critério de Aceite:** Deve retornar silenciosamente
- **Previne:** Segfault

#### Test 16: `q_arena_reset` - Uninitialized arena
- **Objetivo:** Verificar que reset funciona mesmo sem buffer inicializado
- **Critério de Aceite:** Deve resetar `scratch_head` para 0 sem crash
- **Previne:** Segfault em edge cases

### 7. Free Tests (3 testes)

#### Test 17: `q_free_memory` - Null context pointer
- **Objetivo:** Verificar que free não crasha com NULL
- **Critério de Aceite:** Deve retornar silenciosamente
- **Previne:** Segfault

#### Test 18: `q_free_memory` - Double free
- **Objetivo:** Verificar que double free é seguro
- **Critério de Aceite:** Segunda chamada não deve crashar
- **Previne:** Double free crash

#### Test 19: `q_free_memory` - Partial allocation
- **Objetivo:** Verificar que free funciona com alocações parciais
- **Critério de Aceite:** Deve limpar apenas o que foi alocado
- **Previne:** Use-after-free

#### Test 20: `q_free_memory` - LIFO order verification
- **Objetivo:** Verificar que recursos são liberados em ordem LIFO
- **Critério de Aceite:** Todos os ponteiros devem ser NULL após free
- **Previne:** Dangling pointers

### 8. Stress Tests (2 testes)

#### Test 21: Arena - Multiple allocations and resets
- **Objetivo:** Verificar robustez sob carga repetida
- **Critério de Aceite:** 100 iterações de alocação/reset devem funcionar
- **Previne:** Corrupção de memória após uso prolongado

#### Test 22: Arena - Alignment preservation
- **Objetivo:** Verificar que alinhamento é preservado para todos os tamanhos
- **Critério de Aceite:** Todos os ponteiros retornados devem estar alinhados
- **Previne:** Desalinhamento acumulado

## Metodologia

### Crash Detection

Usa `setjmp`/`longjmp` com signal handlers para detectar crashes:

```c
if (setjmp(crash_jmp_buf) == 0) {
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGABRT, crash_handler);
    
    // Test code here
    
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGABRT, SIG_DFL);
    
    TEST_PASS();
} else {
    TEST_CRASH();
}
```

### Padrão AAA

Todos os testes seguem o padrão **Arrange, Act, Assert**:

1. **Arrange:** Configurar contexto e dados de teste
2. **Act:** Executar a função sob teste
3. **Assert:** Verificar resultado esperado

## Resultados Esperados

### Taxa de Sucesso Alvo: 100%

Todos os testes devem passar, indicando que:
- ✅ Proteções de segurança estão funcionando
- ✅ Edge cases são tratados corretamente
- ✅ Não há crashes ou comportamento indefinido
- ✅ Memory leaks são prevenidos

## Execução

```bash
# Compilar e executar testes adversarial de memória
make test-memory-adversarial

# Executar com sanitizers (recomendado)
make DEBUG=1 test-memory-adversarial
```

## Cobertura

### Funções Testadas

- ✅ `safe_align_size()` (indiretamente)
- ✅ `q_alloc_kv_cache()`
- ✅ `q_alloc_arena()`
- ✅ `q_arena_alloc()`
- ✅ `q_arena_reset()`
- ✅ `q_free_memory()`

### Cenários Cobertos

- ✅ Null pointers
- ✅ Memory leaks
- ✅ Integer overflow
- ✅ Buffer overflow
- ✅ Misalignment
- ✅ Double free
- ✅ Use-after-free
- ✅ Stress testing

## Notas de Implementação

1. **Signal Handlers:** Usados apenas para detecção de crash, não para recuperação
2. **Test Isolation:** Cada teste é independente e limpa seus recursos
3. **Error Codes:** Testes verificam códigos de erro específicos, não apenas sucesso/falha
4. **Performance:** Testes são rápidos (< 1 segundo total) para permitir execução frequente

## Histórico de Bugs Encontrados

Nenhum bug encontrado até o momento. Todos os testes passam, confirmando que as melhorias de segurança implementadas estão funcionando corretamente.

