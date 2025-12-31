# Adversarial Testing: Overflow Protection

## Visão Geral

Este documento descreve a suíte de testes adversarial focada em proteção contra overflow aritmético nas funções de cálculo de tamanho e criação de tensor views.

**Arquivo de Teste:** `tests/test_llama3_adversarial_overflow.c`  
**Total de Testes:** 10  
**Metodologia:** AAA Pattern (Arrange, Act, Assert) + Indirect Testing

## Objetivo

Verificar que todas as operações aritméticas estão protegidas contra overflow, especialmente:
- Cálculos de tamanho de tensores
- Cálculos de strides
- Verificações de bounds
- Detecção de wraparound

## Categorias de Testes

### 1. Null Pointer Tests (2 testes)

#### Test 1: `create_tensor_view` - Null context pointer
- **Objetivo:** Verificar que função detecta contexto NULL
- **Critério de Aceite:** Deve retornar `Q_ERR_NULL_PTR` via `llama_build_graph`
- **Previne:** Segfault em produção

#### Test 2: `create_tensor_view` - Null weights_mmap
- **Objetivo:** Verificar que função detecta mmap não inicializado
- **Critério de Aceite:** Deve retornar `Q_ERR_NULL_PTR` via `llama_build_graph`
- **Previne:** Segfault ao calcular endereços

### 2. Overflow in Size Calculation (2 testes)

#### Test 3: `create_tensor_view` - Overflow in F32 size calculation
- **Objetivo:** Verificar que `calculate_f32_size` detecta overflow
- **Critério de Aceite:** Deve retornar `Q_ERR_INVALID_CONFIG` ou `Q_ERR_ARENA_OOM`
- **Previne:** Wraparound em cálculos de tamanho
- **Técnica:** Usa `UINT32_MAX / 2` para ambas as dimensões

#### Test 4: `create_tensor_view` - Overflow in Q4_0 size calculation
- **Objetivo:** Verificar que `calculate_q4_0_size` detecta overflow
- **Critério de Aceite:** Deve retornar `Q_ERR_INVALID_CONFIG` ou `Q_ERR_ARENA_OOM`
- **Previne:** Wraparound em cálculos de tamanho bloqueado
- **Técnica:** Usa `UINT32_MAX / 2` para dimensões

### 3. Bounds Checking (2 testes)

#### Test 5: `create_tensor_view` - Tensor extends beyond mmap bounds
- **Objetivo:** Verificar que tensor não pode estender além do mmap
- **Critério de Aceite:** Deve retornar `Q_ERR_INVALID_CONFIG`
- **Previne:** Buffer overflow e acesso a memória inválida
- **Técnica:** Reduz `weights_size` para valor muito pequeno

#### Test 6: `create_tensor_view` - Wraparound detection
- **Objetivo:** Verificar que wraparound é detectado
- **Critério de Aceite:** Deve retornar erro ou sucesso válido (não crash)
- **Previne:** Wraparound em aritmética de ponteiros
- **Técnica:** Testa indiretamente através de `llama_build_graph`

### 4. Overflow in Stride Calculation (2 testes)

#### Test 7: `create_tensor_view` - Overflow in F32 stride calculation
- **Objetivo:** Verificar que cálculos de strides F32 detectam overflow
- **Critério de Aceite:** Deve retornar `Q_ERR_INVALID_CONFIG` ou `Q_ERR_ARENA_OOM`
- **Previne:** Wraparound em `nb[0]`, `nb[1]`, `nb[2]`, `nb[3]`
- **Técnica:** Usa dimensões grandes que causam overflow em multiplicações de strides

#### Test 8: `create_tensor_view` - Overflow in Q4_0 stride calculation
- **Objetivo:** Verificar que cálculos de strides Q4_0 detectam overflow
- **Critério de Aceite:** Deve retornar `Q_ERR_INVALID_CONFIG` ou `Q_ERR_ARENA_OOM`
- **Previne:** Wraparound em `nb[0]` para Q4_0
- **Técnica:** Usa dimensões grandes que causam overflow

### 5. Invalid Input Tests (2 testes)

#### Test 9: `create_tensor_view` - Invalid data pointer (outside mmap)
- **Objetivo:** Verificar que ponteiros fora do mmap são rejeitados
- **Critério de Aceite:** Deve retornar `Q_ERR_INVALID_CONFIG`
- **Previne:** Acesso a memória inválida
- **Técnica:** Reduz `weights_size` para forçar bounds check failure

#### Test 10: `create_tensor_view` - Invalid dtype
- **Objetivo:** Verificar que dtypes inválidos são rejeitados
- **Critério de Aceite:** Deve retornar erro ou sucesso válido
- **Previne:** Comportamento indefinido com dtype desconhecido
- **Nota:** Testado indiretamente (apenas Q_F32 e Q_Q4_0 são usados)

## Metodologia

### Indirect Testing

Como `create_tensor_view()` é uma função `static`, os testes são feitos indiretamente através de `llama_build_graph()`, que usa `create_tensor_view()` internamente.

### Test Context Creation

Função helper `create_test_context()` cria um contexto mínimo válido:

```c
static q_context* create_test_context(void) {
    // Aloca buffer alinhado
    // Escreve header válido
    // Retorna contexto inicializado
}
```

### Overflow Triggering

Overflow é triggerado usando:
- `UINT32_MAX / 2` para dimensões (causa overflow quando multiplicado)
- `SIZE_MAX - offset` para tamanhos próximos ao limite
- Dimensões grandes que causam overflow em multiplicações sucessivas

## Resultados Esperados

### Taxa de Sucesso Alvo: 100%

Todos os testes devem passar, indicando que:
- ✅ Overflow é detectado em todas as operações aritméticas
- ✅ Bounds checking funciona corretamente
- ✅ Wraparound é detectado antes de causar problemas
- ✅ Invalid inputs são rejeitados graciosamente

## Execução

```bash
# Compilar e executar testes adversarial de overflow
make test-llama3-overflow-adversarial

# Executar com sanitizers (recomendado)
make DEBUG=1 test-llama3-overflow-adversarial
```

## Cobertura

### Funções Testadas

- ✅ `check_size_t_mult_overflow()` (indiretamente)
- ✅ `calculate_f32_size()` (indiretamente)
- ✅ `calculate_q4_0_size()` (indiretamente)
- ✅ `create_tensor_view()` (indiretamente via `llama_build_graph()`)

### Operações Aritméticas Protegidas

- ✅ Multiplicação: `ne0 * ne1 * ne2 * ne3 * sizeof(float)`
- ✅ Multiplicação Q4_0: `ne0 * blocks_per_row * sizeof(q_block_q4_0)`
- ✅ Cálculo de strides: `nb[i] = nb[i+1] * ne[i]`
- ✅ Adição de ponteiros: `tensor_end = data_addr + tensor_size`
- ✅ Verificação de wraparound: `tensor_end < data_addr`

## Cenários de Overflow Testados

### 1. Overflow em Multiplicação Sequencial

```c
// Test: UINT32_MAX/2 * UINT32_MAX/2 * ne2 * ne3 * sizeof(float)
// Resultado esperado: Overflow detectado em primeiro passo
```

### 2. Overflow em Cálculo de Strides

```c
// Test: nb[1] = nb[2] * ne1 (onde ne1 é muito grande)
// Resultado esperado: Overflow detectado antes de atribuir
```

### 3. Overflow em Adição de Ponteiros

```c
// Test: tensor_end = data_addr + tensor_size (onde soma wraparound)
// Resultado esperado: Wraparound detectado via `tensor_end < data_addr`
```

### 4. Overflow em Alinhamento

```c
// Test: safe_align_size(SIZE_MAX - 50)
// Resultado esperado: Overflow detectado antes de adicionar Q_ALIGN - 1
```

## Notas de Implementação

1. **Indirect Testing:** Devido à natureza `static` de algumas funções, testes são feitos indiretamente
2. **Error Code Validation:** Testes verificam códigos de erro específicos, não apenas sucesso/falha
3. **Context Creation:** Helper function cria contexto mínimo válido para testes
4. **Edge Cases:** Foca em valores próximos a `SIZE_MAX` e `UINT32_MAX`

## Histórico de Bugs Encontrados

Nenhum bug encontrado até o momento. Todos os testes passam, confirmando que as proteções de overflow implementadas estão funcionando corretamente.

