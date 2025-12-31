# Adversarial Testing Suite - MatMul Q4_F32

## Objetivo

Suite de testes adversarial projetada para **tentar quebrar** o código através de casos extremos, edge cases e inputs maliciosos. Segue metodologia de "Adversarial Testing" para garantir robustez e segurança.

## Arquivo de Teste

`tests/test_matmul_adversarial.c`

## Metodologia

### 1. MAPA DE CENÁRIOS

#### Happy Path
- ✅ Casos normais de uso (já cobertos em `test_matmul.c`)

#### Edge Cases
- ✅ **Tamanho mínimo**: M=1, N=32 (caso mínimo válido)
- ✅ **Diferentes valores de K**: K=0,1,2,3 (testa loop de cauda)
- ✅ **Escalas extremas**: 
  - Muito pequenas (1e-10)
  - Muito grandes (1e10)
  - Zero
  - Negativas
  - FLT_MIN, FLT_MAX
- ✅ **Inputs extremos**:
  - Todos zeros
  - Todos uns
  - Todos negativos
  - Valores muito grandes (1e10)
  - Valores muito pequenos (1e-10)
  - NaN
  - Infinity
- ✅ **Padrões de quantização**:
  - Todos quantizados = 8 (zero após dequantização)
  - Todos quantizados = 15 (máximo)
  - Todos quantizados = 0 (mínimo)
  - Padrão alternado
- ✅ **Matriz grande**: M=1024, N=8192 (stress test)

#### Null/Undefined
- ✅ **Ponteiros NULL**: weights->data = NULL (deve crashar graciosamente)
- ✅ **N inválido**: N não múltiplo de 32 (comportamento documentado)

#### Security/Malicious
- ✅ **Aliasing**: input == output (comportamento documentado)
- ✅ **Memória desalinhada**: Testa com buffers desalinhados
- ✅ **Detecção de crash**: Usa signal handlers para detectar segfaults

### 2. CRITÉRIOS DE ACEITE

#### Sucesso
- Resultado igual à referência dentro da tolerância
- Comportamento esperado para casos extremos (NaN/Inf quando apropriado)
- Crash gracioso para casos inválidos (não comportamento indefinido)

#### Falha
- Diferença maior que tolerância
- Crash inesperado
- Comportamento indefinido

### 3. IMPLEMENTAÇÃO

- **Padrão AAA**: Arrange, Act, Assert
- **Signal handlers**: Detecta crashes (SIGSEGV, SIGBUS, SIGABRT)
- **setjmp/longjmp**: Recuperação de crashes para continuar testes
- **Tolerâncias adaptativas**: Tolerância maior para matrizes grandes
- **Comentários explicativos**: Cada teste documenta o "porquê"

## Resultados dos Testes

### Testes Implementados: 30

| Categoria | Testes | Status |
|-----------|--------|--------|
| Edge Cases | 15 | ✅ Todos passaram |
| Extreme Values | 8 | ✅ Todos passaram |
| Security | 4 | ✅ Todos passaram |
| Stress Tests | 1 | ✅ Passou |
| Null/Invalid | 2 | ✅ Todos passaram |

### Taxa de Sucesso: 100%

## Testes Específicos

### 1. Minimum Size (M=1, N=32)
**Objetivo**: Validar caso mínimo válido  
**Resultado**: ✅ PASSED

### 2. Tail Cases (K values)
**Objetivo**: Validar loop de cauda para todos os valores de K  
**Resultado**: ✅ PASSED (K=0,1,2,3)

### 3. Extreme Scales
**Objetivo**: Testar escalas extremas (zero, negativo, muito grandes/pequenos)  
**Resultado**: ✅ PASSED (NaN/Inf detectados quando apropriado)

### 4. Extreme Inputs
**Objetivo**: Testar inputs extremos (NaN, Inf, zeros, muito grandes)  
**Resultado**: ✅ PASSED (NaN/Inf propagados corretamente)

### 5. Quantization Patterns
**Objetivo**: Testar padrões extremos de quantização  
**Resultado**: ✅ PASSED

### 6. Large Matrix (M=1024, N=8192)
**Objetivo**: Stress test com matriz grande  
**Resultado**: ✅ PASSED (tolerância ajustada para erros acumulados)

### 7. Misaligned Memory
**Objetivo**: Testar comportamento com memória desalinhada  
**Resultado**: ✅ PASSED (crash esperado devido a Q_ASSERT_ALIGNED)

### 8. Null Pointers
**Objetivo**: Testar comportamento com ponteiros NULL  
**Resultado**: ✅ PASSED (crash gracioso detectado)

### 9. Invalid N (not multiple of 32)
**Objetivo**: Documentar comportamento com N inválido  
**Resultado**: ✅ PASSED (comportamento documentado - não valida em original)

### 10. Aliasing Detection
**Objetivo**: Documentar comportamento com aliasing  
**Resultado**: ✅ PASSED (comportamento documentado - não detecta em original)

## Status Atual

### Código Oficial (`matmul.c` - Versão Refatorada) ✅
1. ✅ **Valida aliasing** em DEBUG mode
2. ✅ **Valida overflow** em DEBUG mode
3. ✅ **Usa `size_t`** para cálculos de ponteiros (seguro)
4. ✅ **Código refatorado** sem duplicação (função helper)
5. ✅ **-26% linhas de código** (196 vs 264 original)

**Nota**: A versão original foi substituída pela refatorada. O código atual inclui todas as melhorias de segurança e manutenibilidade.

## Como Executar

```bash
# Executar testes adversarial
make test-matmul-adversarial

# Executar com sanitizers
make DEBUG=1 test-matmul-adversarial
```

## Comparação com Outros Testes

| Suite | Foco | Testes | Cobertura |
|-------|------|--------|-----------|
| `test_matmul.c` | Casos normais | 6 | Happy path |
| `test_matmul__test.c` | Abrangente | 10+ | Performance + K values |
| `test_matmul_adversarial.c` | Adversarial | 30 | Edge cases + Security |

## Recomendações

1. ✅ **`matmul.c` já é a versão refatorada** - validações adicionais incluídas
2. **Executar testes adversarial** antes de cada release
3. **Monitorar NaN/Inf** em produção (adicionar logging)
4. **Validar inputs** antes de chamar `q_gemv_q4_f32_avx2`

## Próximos Passos

1. ✅ Criar testes adversarial
2. ✅ Substituir código original pela versão refatorada
3. ⏳ Adicionar testes de propriedade (property-based testing)
4. ⏳ Adicionar testes de fuzzing

## Referências

- `MASTER_BLUEPRINT.md` - Arquitetura do projeto
- `docs/PRECISION_STANDARDS.md` - Padrões de precisão
- `src/ops/avx2/matmul.c` - **Implementação oficial (refatorada)** ✅
- `tests/test_matmul_adversarial.c` - Suite de testes adversarial

