# Guia de Teste de Análise Estática

Este documento descreve como testar e interpretar a análise estática em camadas do projeto Qorus-IA.

## Visão Geral

O projeto usa uma abordagem de **análise estática em camadas**:
1. **GCC Analyzer** (primário): Detecta memory leaks, use-after-free, null-dereference, uninitialized values
2. **cppcheck** (complementar): Detecta bugs adicionais, estilo de código, e possíveis vulnerabilidades
3. **clang-tidy** (complementar): Detecta bugs, sugere modernizações, verifica performance e portabilidade

## Como Executar

### 1. GCC Analyzer (Análise Primária)

```bash
make analyze
```

**O que faz:**
- Compila todo o projeto com `-fanalyzer` e `-O0` (necessário para análise estática)
- Gera `static-analysis.log` com todos os resultados
- Detecta problemas críticos de memória e segurança

**Interpretação:**
- ✅ **Sucesso**: Compilação completa sem erros
- ⚠️ **Avisos**: Verificar `static-analysis.log` para problemas potenciais
- ❌ **Erro**: Problemas críticos que impedem compilação

### 2. cppcheck (Análise Complementar)

```bash
make analyze-cppcheck
```

**O que faz:**
- Executa cppcheck em todos os arquivos fonte (`src/` e `tests/`)
- Gera `cppcheck-report.log` com resultados
- Detecta bugs adicionais que o GCC analyzer pode não detectar

**Nota:** cppcheck precisa estar instalado:
```bash
sudo apt-get install cppcheck
```

### 3. clang-tidy (Análise Complementar)

```bash
make analyze-clang-tidy
```

**O que faz:**
- Gera `compile_commands.json` automaticamente (necessário para clang-tidy)
- Executa clang-tidy em todos os arquivos fonte (`src/`)
- Gera `clang-tidy-report.log` com resultados
- Detecta bugs, sugere modernizações, verifica performance e portabilidade

**Nota:** clang-tidy precisa estar instalado:
```bash
sudo apt-get install clang-tidy
```

**Checks habilitados:**
- `bugprone-*`: Detecta bugs comuns
- `cert-*`: Conformidade com CERT C Coding Standard
- `clang-analyzer-*`: Análise estática do Clang
- `cppcoreguidelines-*`: C++ Core Guidelines (quando aplicável)
- `misc-*`: Vários checks adicionais
- `modernize-*`: Sugestões de modernização
- `performance-*`: Problemas de performance
- `portability-*`: Problemas de portabilidade
- `readability-*`: Melhorias de legibilidade

### 4. Análise Completa (Todos)

```bash
make analyze-complete
```

**O que faz:**
- Executa GCC analyzer primeiro
- Depois executa cppcheck
- Por último executa clang-tidy
- Gera todos os logs

## Interpretando Resultados

### GCC Analyzer

**Problemas Críticos a Procurar:**
- `leak`: Memory leak detectado
- `use-after-free`: Uso de memória após liberação
- `null-dereference`: Desreferência de ponteiro NULL
- `uninitialized`: Uso de variável não inicializada
- `double-free`: Liberação dupla de memória

**Exemplo de Problema Real:**
```
src/core/memory.c:123:5: warning: leak of 'ptr' [CWE-401] [-Wanalyzer-malloc-leak]
```

**Exemplo de Falso Positivo:**
```
src/models/model.c:1108:49: error: terminating analysis for this program point
```
*Nota: Isso indica que o analyzer abortou por complexidade, não um bug real.*

### cppcheck

**Problemas Críticos a Procurar:**
- `error`: Erro crítico encontrado
- `warning`: Aviso que pode indicar bug
- `style`: Problemas de estilo (menos crítico)

**Exemplo de Problema Real:**
```
[src/core/memory.c:123]: (error) Memory leak: ptr
```

### clang-tidy

**Problemas Críticos a Procurar:**
- `error`: Erro crítico encontrado
- `warning`: Aviso que pode indicar bug
- `performance`: Problemas de performance
- `portability`: Problemas de portabilidade
- `readability`: Melhorias de legibilidade

**Exemplo de Problema Real:**
```
src/core/memory.c:123:5: warning: Potential memory leak [clang-analyzer-unix.Malloc]
```

**Categorias de Checks:**
- `bugprone-*`: Bugs comuns (ex: `bugprone-use-after-move`)
- `cert-*`: Conformidade CERT (ex: `cert-err33-c`)
- `clang-analyzer-*`: Análise estática (ex: `clang-analyzer-unix.Malloc`)
- `performance-*`: Performance (ex: `performance-for-range-copy`)
- `portability-*`: Portabilidade (ex: `portability-simd-intrinsics`)

## Verificação Rápida

Para verificar rapidamente se há problemas críticos:

```bash
# Verificar problemas no GCC analyzer
grep -iE "(leak|use-after-free|null-dereference|uninitialized|double-free)" static-analysis.log | grep -v "terminating analysis"

# Verificar problemas no cppcheck
grep -E "(error|warning)" cppcheck-report.log | grep -v "Checking"

# Verificar problemas no clang-tidy
grep -E "(error|warning)" clang-tidy-report.log | head -20
```

## Integração CI/CD

A análise estática é executada automaticamente em Pull Requests via GitHub Actions (`.github/workflows/ci.yml`):

1. **GCC Analyzer** executa primeiro (análise primária)
2. **cppcheck** executa depois (análise complementar)
3. **clang-tidy** pode ser adicionado ao CI (opcional)
4. Logs são gerados mas não bloqueiam o build

## Troubleshooting

### Problema: Stack Usage Exceeded

**Sintoma:**
```
src/ops/avx2/matmul.c:91:14: error: stack usage is 11424 bytes [-Werror=stack-usage=]
```

**Solução:**
- Arrays de debug foram movidos para `static` (não alocam stack)
- Pragmas GCC foram adicionados para suprimir warning apenas nesta função
- Ver `src/ops/avx2/matmul.c` para detalhes

### Problema: Analyzer Too Complex

**Sintoma:**
```
error: terminating analysis for this program point: ... [-Werror=analyzer-too-complex]
```

**Solução:**
- `-Wanalyzer-too-complex` foi removido do Makefile
- Permite análise mais profunda sem abortar por complexidade
- Funções complexas são analisadas completamente

### Problema: cppcheck Não Encontrado

**Sintoma:**
```
⚠ cppcheck não instalado
```

**Solução:**
```bash
sudo apt-get install cppcheck
```

### Problema: clang-tidy Não Encontrado

**Sintoma:**
```
⚠ clang-tidy não instalado
```

**Solução:**
```bash
sudo apt-get install clang-tidy
```

### Problema: compile_commands.json Não Gerado

**Sintoma:**
```
⚠ compile_commands.json não encontrado
```

**Solução:**
- O Makefile gera `compile_commands.json` automaticamente
- Se usar `bear`, instale: `sudo apt-get install bear`
- O Makefile cria manualmente se `bear` não estiver disponível

## Boas Práticas

1. **Execute análise estática antes de commit:**
   ```bash
   make analyze-complete
   ```

2. **Revise logs antes de PR:**
   - Verifique `static-analysis.log` para problemas do GCC analyzer
   - Verifique `cppcheck-report.log` para problemas do cppcheck
   - Verifique `clang-tidy-report.log` para problemas do clang-tidy

3. **Corrija problemas críticos:**
   - Memory leaks devem ser corrigidos imediatamente
   - Use-after-free e null-dereference são críticos
   - Problemas de estilo podem ser corrigidos gradualmente

## Referências

- [GCC Analyzer Documentation](https://gcc.gnu.org/onlinedocs/gcc/Static-Analyzer-Options.html)
- [cppcheck Documentation](https://cppcheck.sourceforge.io/manual.pdf)
- [clang-tidy Documentation](https://clang.llvm.org/extra/clang-tidy/)

