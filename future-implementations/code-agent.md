# 🏛️ QORUS-IA CODE AGENT v3.1: ELITE SYSTEM BLUEPRINT

**Data**: 2024-12-29  
**Versão**: v3.1.0 (Elite System - Dual-Agent Architecture)  
**Status**: 📋 Documento Prático de Implementação  
**Arquitetura**: Dual-Agent (Architect + Auditor) com Aprendizado por Reforço  
**Base de Conhecimento**: Elite Repos (Linux/Doom) + Livros de Engenharia  
**Engine**: Qorus-IA v3.0 (C/CUDA Hybrid)  
**Metodologia Core**: TDD + MFR + CoT + Proof (Integrado e Obrigatório)

---

## 📋 ÍNDICE

1. [Visão Geral e Contexto](#visão-geral-e-contexto)
2. [Arquitetura Dual-Agent: The Inner Loop](#arquitetura-dual-agent-the-inner-loop)
3. [Pipeline de Treinamento de Elite](#pipeline-de-treinamento-de-elite)
4. [Metodologia Core: TDD + MFR + CoT + Proof](#metodologia-core-tdd--mfr--cot--proof)
5. [O que Já Existe vs O que Precisa ser Feito](#o-que-já-existe-vs-o-que-precisa-ser-feito)
6. [Roadmap Progressivo de Implementação](#roadmap-progressivo-de-implementação)
7. [Especificações Técnicas](#especificações-técnicas)
8. [Integração e Deploy](#integração-e-deploy)
9. [Referências e Checklist](#referências-e-checklist)

---

## 🎯 VISÃO GERAL E CONTEXO

### Propósito

**Qorus-IA Code Agent v3.1** é um **Sistema Autônomo de Engenharia de Software de Elite** que utiliza arquitetura dual-agente (Architect + Auditor) para gerar código de qualidade seguindo metodologia rigorosa (TDD + MFR + CoT + Proof).

**Características Principais:**
- ✅ **Ferramenta Interna** - Acesso restrito à equipe de desenvolvimento (não para clientes)
- ✅ **Arquitetura Dual-Agent** - Architect (gerador) + Auditor (validador) em loop colaborativo/adversarial
- ✅ **Multi-Linguagem** - Suporta todas linguagens do projeto (C, Python, JavaScript, TypeScript, PHP, SQL, etc)
- ✅ **Latência ultra-baixa** (inferência local no servidor: 10-50ms)
- ✅ **Privacidade total** (código nunca sai do servidor)
- ✅ **Código sempre testado** (TDD automático integrado)
- ✅ **Performance CPU** (157.79 GFLOPS sem GPU)
- ✅ **LoRA Adapters** - Economia massiva de VRAM (87.5% de redução)
- ✅ **Treinamento Elite** - Kernel Linux + Doom + CSAPP + LeetCode + AlphaZero

**Objetivo**: Criar um **Engenheiro de Software Artificial Sênior** especializado em C e Sistemas, capaz de se auto-melhorar através de aprendizado por reforço.

### Casos de Uso

- **Code Completion**: Autocompletar código enquanto você digita
- **Code Generation**: Gerar código a partir de descrições em linguagem natural
- **Code Refactoring**: Refatorar código existente seguindo instruções
- **Code Explanation**: Explicar código existente
- **Design-to-Code**: Gerar código frontend a partir de imagens de design (mockups, Figma)

### Arquitetura Simplificada

```
┌─────────────────────────────────────────────────────────┐
│              CLIENTE (PhpStorm ou Chat Web)            │
│              - LSP Client ou Interface Web             │
└───────────────────────┬─────────────────────────────────┘
                        │ SSH/HTTP
                        ↓
┌─────────────────────────────────────────────────────────┐
│              SERVIDOR (tempo-main)                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Chat Server (Node.js/TypeScript) - Opcional    │  │
│  └───────────────────┬──────────────────────────────┘  │
│                      ↕ IPC                              │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Qorus-IA Engine (C Native)                      │  │
│  │  - Base Model (Llama-3 Coder) - Congelado        │  │
│  │  - LoRA Architect Adapter (Pequeno)             │  │
│  │  - LoRA Auditor Adapter (Pequeno)               │  │
│  │  - Tokenizer Multi-Linguagem                    │  │
│  │  - TDD + MFR + CoT + Proof integrado           │  │
│  │  - 157.79 GFLOPS                                │  │
│  └───────────────────┬──────────────────────────────┘  │
│                      ↕ The Inner Loop                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  ARCHITECT (Generator)                           │  │
│  │  - Gera código baseado em MFR + Proof           │  │
│  │  - System Prompt: "John Carmack + Linus"        │  │
│  └───────────────────┬──────────────────────────────┘  │
│                      ↕                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  AUDITOR (Verifier/Bug Hunter)                  │  │
│  │  - Analisa código do Architect                  │  │
│  │  - System Prompt: "Security Analyst + Kernel"   │  │
│  │  - Rejeita código inseguro/perigoso             │  │
│  └───────────────────┬──────────────────────────────┘  │
│                      ↕                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  COMPILER (Final Judge)                         │  │
│  │  - GCC + ASAN + Testes                           │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🤝 ARQUITETURA DUAL-AGENT: THE INNER LOOP

### Conceito: Dois Especialistas em Loop

Não teremos apenas um modelo tentando acertar. Teremos **dois especialistas rodando em loop** (Adversarial/Collaborative Refinement).

**Filosofia**: "Dois agentes são melhores que um" - especialização e validação cruzada através de loop iterativo.

### 🧠 Agente A: O ARQUITETO (Architect - Generator)

**Perfil**: Criativo, focado em performance, algoritmos e "First Principles Thinking".

**System Prompt**: *"Você é John Carmack misturado com Linus Torvalds. Pense na memória, no cache e na complexidade assintótica antes de escrever. Use AVX2 onde possível. Sempre prove matematicamente sua solução antes de implementar."*

**Responsabilidades**:
- Geração de código baseada em requisitos
- Implementação seguindo TDD + MFR + CoT + Proof
- Geração de testes iniciais
- Proposta de modelo de dados (MFR)
- Otimização de performance (cache, SIMD, algoritmos)

**Especialização**: Criatividade, geração, implementação, otimização

### 🕵️ Agente B: O AUDITOR (Auditor - Verifier/Bug Hunter)

**Perfil**: Paranoico, especialista em segurança e QA.

**System Prompt**: *"Você é um Analista de Segurança Sênior e Mantenedor do Kernel Linux. Procure por memory leaks, race conditions, buffer overflows e violações de estilo. Seja impiedoso. Rejeite código inseguro ou perigoso."*

**Responsabilidades**:
- **Code Review**: Análise estática de código, detecção de bugs potenciais
- **Security Analysis**: Memory leaks, buffer overflows, race conditions
- **Test Generation**: Geração adicional de testes (edge cases, stress tests)
- **Debug Analysis**: Identificação de problemas, sugestões de correção
- **Quality Assurance**: Validação contra padrões, métricas de qualidade
- **Proof Verification**: Verificação matemática de complexidade e corretude

**Especialização**: Análise crítica, validação, garantia de qualidade, segurança

### O Loop de Execução (The Inner Loop)

**Implementação em C** (`src/agent/core.c`):

```c
#include "qorus.h"  // New-QorusIA v3.0 API

typedef enum {
    TURN_ARCHITECT,
    TURN_AUDITOR,
    TURN_COMPILER
} q_agent_turn;

typedef struct {
    q_context* ctx;                    // New-QorusIA context
    void* base_model;                  // Base model (Llama-3 Coder) - congelado
    void* architect_lora;              // LoRA adapter para Architect
    void* auditor_lora;                 // LoRA adapter para Auditor
    q_tokenizer* tokenizer;
    uint32_t max_retries;
} q_dual_agent;

typedef struct {
    char* code;
    char* tests;
    char* data_model;
    char* proof;
    bool approved;
    char* critique;
    q_error_code compiler_result;
} q_agent_output;

// Executar ciclo elite (The Inner Loop)
q_error_code q_run_elite_cycle(q_dual_agent* agent,
                                 const char* problem,
                                 const char* language,
                                 q_agent_output* output) {
    if (!agent || !problem || !output) {
        return Q_ERR_NULL_PTR;
    }
    
    char* code = NULL;
    char* critique = NULL;
    q_error_code ret = Q_OK;
    
    // 1. ARCHITECT gera código (Baseado em MFR + Proof)
    ret = q_architect_generate(agent, problem, language, &code);
    if (ret != Q_OK) {
        return ret;
    }
    
    // Loop de refinamento colaborativo/adversarial
    for (uint32_t i = 0; i < agent->max_retries; i++) {
        // 2. AUDITOR analisa código (Static Analysis Mental)
        ret = q_auditor_review(agent, code, language, &critique);
        if (ret != Q_OK) {
            free(code);
            return ret;
        }
        
        // Verificar se foi aprovado
        bool approved = q_is_approved(critique);
        
        if (approved) {
            // 3. O Juiz Final (Compilador + Testes)
            ret = q_compiler_check(agent, code, language, &output->compiler_result);
            if (ret == Q_OK && output->compiler_result == Q_OK) {
                // Código de Elite pronto
                output->code = code;
                output->approved = true;
                free(critique);
                return Q_OK;
            }
        }
        
        // 4. Feedback Loop - Architect refina baseado em crítica
        char* refined_code = NULL;
        ret = q_architect_refine(agent, code, critique, language, &refined_code);
        if (ret != Q_OK) {
            free(code);
            free(critique);
            return ret;
        }
        
        free(code);
        code = refined_code;
        free(critique);
        critique = NULL;
    }
    
    // Se chegou aqui, falhou após max_retries
    free(code);
    output->approved = false;
    return Q_ERR_MAX_RETRIES;
}
```

### Fluxo Detalhado do Inner Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    REQUEST (Desenvolvedor)                  │
│              "Implemente função de ordenação"               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              ARCHITECT (Geração Inicial)                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ FASE 0: CoT - Raciocínio                            │  │
│  │ FASE 0.5: Proof - Validação Matemática              │  │
│  │ FASE 1: MFR - Modelo de Dados                       │  │
│  │ FASE 2: TDD - Testes Iniciais                       │  │
│  │ FASE 3: Implementação - Código                      │  │
│  └───────────────────┬──────────────────────────────────┘  │
│                      │ Código + Testes + Modelo + Proof    │
│                      ↓                                      │
┌─────────────────────────────────────────────────────────────┐
│              AUDITOR (Análise Crítica)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Code Review: Análise estática                     │  │
│  │ 2. Security Check: Memory leaks, buffer overflows    │  │
│  │ 3. Test Expansion: Testes adicionais                 │  │
│  │ 4. Proof Verification: Validação matemática          │  │
│  │ 5. Quality Metrics: Complexidade, manutenibilidade   │  │
│  │ 6. Debug Analysis: Identificação de problemas        │  │
│  └───────────────────┬──────────────────────────────────┘  │
│                      │ Aprovado?                           │
│                      ↓                                      │
│              ┌───────┴───────┐                            │
│              │ SIM           │ NÃO                         │
│              ↓               ↓                             │
│  ┌───────────────────┐  ┌──────────────────────────────┐ │
│  │ COMPILER CHECK     │  │ FEEDBACK LOOP                │ │
│  │ GCC + ASAN + Tests │  │ Architect refina código      │ │
│  └───────┬───────────┘  └───────────┬──────────────────┘ │
│          │                           │                      │
│          ↓                           └──────────┐          │
│  ┌───────────────┐                              │          │
│  │ PASSOU?       │                              │          │
│  └───┬───────┬───┘                              │          │
│      │ SIM   │ NÃO                              │          │
│      ↓       └──────────────────────────────────┘          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              RESPONSE (Desenvolvedor)                   ││
│  │              Código de Elite validado e testado        ││
│  └─────────────────────────────────────────────────────────┘│
```

### Implementação Técnica com New-QorusIA v3.0

```c
// Estrutura para Dual-Agent usando New-QorusIA v3.0
typedef struct {
    q_context* ctx;                    // Contexto de memória New-QorusIA
    q_tokenizer* tokenizer;            // Tokenizer multi-linguagem
    
    // Base Model (congelado, compartilhado)
    void* base_model_weights;          // Pesos do modelo base (mmap)
    
    // LoRA Adapters (pequenos, trocáveis)
    void* architect_lora_weights;      // Pesos do adaptador Architect (~2GB)
    void* auditor_lora_weights;        // Pesos do adaptador Auditor (~2GB)
    
    // Estado atual
    q_agent_turn current_turn;
    uint32_t iteration_count;
    uint32_t max_iterations;
} q_dual_agent;

// Carregar adaptador LoRA (troca rápida)
q_error_code q_load_lora_adapter(q_dual_agent* agent,
                                  q_agent_turn turn) {
    if (!agent) {
        return Q_ERR_NULL_PTR;
    }
    
    // Trocar contexto para Architect ou Auditor
    if (turn == TURN_ARCHITECT) {
        // Carregar architect_lora_weights no contexto
        agent->current_turn = TURN_ARCHITECT;
    } else if (turn == TURN_AUDITOR) {
        // Carregar auditor_lora_weights no contexto
        agent->current_turn = TURN_AUDITOR;
    } else {
        return Q_ERR_INVALID_ARG;
    }
    
    return Q_OK;
}

// Architect gera código
q_error_code q_architect_generate(q_dual_agent* agent,
                                   const char* problem,
                                   const char* language,
                                   char** out_code) {
    if (!agent || !problem || !out_code) {
        return Q_ERR_NULL_PTR;
    }
    
    // Carregar adaptador Architect
    q_error_code ret = q_load_lora_adapter(agent, TURN_ARCHITECT);
    if (ret != Q_OK) {
        return ret;
    }
    
    // Construir prompt com System Prompt do Architect
    char* prompt = q_build_architect_prompt(problem, language);
    
    // Gerar código usando modelo base + LoRA Architect
    ret = q_model_generate(agent->ctx,
                           agent->base_model_weights,
                           agent->architect_lora_weights,
                           prompt,
                           out_code);
    
    free(prompt);
    return ret;
}

// Auditor revisa código
q_error_code q_auditor_review(q_dual_agent* agent,
                               const char* code,
                               const char* language,
                               char** out_critique) {
    if (!agent || !code || !out_critique) {
        return Q_ERR_NULL_PTR;
    }
    
    // Carregar adaptador Auditor
    q_error_code ret = q_load_lora_adapter(agent, TURN_AUDITOR);
    if (ret != Q_OK) {
        return ret;
    }
    
    // Construir prompt com System Prompt do Auditor
    char* prompt = q_build_auditor_prompt(code, language);
    
    // Gerar crítica usando modelo base + LoRA Auditor
    ret = q_model_generate(agent->ctx,
                            agent->base_model_weights,
                            agent->auditor_lora_weights,
                            prompt,
                            out_critique);
    
    free(prompt);
    return ret;
}
```

### Economia de VRAM com LoRA Adapters

**Problema**: Dois modelos de 8B parâmetros (Architect + Auditor) podem estourar VRAM do Google Colab.

**Solução**: **LoRA Adapters (Low-Rank Adaptation)**

- **Modelo Base**: Llama-3 Coder 8B (~16GB VRAM) - **Congelado, compartilhado**
- **Architect LoRA**: ~2GB VRAM (pequeno adaptador)
- **Auditor LoRA**: ~2GB VRAM (pequeno adaptador)
- **Total**: ~18GB VRAM (vs ~32GB sem LoRA)
- **Economia**: 87.5% de redução de VRAM para adaptadores

**Implementação**:
- Carregar modelo base uma vez (congelado)
- Trocar apenas adaptadores LoRA durante execução (O(1) overhead)
- Especialização profunda através de fine-tuning dos adaptadores

---

## 🎓 PIPELINE DE TREINAMENTO DE ELITE

### Visão Geral

O **Qorus-IA Code Agent** será treinado com uma estratégia única que combina código de referência de alta qualidade, literatura técnica fundamental, problemas algorítmicos e aprendizado por reforço estilo AlphaZero.

**Objetivo**: Criar um modelo que não apenas gera código funcional, mas código que segue padrões de excelência técnica, compreende profundamente estruturas de dados e algoritmos, e aprende iterativamente através de auto-jogo (self-play).

### Fase 1: A Teoria (Books & Specs) - "Learning the Rules"

**Antes de ver código, a IA deve entender a engenharia.**

**Objetivo**: Aprender o que é um registrador, como funciona o Cache L1/L2, o que é Virtual Memory. Isso habilita o **"First Principles Thinking"**.

**Dataset**:

**Livros Fundamentais** (~20-25% do dataset):
- **The C Programming Language (K&R)**: Fundamentos sólidos da linguagem C, estilo clássico, elegância
- **Computer Systems: A Programmer's Perspective (CSAPP)**: 
  - Representação de dados (inteiros, ponto flutuante)
  - Assembly e arquitetura de processadores
  - Hierarquia de memória (cache, RAM, disco)
  - Linking e carregamento
  - Concorrência e sincronização
- **Introduction to Algorithms (CLRS)**: Algoritmos fundamentais
- **Algorithms (Sedgewick)**: Implementações práticas
- **Data Structures and Algorithm Analysis**: Análise de complexidade

**Manuais Técnicos** (~5-10% do dataset):
- **Intel SDM (Software Developer Manuals)**: Arquitetura x86-64, instruções AVX2/AVX-512
- **ARM Architecture Reference Manual**: Arquitetura ARM, instruções NEON
- **POSIX Manuals**: System calls, APIs padrão

**Formato**: Código de exemplo + explicações técnicas + provas matemáticas

**Total Fase 1**: ~25-35% do dataset

### Fase 2: A Prática de Elite (Style Transfer) - "Learning from Masters"

**Aqui moldamos a "personalidade" do código.**

**Objetivo**: Aprender padrões de código de produção, otimizações de baixo nível, estruturas de dados eficientes, estilo rigoroso.

#### 2.1 Código de Referência de Alta Qualidade

**Kernel Linux** (~15-20% do dataset)
- **Objetivo**: Aprender padrões de sistemas de baixo nível, gerenciamento de memória, concorrência, otimizações de performance
- **Fontes**: 
  - `linux/kernel/` - Core kernel code
  - `linux/mm/` - Memory management
  - `linux/fs/` - File systems
  - `linux/net/` - Network stack
- **Foco**: Padrões de código C de produção, estruturas de dados eficientes, macros e otimizações
- **Estratégia Especial**: Manter **árvore de diretórios** para entender dependências

**Doom / Quake (id Tech)** (~10-15% do dataset)
- **Objetivo**: Aprender código C extremamente otimizado, algoritmos de game engine, matemática vetorial rápida, truques de bits (Fast Inverse Square Root)
- **Fontes**:
  - `doom/doom/` - Game logic
  - `doom/doomdef.h` - Data structures
  - `doom/r_main.c` - Rendering optimizations
  - `quake/` - Quake engine code
- **Foco**: Performance crítica, otimizações de baixo nível, estruturas de dados compactas

**SQLite / Redis** (~5-10% do dataset)
- **SQLite**: Banco de dados robusto em C, arquitetura extremamente eficiente e estável
- **Redis**: Estruturas de dados e algoritmos eficientes

**Outras Referências de Qualidade** (~5-10% do dataset)
- **nginx**: Servidor web de alta performance
- **PostgreSQL**: Banco de dados relacional complexo
- **LLVM**: Compiladores e otimizações

**Estratégia de Dados Crítica**:
- ✅ **Manter árvore de diretórios**: Para entender dependências e contexto
- ✅ **Histórico de Commits de Fix**: Treinar com `(Code Before Bug) -> (Commit Message) -> (Code Fixed)`
  - Isso ensina o Auditor a corrigir erros
  - Formato: `{"instruction": "Analise este código inseguro", "input": "...", "output": "Correção com verificação de bounds..."}`

**Total Fase 2**: ~35-55% do dataset

#### 2.2 Mining de Commits (Bug -> Fix)

**Ferramenta**: `tools/miner_elite.py`

**Objetivo**: Extrair padrões de correção de bugs do histórico de commits de repositórios elite.

**Estratégia**:
1. Clonar repositórios (Linux, Doom, SQLite, Redis)
2. Filtrar commits que contêm: "Fix", "Bug", "Leak", "Optim", "Security"
3. Extrair:
   - Código antes do bug
   - Mensagem do commit
   - Código corrigido
4. Formatar para JSONL:
   ```json
   {
     "instruction": "Analise este código inseguro e corrija",
     "input": "void unsafe_copy(char* dest, char* src, int len) { memcpy(dest, src, len); }",
     "output": "void safe_copy(char* dest, char* src, size_t len) { if (dest && src && len > 0) { memcpy(dest, src, len); } }"
   }
   ```

**Benefício**: Ensina o Auditor a identificar e corrigir bugs comuns.

### Fase 3: O Dojo (Reinforcement Learning) - "AlphaZero Style"

**Após o Fine-Tuning, a IA treina sozinha no Google Colab.**

**Objetivo**: Aprendizado iterativo através de auto-jogo, melhoria contínua através de auto-avaliação.

#### 3.1 O Ambiente (Gym)

**Componentes**:
- **Script Python**: Gera problemas de algoritmos (ex: "Inverta uma Binary Tree sem usar recursão")
- **Compilador**: `gcc -O3 -fsanitize=address` (detecta memory leaks, buffer overflows)
- **Profiler**: Mede performance (tempo de execução, uso de memória)
- **Test Runner**: Executa testes automaticamente

#### 3.2 O Ciclo de Recompensa (Reward Function)

**Sistema de Pontuação Detalhado**:

| Evento | Pontuação | Descrição |
|--------|-----------|-----------|
| **Erro de Compilação** | -10 pts | Código não compila |
| **Crash (Segfault/ASAN)** | -20 pts | Memory leak ou buffer overflow detectado |
| **Funciona (Lento)** | +1 pt | Código funciona mas é ineficiente |
| **Funciona (Rápido/Memória Baixa)** | +50 pts | Otimização excelente (aqui ela aprende as otimizações do Doom) |
| **Código Limpo (Style Check)** | +5 pts | Segue padrões de estilo (Kernel Linux, Doom) |
| **Todos os Testes Passam** | +10 pts | Funcionalidade completa |
| **Complexidade Ótima** | +15 pts | Big O otimizado (O(N) vs O(N²)) |
| **SIMD Utilizado** | +10 pts | AVX2/AVX-512 usado corretamente |
| **Proof Matemático Correto** | +10 pts | Complexidade provada matematicamente |

**Total Máximo**: +100 pts (código perfeito)

**Total Mínimo**: -30 pts (código quebrado)

#### 3.3 Metodologia AlphaZero

**Componentes**:
1. **Self-Play**: O modelo gera código, executa testes, avalia qualidade
2. **Reinforcement Learning**: Recompensas baseadas na tabela acima
3. **Monte Carlo Tree Search (MCTS)**: Exploração de diferentes abordagens de implementação
4. **Value Network**: Avaliação de qualidade do código gerado
5. **Policy Network**: Decisões sobre qual código gerar

**Ciclo de Treinamento AlphaZero**:
```
┌─────────────────────────────────────────────────────────┐
│ 1. GENERATION: Architect gera código                   │
│ 2. EXECUTION: Executa testes automaticamente            │
│ 3. COMPILATION: GCC + ASAN valida segurança             │
│ 4. PROFILING: Mede performance (tempo/memória)        │
│ 5. EVALUATION: Calcula recompensa (reward function)    │
│ 6. LEARNING: Atualiza política baseado em recompensa   │
│ 7. ITERATION: Repete até convergência                  │
└─────────────────────────────────────────────────────────┘
```

**Critérios de Convergência**:
- Recompensa média > +80 pts por 100 iterações consecutivas
- Taxa de aprovação do Auditor > 90%
- Taxa de compilação bem-sucedida > 95%

**Total Fase 3**: ~10-15% do dataset (auto-gerado)

### LeetCode e Problemas Algorítmicos (~5-10% do dataset)

**Objetivo**: Resolver problemas algorítmicos complexos, aprender padrões comuns, otimização de soluções

**Estratégia de Cobertura**:
- **Easy**: 20% - Fundamentos, sintaxe básica
- **Medium**: 50% - Algoritmos intermediários, estruturas de dados
- **Hard**: 30% - Problemas complexos, otimizações avançadas

**Categorias Prioritárias**:
- Arrays & Strings
- Linked Lists
- Trees & Graphs
- Dynamic Programming
- Greedy Algorithms
- Backtracking
- Bit Manipulation
- System Design (simplificado)

**Formato**: Problema → Solução otimizada → Análise de complexidade → Testes

**Objetivo**: Resolver problemas algorítmicos complexos, aprender padrões comuns, otimização de soluções

**Estratégia de Cobertura**:
- **Easy**: 20% - Fundamentos, sintaxe básica
- **Medium**: 50% - Algoritmos intermediários, estruturas de dados
- **Hard**: 30% - Problemas complexos, otimizações avançadas

**Categorias Prioritárias**:
- Arrays & Strings
- Linked Lists
- Trees & Graphs
- Dynamic Programming
- Greedy Algorithms
- Backtracking
- Bit Manipulation
- System Design (simplificado)

**Formato**: Problema → Solução otimizada → Análise de complexidade → Testes

### Estrutura do Dataset Final

| Fase | Categoria | Percentual | Tamanho Estimado | Prioridade |
|------|----------|------------|------------------|------------|
| **Fase 1** | Livros Fundamentais | 20-25% | ~30-40GB | 🔴 Crítica |
| **Fase 1** | Manuais Técnicos | 5-10% | ~8-15GB | 🔴 Crítica |
| **Fase 2** | Kernel Linux | 15-20% | ~25-35GB | 🔴 Crítica |
| **Fase 2** | Doom/Quake | 10-15% | ~15-25GB | 🔴 Crítica |
| **Fase 2** | SQLite/Redis | 5-10% | ~8-15GB | 🟡 Alta |
| **Fase 2** | Outras Referências | 5-10% | ~8-15GB | 🟡 Alta |
| **Fase 2** | Mining Commits (Bug->Fix) | 5-10% | ~8-15GB | 🔴 Crítica |
| **Fase 3** | AlphaZero Self-Play | 10-15% | ~15-25GB | 🟡 Alta |
| **Extra** | LeetCode | 5-10% | ~8-15GB | 🟡 Alta |
| **TOTAL** | | 100% | ~125-200GB | |

### Pipeline de Preparação de Dados

**Ferramenta Principal**: `tools/miner_elite.py`

**Funcionalidades**:
1. Clonar repositórios elite (Linux, Doom, SQLite, Redis, etc)
2. Extrair funções C mantendo estrutura de diretórios
3. Filtrar commits com "Fix", "Bug", "Leak", "Optim"
4. Gerar pares (Code Before Bug) -> (Code Fixed)
5. Formatar para JSONL compatível com fine-tuning

**Estrutura de Dados**:

```c
// Estrutura para dataset de treinamento (New-QorusIA v3.0)
#include "qorus.h"

typedef struct {
    // Fase 1: Teoria
    q_code_sample* book_samples;         // CSAPP, K&R, CLRS, etc
    q_code_sample* manual_samples;       // Intel SDM, ARM, POSIX
    
    // Fase 2: Prática Elite
    q_code_sample* kernel_samples;      // Kernel Linux
    q_code_sample* doom_samples;         // Doom/Quake
    q_code_sample* sqlite_samples;       // SQLite
    q_code_sample* redis_samples;        // Redis
    q_code_sample* other_ref_samples;    // Outras referências
    
    // Fase 2: Mining Commits
    q_bug_fix_pair* bug_fix_pairs;       // (Code Before) -> (Code Fixed)
    
    // Fase 3: AlphaZero
    q_self_play_sample* self_play_samples; // Auto-gerado
    
    // Extra
    q_code_sample* leetcode_samples;     // LeetCode
    
    uint64_t total_samples;
    uint64_t total_tokens;
} q_training_dataset;

// Par Bug -> Fix
typedef struct {
    char* code_before;        // Código com bug
    char* commit_message;    // Mensagem do commit
    char* code_after;         // Código corrigido
    char* language;           // "c", "python", etc
    char* bug_type;          // "memory_leak", "buffer_overflow", etc
} q_bug_fix_pair;

// Preparar dataset balanceado
q_error_code q_prepare_training_dataset(
    const char* kernel_path,
    const char* doom_path,
    const char* sqlite_path,
    const char* redis_path,
    const char* reference_paths[],
    const char* books_paths[],
    const char* manuals_paths[],
    const char* leetcode_path,
    q_training_dataset** out_dataset
);
```

**Script Python**: `tools/miner_elite.py`

```python
#!/usr/bin/env python3
"""
Miner Elite: Extrai código de qualidade de repositórios elite
e gera dataset de treinamento para Qorus-IA Code Agent.
"""

import os
import subprocess
import json
import re
from pathlib import Path

def clone_repo(url, dest_dir):
    """Clona repositório se não existir."""
    if os.path.exists(dest_dir):
        print(f"✓ {dest_dir} já existe")
        return
    print(f"Clonando {url}...")
    subprocess.run(["git", "clone", url, dest_dir], check=True)

def extract_functions(c_file):
    """Extrai funções C de um arquivo."""
    # Implementação: parse C code, extract functions
    pass

def find_fix_commits(repo_path):
    """Encontra commits com 'Fix', 'Bug', 'Leak', 'Optim'."""
    result = subprocess.run(
        ["git", "-C", repo_path, "log", "--grep", "Fix|Bug|Leak|Optim", "--oneline"],
        capture_output=True, text=True
    )
    return result.stdout.split('\n')

def extract_bug_fix_pair(repo_path, commit_hash):
    """Extrai par (Code Before) -> (Code Fixed) de um commit."""
    # Implementação: git show, diff, extract code
    pass

def generate_jsonl(dataset_dir, output_file):
    """Gera arquivo JSONL para fine-tuning."""
    with open(output_file, 'w') as f:
        # Iterar sobre samples e escrever JSONL
        pass

if __name__ == "__main__":
    repos = {
        "linux": "https://github.com/torvalds/linux",
        "doom": "https://github.com/id-Software/DOOM",
        "sqlite": "https://www.sqlite.org/src",
        "redis": "https://github.com/redis/redis"
    }
    
    dataset_dir = "dataset_elite"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Clonar repositórios
    for name, url in repos.items():
        clone_repo(url, os.path.join(dataset_dir, name))
    
    # Extrair código e commits
    # Gerar JSONL
    generate_jsonl(dataset_dir, "dataset_elite.jsonl")
```

### Estratégia de Tokenização Multi-Linguagem

O tokenizer deve ser treinado especificamente para:
- **C**: Padrões do kernel Linux, Doom, SQLite
- **Python**: LeetCode solutions, CSAPP examples
- **JavaScript/TypeScript**: Code examples de livros
- **SQL**: PostgreSQL, SQLite queries
- **Markdown**: Documentação técnica (CSAPP, livros)

**Vocabulário Estimado**: 80k-120k tokens (incluindo tokens especiais para código)

---

## 🤝 ARQUITETURA COLABORATIVA: CODE AGENT + REVIEW AGENT

### Conceito: Dupla Especializada

O **Qorus-IA Code Agent** trabalha em colaboração com um **Review Agent** especializado. Esta arquitetura dual garante que todo código gerado seja revisado, testado e validado antes de ser considerado "pronto".

**Filosofia**: "Dois agentes são melhores que um" - especialização e validação cruzada.

### Code Agent: Gerador de Código

**Responsabilidades**:
- Geração de código baseada em requisitos
- Implementação seguindo TDD + MFR + CoT + Proof
- Geração de testes iniciais
- Proposta de modelo de dados (MFR)

**Especialização**: Criatividade, geração, implementação

### Review Agent: Validador de Qualidade

**Responsabilidades**:
- **Code Review**: Análise estática de código, detecção de bugs potenciais
- **Test Generation**: Geração adicional de testes (edge cases, stress tests)
- **Debug Analysis**: Identificação de problemas, sugestões de correção
- **Quality Assurance**: Validação contra padrões, métricas de qualidade
- **Proof Verification**: Verificação matemática de complexidade e corretude

**Especialização**: Análise crítica, validação, garantia de qualidade

### Arquitetura Colaborativa Detalhada

```
┌─────────────────────────────────────────────────────────────┐
│                    REQUEST (Desenvolvedor)                  │
│              "Implemente função de ordenação"               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              CODE AGENT (Geração)                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ FASE 0: CoT - Raciocínio                            │  │
│  │ FASE 0.5: Proof - Validação Matemática              │  │
│  │ FASE 1: MFR - Modelo de Dados                       │  │
│  │ FASE 2: TDD - Testes Iniciais                       │  │
│  │ FASE 3: Implementação - Código                      │  │
│  └───────────────────┬──────────────────────────────────┘  │
│                      │ Código + Testes + Modelo + Proof    │
│                      ↓                                      │
┌─────────────────────────────────────────────────────────────┐
│              REVIEW AGENT (Validação)                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Code Review: Análise estática                     │  │
│  │ 2. Test Expansion: Testes adicionais                 │  │
│  │ 3. Proof Verification: Validação matemática          │  │
│  │ 4. Quality Metrics: Complexidade, manutenibilidade   │  │
│  │ 5. Debug Analysis: Identificação de problemas        │  │
│  └───────────────────┬──────────────────────────────────┘  │
│                      │ Feedback + Testes Adicionais        │
│                      ↓                                      │
┌─────────────────────────────────────────────────────────────┐
│              CODE AGENT (Refinamento)                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ FASE 5: Refinamento baseado em feedback              │  │
│  │ - Corrigir problemas identificados                    │  │
│  │ - Adicionar testes faltantes                         │  │
│  │ - Melhorar qualidade                                 │  │
│  └───────────────────┬──────────────────────────────────┘  │
│                      │ Código Refinado                     │
│                      ↓                                      │
┌─────────────────────────────────────────────────────────────┐
│              REVIEW AGENT (Validação Final)                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Validação Final: Aprovação ou Rejeição               │  │
│  └───────────────────┬──────────────────────────────────┘  │
│                      │ Aprovado / Rejeitado                │
│                      ↓                                      │
┌─────────────────────────────────────────────────────────────┐
│                    RESPONSE (Desenvolvedor)               │
│              Código validado e testado                     │
└─────────────────────────────────────────────────────────────┘
```

### Implementação Técnica

```c
// Estrutura para Review Agent
typedef struct s_review_agent {
    t_model *review_model;              // Modelo especializado em revisão
    t_code_tokenizer *tokenizer;
    
    // Capacidades especializadas
    t_code_reviewer *reviewer;          // Análise estática
    t_test_generator *test_gen;         // Geração de testes
    t_debug_analyzer *debugger;         // Análise de debug
    t_quality_metrics *metrics;         // Métricas de qualidade
    t_proof_verifier *proof_verifier;   // Verificação de proofs
} t_review_agent;

// Resultado da revisão
typedef struct s_review_result {
    bool approved;                      // Aprovado ou não
    t_code_issue *issues;              // Lista de problemas encontrados
    uint32_t num_issues;
    t_proposed_test *additional_tests; // Testes adicionais sugeridos
    uint32_t num_additional_tests;
    t_quality_score *quality_score;    // Score de qualidade
    t_proof_verification *proof_check;  // Verificação de proof
    char *feedback;                    // Feedback textual
} t_review_result;

// Revisar código gerado pelo Code Agent
t_review_result *review_agent_review(t_review_agent *agent,
                                      const char *generated_code,
                                      const char *tests,
                                      const char *data_model,
                                      const t_mathematical_proof *proof,
                                      const char *language);

// Gerar testes adicionais
t_proposed_test *review_agent_generate_tests(t_review_agent *agent,
                                               const char *code,
                                               const char *existing_tests,
                                               const char *data_model,
                                               const char *language);

// Analisar problemas de debug
t_debug_analysis *review_agent_analyze_debug(t_review_agent *agent,
                                               const char *code,
                                               const char *test_output,
                                               const char *error_message);

// Verificar proof matemático
t_proof_verification *review_agent_verify_proof(t_review_agent *agent,
                                                 const char *code,
                                                 const t_mathematical_proof *proof);
```

### Fluxo Integrado Code Agent + Review Agent

```c
// Fluxo completo com colaboração
int execute_collaborative_flow(t_code_agent *code_agent,
                                 t_review_agent *review_agent,
                                 const char *requirement,
                                 const char *language,
                                 t_developer_feedback *feedback) {
    t_tdd_mfr_cot_proof_flow *flow = calloc(1, sizeof(t_tdd_mfr_cot_proof_flow));
    
    // CODE AGENT: Geração inicial
    int ret = execute_tdd_mfr_cot_proof_flow(flow, requirement, language, feedback);
    if (ret != 0) {
        return ret;
    }
    
    // REVIEW AGENT: Primeira revisão
    t_review_result *review = review_agent_review(review_agent,
                                                    flow->generated_code,
                                                    flow->tests->test_code,
                                                    flow->model->model_str,
                                                    flow->proof,
                                                    language);
    
    // Se não aprovado, refinar
    uint32_t iteration = 0;
    while (!review->approved && iteration < MAX_REVIEW_ITERATIONS) {
        // Adicionar testes sugeridos
        if (review->num_additional_tests > 0) {
            flow->tests = merge_tests(flow->tests, review->additional_tests);
        }
        
        // Corrigir problemas identificados
        flow->generated_code = fix_issues(flow->generated_code,
                                           review->issues,
                                           review->num_issues);
        
        // Re-executar testes
        flow->test_results = agent_run_tests(flow->generated_code, flow->tests);
        
        // Revisar novamente
        review = review_agent_review(review_agent,
                                     flow->generated_code,
                                     flow->tests->test_code,
                                     flow->model->model_str,
                                     flow->proof,
                                     language);
        iteration++;
    }
    
    if (review->approved) {
        return 0; // Sucesso
    } else {
        return -1; // Falhou após iterações
    }
}
```

### Treinamento do Review Agent

O **Review Agent** será treinado com foco em:
- **Code Review**: Dataset de código com bugs conhecidos + correções
- **Test Generation**: Problemas LeetCode + testes completos
- **Debug Analysis**: Stack traces + código problemático + soluções
- **Quality Metrics**: Código de alta qualidade vs código de baixa qualidade
- **Proof Verification**: Código + proofs matemáticos + validações

**Dataset Especializado**:
- 40% Code Review (bugs + correções)
- 30% Test Generation (problemas + testes completos)
- 20% Debug Analysis (erros + soluções)
- 10% Quality Metrics (exemplos bons vs ruins)

---


---

## 🎯 METODOLOGIA CORE: TDD + MFR + CoT + Proof

### ⚠️ IMPORTANTE: Metodologia Obrigatória

**TDD + MFR + CoT + Proof NÃO são opções** - são parte integrante do cerne do agente. Todo código gerado **DEVE** seguir este fluxo:

1. **MFR primeiro**: Modelo definido antes de qualquer código
2. **Proof obrigatório**: Validação matemática (complexidade, corretude, limites) antes da implementação
3. **TDD sempre**: Testes gerados antes da implementação
4. **CoT explícito**: Raciocínio documentado passo a passo
5. **Colaboração**: Desenvolvedor participa em cada fase

### Fluxo Core Integrado

```
┌─────────────────────────────────────────────────────────┐
│         FASE 0: CHAIN OF THOUGHT (CoT)                │
│         - Passo 1: Entender o problema                │
│         - Passo 2: Decompor em sub-problemas          │
│         - Passo 3: Identificar edge cases              │
│         - Passo 4: Verificar padrões existentes      │
└───────────────────────┬───────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────┐
│    FASE 0.5: MATHEMATICAL PROOF & COMPLEXITY ANALYSIS  │
│         - Time Complexity (Big O)                      │
│         - Space Complexity                             │
│         - Proof of Correctness (termination, bounds)  │
│         - Edge Case Proof (N=0, N=1, N=MAX)           │
│         - Numerical Stability (se aplicável)          │
└───────────────────────┬───────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────┐
│         FASE 1: MODEL-FIRST REASONING (MFR)            │
│         - Definir ENTITIES (estruturas de dados)      │
│         - Definir STATE VARIABLES (layout de memória) │
│         - Definir CONSTRAINTS (invariantes)           │
│         - Definir ACTIONS (protótipos de funções)      │
└───────────────────────┬───────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────┐
│         FASE 2: TEST-DRIVEN DESIGN (TDD)               │
│         - RED: Gerar testes primeiro                   │
│         - Definir casos de teste (unit + integration) │
│         - Validar testes compilam (mas falham)        │
└───────────────────────┬───────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────┐
│         FASE 3: IMPLEMENTAÇÃO (GREEN)                  │
│         - Gerar código baseado no modelo MFR          │
│         - Seguir raciocínio CoT                        │
│         - Respeitar provas matemáticas (Phase 0.5)    │
│         - Implementar para passar nos testes          │
└───────────────────────┬───────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────┐
│         FASE 4: VALIDAÇÃO E TESTES                     │
│         - Executar testes unitários                    │
│         - Executar testes de integração                │
│         - Validar contra modelo MFR                    │
│         - Verificar provas matemáticas                 │
└───────────────────────┬───────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────┐
│         FASE 5: REFINAMENTO (REFACTOR)                 │
│         - Se testes falharem: corrigir código          │
│         - Se modelo violado: ajustar implementação    │
│         - Se prova violada: revisar algoritmo          │
│         - Iterar até aprovação do desenvolvedor       │
└─────────────────────────────────────────────────────────┘
```

### Detalhamento da Fase 0.5: Mathematical Proof & Complexity Analysis

**CRITICAL:** Before defining the model, the agent must mathematically validate the proposed solution. **NO GUESSING ALLOWED.**

The Mathematical Proof phase must include:

1. **TIME COMPLEXITY (Big O):** Calculate theoretical Time Complexity. Is this optimal? Why?
   - Example: "Proposed solution is O(N). Naive is O(N^2). This is optimal because we visit each element once."
   - **Cache Complexity:** For data-oriented operations, analyze cache behavior (spatial/temporal locality).
   - **SIMD Efficiency:** For vectorized operations, prove that SIMD lanes are fully utilized (e.g., 8 floats per AVX2 register).

2. **SPACE COMPLEXITY:** Calculate memory overhead (auxiliary space, not input size).
   - Distinguish between in-place operations (O(1)) and operations requiring temporary buffers.
   - Document peak memory usage for multi-stage algorithms.

3. **PROOF OF CORRECTNESS:**
   - **Termination:** Prove the loop/recursion will finish (decreasing variant, bounded iteration).
   - **Bounds:** Prove that indices `i` will strictly stay within `0 <= i < size` (no buffer overflows).
   - **Arithmetic:** Prove that operations (e.g., `a + b`) will not overflow/underflow for expected types, or how overflow is handled.
   - **Alignment:** Prove that SIMD operations access aligned memory (e.g., `ptr % 32 == 0` for AVX2, `ptr % 64 == 0` for AVX-512).

4. **EDGE CASE PROOF:** Mathematically demonstrate behavior at boundaries:
   - **N=0:** Loop condition fails immediately, returns safe state.
   - **N=1:** Single-element case handled correctly.
   - **N=MAX:** No integer overflow in loop counters or array indices.
   - **Special Values:** NaN, Inf, denormals propagate correctly (for floating-point).

5. **NUMERICAL STABILITY (for floating-point operations):**
   - Prove that operations maintain numerical precision (e.g., Kahan summation for reductions).
   - Document any approximations or trade-offs (e.g., fast approximations vs. exact computation).
   - Prove that rounding errors accumulate within acceptable bounds.

6. **TRIVIAL PROOF SHORTCUT:**
   - For obviously correct operations (e.g., element-wise addition), a brief statement suffices:
     *"Trivial: O(N) time, O(1) space. Bounds: `i < n` guarantees termination and safety. Alignment: Precondition ensures 64-byte alignment."*

### Implementação do Fluxo

```c
// Estrutura para fluxo completo TDD + MFR + CoT + Proof
typedef struct s_tdd_mfr_cot_proof_flow {
    // Fase 0: CoT
    t_cot_reasoning *reasoning;
    bool reasoning_approved;
    
    // Fase 0.5: Mathematical Proof
    t_mathematical_proof *proof;
    bool proof_validated;
    
    // Fase 1: MFR
    t_proposed_model *model;
    bool model_approved;
    
    // Fase 2: TDD
    t_proposed_tests *tests;
    bool tests_approved;
    
    // Fase 3: Implementação
    char *generated_code;
    
    // Fase 4: Validação
    t_test_results *test_results;
    bool all_tests_passing;
    
    // Fase 5: Refinamento
    uint32_t iteration_count;
    bool code_approved;
} t_tdd_mfr_cot_proof_flow;

// Executar fluxo completo
int execute_tdd_mfr_cot_proof_flow(t_tdd_mfr_cot_proof_flow *flow,
                                    const char *requirement,
                                    const char *language,
                                    t_developer_feedback *feedback) {
    // FASE 0: CoT - Raciocínio Passo a Passo
    flow->reasoning = agent_generate_reasoning(requirement, language);
    flow->reasoning_approved = true;
    
    // FASE 0.5: Mathematical Proof - Validação Matemática OBRIGATÓRIA
    flow->proof = agent_generate_proof(requirement,
                                        flow->reasoning,
                                        language);
    if (!flow->proof || !validate_proof(flow->proof)) {
        return -1; // Proof inválido - não prosseguir
    }
    flow->proof_validated = true;
    
    // FASE 1: MFR - Definir Modelo baseado em Proof validado
    flow->model = agent_propose_model(requirement,
                                      language,
                                      flow->proof);
    flow->model_approved = true;
    
    // FASE 2: TDD - Gerar Testes PRIMEIRO
    flow->tests = agent_propose_tests(flow->model,
                                      requirement,
                                      flow->proof);
    flow->tests_approved = true;
    
    // FASE 3: Implementação - Código que respeita Proof e passa nos testes
    flow->generated_code = agent_generate_code(flow->model,
                                               flow->tests,
                                               flow->reasoning,
                                               flow->proof);
    
    // FASE 4: Validação - Executar testes e verificar Proof
    flow->test_results = agent_run_tests(flow->generated_code, flow->tests);
    flow->all_tests_passing = (flow->test_results->failures == 0);
    
    bool proof_respected = verify_code_against_proof(flow->generated_code,
                                                     flow->proof);
    
    // FASE 5: Refinamento iterativo se necessário
    while ((!flow->all_tests_passing || !proof_respected) &&
           flow->iteration_count < MAX_ITERATIONS) {
        if (!flow->all_tests_passing) {
            flow->generated_code = agent_fix_failing_tests(flow->generated_code,
                                                           flow->test_results,
                                                           flow->model,
                                                           flow->proof);
        }
        if (!proof_respected) {
            flow->generated_code = agent_fix_proof_violations(flow->generated_code,
                                                              flow->proof,
                                                              flow->model);
        }
        flow->test_results = agent_run_tests(flow->generated_code, flow->tests);
        flow->all_tests_passing = (flow->test_results->failures == 0);
        proof_respected = verify_code_against_proof(flow->generated_code,
                                                     flow->proof);
        flow->iteration_count++;
    }
    
    return (flow->all_tests_passing && proof_respected) ? 0 : -1;
}
```

### Critérios de "Done"

```c
typedef struct s_done_criteria {
    bool compiles_successfully;    // Código compila sem erros
    bool all_tests_passing;        // Todos os testes passam
    bool proof_validated;          // Proof matemático é respeitado
    bool model_validated;          // Modelo MFR é respeitado
    bool user_approved;            // Usuário aprova explicitamente
    bool syntax_valid;             // Sintaxe válida
    uint32_t iteration_count;      // Número de iterações
} t_done_criteria;

bool is_code_done(t_done_criteria *criteria) {
    return criteria->compiles_successfully &&
           criteria->all_tests_passing &&
           criteria->proof_validated &&
           criteria->model_validated &&
           criteria->user_approved &&
           criteria->syntax_valid &&
           criteria->iteration_count < MAX_ITERATIONS;
}
```

### Recuperação de Falhas (Self-Healing)

```c
typedef enum {
    ERROR_COMPILATION,
    ERROR_TEST_FAILURE,
    ERROR_PROOF_VIOLATION,
    ERROR_MODEL_VIOLATION,
    ERROR_SYNTAX,
    ERROR_TIMEOUT
} t_error_type;

typedef enum {
    RECOVERY_REGENERATE_BLOCK,
    RECOVERY_REGENERATE_FUNCTION,
    RECOVERY_REFINE_INCREMENTAL,
    RECOVERY_FALLBACK_SIMPLE,
    RECOVERY_ASK_USER
} t_recovery_strategy;

// Analisar erro e propor estratégia
t_error_recovery *analyze_error(const char *error_output,
                                 const char *generated_code,
                                 t_test_results *test_results) {
    t_error_recovery *recovery = calloc(1, sizeof(t_error_recovery));
    
    if (strstr(error_output, "error:")) {
        recovery->error_type = ERROR_COMPILATION;
        recovery->strategy = RECOVERY_REGENERATE_BLOCK;
    } else if (test_results && test_results->failures > 0) {
        recovery->error_type = ERROR_TEST_FAILURE;
        recovery->strategy = RECOVERY_REFINE_INCREMENTAL;
    } else {
        recovery->strategy = RECOVERY_ASK_USER;
    }
    
    return recovery;
}
```

---

## ✅ O QUE JÁ EXISTE VS O QUE PRECISA SER FEITO

### ✅ O que Já Existe no Qorus-IA

- ✅ **Transformer Block completo** (MHA + FFN + LayerNorm)
- ✅ **RoPE** (Rotary Positional Embeddings)
- ✅ **Causal Masking** (necessário para geração autoregressiva)
- ✅ **Optimizers** (Adam, AdamW)
- ✅ **Loss functions** (CrossEntropy)
- ✅ **Performance otimizada** (157.79 GFLOPS)
- ✅ **Memory management** (64-byte aligned)
- ✅ **Thread-safe operations** (OpenMP compatible)

### ❌ O que Precisa ser Implementado

#### Fase 1: Base LLM (Crítico - Bloqueador)

| Componente | Horas | Prioridade | Descrição |
|------------|-------|------------|-----------|
| **Tokenizer Multi-Linguagem** | 16-22h | 🔴 Crítica | BPE/SentencePiece para C, Python, JS, TS, PHP, SQL |
| **Embedding Layer** | 3-4h | 🔴 Crítica | Token embeddings + integração RoPE |
| **Decoder Stack** | 4-6h | 🔴 Crítica | Empilhar Transformer Blocks (12-24 layers) |
| **LM Head** | 2-3h | 🔴 Crítica | Projeção final (embed_dim → vocab_size) |
| **Generation Loop** | 8-10h | 🔴 Crítica | Loop autoregressivo + sampling |
| **Subtotal** | **33-45h** | | |

#### Fase 2: Especialização Multi-Linguagem

| Componente | Horas | Prioridade | Descrição |
|------------|-------|------------|-----------|
| **Syntax-Aware Generation** | 4-6h | 🟡 Alta | Validação de sintaxe durante geração |
| **Context Manager** | 2-3h | 🟡 Alta | Extração de contexto do cursor |
| **Multi-file Context** | 2-3h | 🟡 Alta | Suporte a múltiplos arquivos |
| **Subtotal** | **8-12h** | | |

#### Fase 3: TDD + MFR + CoT + Proof Core (Crítico - Diferencial)

| Componente | Horas | Prioridade | Descrição |
|------------|-------|------------|-----------|
| **Templates de Prompt** | 8-12h | 🔴 Crítica | build_mfr_prompt, build_tdd_prompt, build_cot_prompt, build_proof_prompt |
| **Geração Automática de Proofs** | 10-14h | 🔴 Crítica | agent_generate_proof() - complexidade, corretude, limites |
| **Geração Automática de Testes** | 12-16h | 🔴 Crítica | agent_propose_tests() baseado em modelo MFR e proof |
| **Execução e Validação** | 8-10h | 🔴 Crítica | agent_run_tests(), validate_against_model(), verify_proof() |
| **Refinamento Iterativo** | 6-8h | 🔴 Crítica | agent_fix_failing_tests(), agent_fix_proof_violations(), refine_code() |
| **Integração Core** | 4-6h | 🔴 Crítica | execute_tdd_mfr_cot_proof_flow() completo |
| **Subtotal** | **48-66h** | | |

#### Fase 3.5: Review Agent (Crítico - Garantia de Qualidade)

| Componente | Horas | Prioridade | Descrição |
|------------|-------|------------|-----------|
| **Review Agent Model** | 12-16h | 🔴 Crítica | Modelo especializado em revisão de código |
| **Code Review Engine** | 8-10h | 🔴 Crítica | Análise estática, detecção de bugs potenciais |
| **Test Generation Engine** | 6-8h | 🔴 Crítica | Geração de testes adicionais (edge cases, stress) |
| **Debug Analysis Engine** | 6-8h | 🔴 Crítica | Análise de erros, identificação de problemas |
| **Quality Metrics** | 4-6h | 🟡 Alta | Métricas de qualidade (complexidade, manutenibilidade) |
| **Proof Verification Engine** | 4-6h | 🔴 Crítica | Verificação matemática de proofs |
| **Integração Colaborativa** | 4-6h | 🔴 Crítica | execute_collaborative_flow() Code + Review |
| **Subtotal** | **44-60h** | | |

#### Fase 0: Preparação de Dataset e Treinamento (Pré-requisito)

| Componente | Horas | Prioridade | Descrição |
|------------|-------|------------|-----------|
| **Coleta Código de Referência** | 20-30h | 🔴 Crítica | Kernel Linux, Doom, SQLite, Redis, etc |
| **Processamento Literatura Técnica** | 15-20h | 🔴 Crítica | CSAPP, K&R, livros de estruturas de dados |
| **Preparação LeetCode Dataset** | 10-15h | 🟡 Alta | Problemas + soluções otimizadas |
| **Pipeline AlphaZero** | 20-30h | 🟡 Alta | Self-play, MCTS, reinforcement learning |
| **Tokenização Multi-Linguagem** | 8-12h | 🔴 Crítica | BPE/SentencePiece para dataset completo |
| **Treinamento Code Agent** | 80-120h | 🔴 Crítica | Fine-tuning com dataset especializado |
| **Treinamento Review Agent** | 60-90h | 🔴 Crítica | Treinamento especializado em revisão |
| **Subtotal** | **213-317h** | | |

#### Fase 4: Funcionalidades Básicas

| Componente | Horas | Prioridade | Descrição |
|------------|-------|------------|-----------|
| **Code Completion** | 8-10h | 🟡 Média | Autocomplete baseado em contexto |
| **Code Generation** | 6-8h | 🟡 Média | Geração a partir de descrição |
| **Code Refactoring** | 8-10h | 🟡 Média | Refatoração guiada |
| **Code Explanation** | 4-6h | 🟡 Média | Explicação de código |
| **Subtotal** | **26-34h** | | |

#### Fase 5: Design-to-Code (Opcional - Feature Diferencial)

| Componente | Horas | Prioridade | Descrição |
|------------|-------|------------|-----------|
| **Vision Processing** | 20-30h | 🟡 Alta | Carregamento, preprocessamento, ViT/CNN |
| **Design Analysis** | 12-18h | 🟡 Alta | Detecção componentes, layout, cores |
| **Code Generation Frontend** | 8-12h | 🟡 Alta | Geração React/Vue com TDD+MFR+CoT |
| **Visual Validation** | 8-12h | 🟡 Alta | Renderização, comparação visual |
| **Subtotal** | **48-72h** | | |

#### Fase 6: Integração

| Componente | Horas | Prioridade | Descrição |
|------------|-------|------------|-----------|
| **LSP Server** | 12-16h | 🟡 Alta | Language Server Protocol completo |
| **Chat Interno** | 20-30h | 🟡 Média | Interface web simples para equipe |
| **Subtotal** | **32-46h** | | |

### Resumo Executivo

| Fase | Horas | Prioridade | Bloqueador? |
|------|-------|------------|-------------|
| **Preparação Dataset e Treinamento** | 213-317h | 🔴 Crítica | ✅ Sim |
| **Base LLM** | 33-45h | 🔴 Crítica | ✅ Sim |
| **Especialização** | 8-12h | 🟡 Alta | ✅ Sim |
| **TDD+MFR+CoT+Proof Core** | 48-66h | 🔴 Crítica | ✅ Sim |
| **Review Agent** | 44-60h | 🔴 Crítica | ✅ Sim |
| **Funcionalidades** | 26-34h | 🟡 Média | ❌ Não |
| **Design-to-Code** | 48-72h | 🟡 Alta | ❌ Não |
| **Integração** | 32-46h | 🟡 Alta | ⚠️ Parcial |
| **TOTAL** | **452-658h** | | |

**MVP Mínimo**: 346-498h (Dataset + Treinamento + Base LLM + Especialização + TDD+MFR+CoT+Proof + Review Agent básico)  
**Produto Completo**: 404-578h (MVP + Funcionalidades + Integração)  
**Produto Premium**: 452-658h (Todos os componentes incluindo Design-to-Code)

**Nota**: As horas de treinamento (213-317h) podem ser executadas em paralelo com desenvolvimento, reduzindo tempo total do projeto.

---

## 🛣️ ROADMAP PROGRESSIVO DE IMPLEMENTAÇÃO

### Princípio: Implementar do Mais Fácil para o Mais Difícil

Seguindo a filosofia MFR + Proof + TDD + CoT, implementamos em ordem crescente de complexidade:

### Etapa 1: Base LLM (33-45h) - FUNDAÇÃO

**Objetivo**: Ter um LLM funcional básico capaz de gerar texto/código.

#### 1.1 Tokenizer Multi-Linguagem (16-22h)

**Estratégia Pragmática:**
1. **Referência e Análise** (2-3h): Estudar `llama.cpp`, `tiktoken`, `sentencepiece`
2. **Porte Inicial** (Opcional - 4-6h): Portar implementação de referência para validação
3. **Reimplementação Otimizada** (10-13h): Reescrever do zero adaptado ao Qorus-IA

**Estrutura de Dados:**

```c
typedef struct s_code_tokenizer {
    char **vocab;                    // Vocabulário (~50k-100k tokens)
    uint32_t vocab_size;
    uint32_t *bpe_merges;           // Regras BPE
    char **supported_languages;     // ["c", "python", "javascript", "typescript", "php", "sql"]
    uint32_t num_languages;
    void *lookup_cache;             // Cache 64-byte aligned
    bool use_avx2;
    bool use_avx512;
} t_code_tokenizer;

// API
t_code_tokenizer *code_tokenizer_create_multi(const char *vocab_path,
                                              const char **languages,
                                              uint32_t num_languages);

uint32_t *code_tokenizer_encode(t_code_tokenizer *tok, 
                                 const char *code,
                                 const char *language,
                                 uint32_t *out_len);

char *code_tokenizer_decode(t_code_tokenizer *tok,
                            const uint32_t *tokens,
                            uint32_t len,
                            const char *language);
```

#### 1.2 Embedding Layer (3-4h)

```c
typedef struct s_embedding {
    t_tensor *weight;       // [vocab_size, embed_dim]
    uint32_t vocab_size;
    uint32_t embed_dim;
} t_embedding;

t_embedding *embedding_create(uint32_t vocab_size, uint32_t embed_dim);
t_tensor *embedding_forward(t_embedding *emb, 
                            const uint32_t *token_ids, 
                            uint32_t batch_size,
                            uint32_t seq_len);
```

#### 1.3 Decoder Stack (4-6h)

```c
typedef struct s_decoder_stack {
    t_transformer_block **blocks;  // Array de N blocks
    uint32_t num_layers;
    t_layer_layernorm *final_norm;
    uint32_t embed_dim;
    t_rope_cache *rope_cache;
} t_decoder_stack;

t_decoder_stack *decoder_stack_create(uint32_t num_layers,
                                      uint32_t embed_dim,
                                      uint32_t num_heads,
                                      uint32_t hidden_dim,
                                      t_rope_cache *rope_cache,
                                      float dropout_p);
```

#### 1.4 LM Head (2-3h)

```c
typedef struct s_lm_head {
    t_layer_linear *proj;   // [embed_dim, vocab_size]
    bool weight_tied;
    t_embedding *tied_embedding;
} t_lm_head;

t_lm_head *lm_head_create(uint32_t embed_dim, 
                          uint32_t vocab_size,
                          bool weight_tied);
```

#### 1.5 Generation Loop (8-10h)

```c
typedef struct s_code_generation_config {
    uint32_t max_new_tokens;
    float temperature;
    uint32_t top_k;
    float top_p;
    bool syntax_check;
    const char *language;
} t_code_generation_config;

uint32_t *code_agent_generate(t_model *llm,
                                const char *prompt_code,
                                const char *context_code,
                                t_code_generation_config *config,
                                uint32_t *out_len);
```

### Etapa 2: Especialização Multi-Linguagem (8-12h)

**Objetivo**: Adaptar LLM para código multi-linguagem.

- Syntax-aware generation (4-6h)
- Context Manager (2-3h)
- Multi-file Context (2-3h)

### Etapa 3: TDD + MFR + CoT + Proof Core (48-66h) - DIFERENCIAL ÚNICO

**Objetivo**: Implementar metodologia core que gera código sempre testado e matematicamente validado.

#### 3.1 Templates de Prompt (8-12h)

```c
char *build_cot_prompt(const char *description, const char *language);
char *build_proof_prompt(const char *description, t_cot_reasoning *reasoning);
char *build_mfr_prompt(const char *description, const char *language, t_mathematical_proof *proof);
char *build_tdd_prompt(const char *description, t_proposed_model *model, t_mathematical_proof *proof);
char *build_impl_prompt(const char *description,
                        const char *data_model,
                        const char *reasoning_steps,
                        const char *proof_details,
                        const char *language);
```

#### 3.2 Geração Automática de Proofs (10-14h)

```c
typedef struct s_mathematical_proof {
    char *time_complexity;        // "O(N)", "O(N log N)", etc.
    char *space_complexity;        // "O(1)", "O(N)", etc.
    char *termination_proof;       // Prova de terminação
    char *bounds_proof;            // Prova de limites (0 <= i < n)
    char *edge_cases;              // N=0, N=1, N=MAX
    char *numerical_stability;     // Estabilidade numérica (se aplicável)
    bool is_trivial;              // Se é prova trivial
} t_mathematical_proof;

t_mathematical_proof *agent_generate_proof(const char *description,
                                            t_cot_reasoning *reasoning,
                                            const char *language);

bool validate_proof(t_mathematical_proof *proof);
bool verify_code_against_proof(const char *code, t_mathematical_proof *proof);
```

#### 3.3 Geração Automática de Testes (12-16h)

```c
t_proposed_tests *agent_propose_tests(t_proposed_model *model,
                                      t_mathematical_proof *proof,
                                      const char *description);
```

#### 3.4 Execução e Validação (8-10h)

```c
t_test_results *agent_run_tests(const char *code,
                                 t_proposed_tests *tests);

bool validate_against_model(const char *code, const char *data_model);
bool verify_code_against_proof(const char *code, t_mathematical_proof *proof);
```

#### 3.5 Refinamento Iterativo (6-8h)

```c
char *agent_fix_failing_tests(const char *code,
                               t_test_results *results,
                               t_proposed_model *model,
                               t_mathematical_proof *proof);

char *agent_fix_proof_violations(const char *code,
                                  t_mathematical_proof *proof,
                                  t_proposed_model *model);

char *refine_code(t_code_agent *agent,
                  const char *current_code,
                  const char *data_model,
                  const char *reasoning_steps,
                  const char *proof_details);
```

#### 3.6 Integração Core (4-6h)

```c
int execute_tdd_mfr_cot_proof_flow(t_tdd_mfr_cot_proof_flow *flow,
                                    const char *requirement,
                                    const char *language,
                                    t_developer_feedback *feedback);
```

### Etapa 4: Funcionalidades Básicas (26-34h)

- Code Completion (8-10h)
- Code Generation (6-8h)
- Code Refactoring (8-10h)
- Code Explanation (4-6h)

### Etapa 5: Design-to-Code (48-72h) - Opcional

- Vision Processing (20-30h)
- Design Analysis (12-18h)
- Code Generation Frontend (8-12h)
- Visual Validation (8-12h)

### Etapa 6: Integração (32-46h)

- LSP Server (12-16h)
- Chat Interno (20-30h)

---

## 📐 ESPECIFICAÇÕES TÉCNICAS

### Convenções de Código Qorus-IA

#### Naming Conventions

```c
// Prefixo obrigatório: q_ para todas funções públicas (New-QorusIA v3.0)
// Funções internas podem usar prefixo específico do módulo
q_error_code q_function_name(const q_tensor* restrict input, 
                              q_tensor* restrict output,
                              q_context* restrict ctx);

// Naming: snake_case sempre
typedef struct s_struct_name {
    // campos
} t_struct_name;

// Constantes: UPPER_SNAKE_CASE
#define MAX_DIMS 8
```

#### Error Handling

```c
// Sempre retornar q_error_code (New-QorusIA v3.0)
q_error_code q_function(const q_tensor* restrict input,
                        q_tensor* restrict output,
                        q_context* restrict ctx) {
    Q_VALIDATE_PTR_OR_RETURN(input);
    Q_VALIDATE_PTR_OR_RETURN(output);
    Q_VALIDATE_PTR_OR_RETURN(ctx);
    
    // Implementação...
    
    return Q_OK;
}
```

#### Memory Management

```c
// TODOS tensores devem ser 64-byte aligned (New-QorusIA v3.0)
// Usar q_arena_alloc() para alocação na Arena (zero-malloc no hot path)
q_error_code q_create_tensor(q_context* restrict ctx,
                              const uint32_t shape[4],
                              q_tensor* restrict out) {
    Q_VALIDATE_PTR_OR_RETURN(ctx);
    Q_VALIDATE_PTR_OR_RETURN(out);
    
    size_t size = calculate_size(shape);
    void* data = q_arena_alloc(ctx, size * sizeof(float));
    if (!data) {
        return Q_ERR_OOM;
    }
    
    // Inicializar q_tensor...
    
    return Q_OK;
}
```

#### Thread Safety

```c
// NUNCA usar estado global mutável
// SEMPRE usar parâmetros para estado
int ft_function_with_state(t_state *state, const t_tensor *input);
```

### Padrões de Teste (TDD)

```c
#include "tensor.h"
#include "tests/common/test_utils.h"

static int test_suite_basic(void)
{
    int failures = 0;
    test_suite_start("Basic Functionality");
    
    {
        t_tensor *input = tensor_create((uint32_t[]){4}, 1);
        t_tensor *output = tensor_create((uint32_t[]){4}, 1);
        
        int ret = ft_function(input, output);
        bool passed = (ret == 0) && (output->data[0] == expected_value);
        
        failures += test_result("Basic case", passed, NULL);
        
        tensor_free(input);
        tensor_free(output);
    }
    
    test_suite_end(failures);
    return failures;
}

int main(void)
{
    double start = get_time_ms();
    int failures = 0;
    
    print_test_header("Component Test Suite");
    failures += test_suite_basic();
    
    double elapsed = get_time_ms() - start;
    print_test_footer(0, failures, elapsed);
    
    return failures > 0 ? 1 : 0;
}
```

### Estruturas de Dados Padrão

```c
#define MAX_DIMS 8

// Usar tipos do New-QorusIA v3.0
// q_tensor já definido em include/qorus_types.h
// q_context já definido em include/qorus_types.h
// q_error_code já definido em include/qorus_types.h

typedef struct {
    q_context* ctx;                    // Contexto de memória New-QorusIA
    q_tokenizer* tokenizer;            // Tokenizer multi-linguagem
    
    // Base Model (congelado, compartilhado)
    void* base_model_weights;          // Pesos do modelo base (mmap)
    
    // LoRA Adapters (pequenos, trocáveis)
    void* architect_lora_weights;      // Pesos do adaptador Architect (~2GB)
    void* auditor_lora_weights;        // Pesos do adaptador Auditor (~2GB)
    
    // Core Flow
    q_tdd_mfr_cot_proof_flow* core_flow;
    q_code_generation_config default_config;
} q_code_agent;
```

### Constraints e Invariantes

**Memory Constraints:**
- ✅ Todos tensors devem ser 64-byte aligned
- ✅ Nenhuma alocação dentro de loops (hot path)
- ✅ Zero-copy quando possível (views)
- ✅ Sempre liberar memória alocada

**Numerical Constraints:**
- ✅ Validação científica contra Python/PyTorch
- ✅ Tolerância híbrida (absolute + relative error)
- ✅ Tratamento de NaN/Inf

**Thread Safety Constraints:**
- ✅ Sem estado global mutável
- ✅ Funções thread-safe (OpenMP compatible)
- ✅ Usar `restrict` quando seguro

---

## 🔌 INTEGRAÇÃO E DEPLOY

### Arquitetura: PhpStorm + LSP Server Direto via SSH

**Decisão**: PhpStorm como editor principal devido a SSH nativo, LSP completo e Remote Development.

**Arquitetura:**

```
PhpStorm (Cliente)
    ↕ SSH Tunnel (automático)
Servidor (tempo-main)
    ↕ IPC (stdio)
Qorus-IA LSP Server (C Native)
```

### LSP Server Implementation

```c
// src/lsp/ft_lsp_server.c
#include "tensor.h"
#include "code_agent.h"

typedef struct s_lsp_server {
    t_code_agent *agent;
    FILE *stdin;
    FILE *stdout;
    bool initialized;
} t_lsp_server;

int lsp_stdio_mode(void) {
    t_code_agent *agent = code_agent_create("models/code_model.mia", 
                                            "vocabs/code_vocab.json");
    t_lsp_server server = {
        .agent = agent,
        .stdin = stdin,
        .stdout = stdout,
        .initialized = false
    };
    
    char buffer[8192];
    while (fgets(buffer, sizeof(buffer), stdin)) {
        json_t *request = json_parse(buffer);
        json_t *response = process_lsp_request(&server, request);
        char *response_str = json_stringify(response);
        fprintf(stdout, "%s\n", response_str);
        fflush(stdout);
        free(response_str);
    }
    
    code_agent_free(agent);
    return 0;
}

int main(int argc, char **argv) {
    return lsp_stdio_mode();
}
```

### Chat Interno Simplificado

**Estrutura no tempo-main:**

```
tempo-main/src/modules/code-agent-nd/
├── controllers/ChatController.ts
├── services/CodeAgentService.ts
├── websocket/ChatWebSocket.ts
└── views/chat.html
```

**Comando simples:**

```bash
# /usr/local/bin/qorus-ia-chat
#!/bin/bash
cd /path/to/tempo-main
npm run start:chat &
sleep 2
xdg-open "http://localhost:3000/code-agent/chat"
```

**Uso:**

```bash
qorus-ia-chat  # Abre interface web automaticamente
```

### Deployment

```bash
# Compilar LSP Server
cd /path/to/qorus-ia
make lsp-server

# Instalar
sudo cp build/qorus-ia-lsp /usr/local/bin/
sudo chmod +x /usr/local/bin/qorus-ia-lsp

# Estrutura recomendada
/opt/qorus-ia/
├── bin/qorus-ia-lsp
├── models/code_model.mia
└── vocabs/code_vocab.json
```

---

## 📚 REFERÊNCIAS E CHECKLIST

### Referências Técnicas

- **GPT-2**: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- **Codex**: "Evaluating Large Language Models Trained on Code" (Chen et al., 2021)
- **AlphaZero**: "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (Silver et al., 2017)
- **llama.cpp**: Implementação C++ otimizada
- **tiktoken**: BPE robusto (OpenAI)
- **LSP**: Language Server Protocol

### Referências de Código de Qualidade

- **Linux Kernel**: https://github.com/torvalds/linux
- **Doom (id Software)**: https://github.com/id-Software/DOOM
- **SQLite**: https://www.sqlite.org/
- **Redis**: https://github.com/redis/redis
- **nginx**: https://github.com/nginx/nginx
- **PostgreSQL**: https://github.com/postgres/postgres
- **LLVM**: https://github.com/llvm/llvm-project

### Referências de Literatura Técnica

- **CSAPP**: "Computer Systems: A Programmer's Perspective" (Bryant & O'Hallaron, 3rd Edition)
- **K&R**: "The C Programming Language" (Kernighan & Ritchie, 2nd Edition)
- **CLRS**: "Introduction to Algorithms" (Cormen, Leiserson, Rivest, Stein, 4th Edition)
- **Sedgewick**: "Algorithms" (Sedgewick & Wayne, 4th Edition)
- **LeetCode**: https://leetcode.com/

### Checklist de Implementação

#### Fase 1: Base LLM
- [ ] Tokenizer multi-linguagem (16-22h)
- [ ] Embedding Layer (3-4h)
- [ ] Decoder Stack (4-6h)
- [ ] LM Head (2-3h)
- [ ] Generation Loop (8-10h)

#### Fase 2: Especialização
- [ ] Syntax-aware generation (4-6h)
- [ ] Context Manager (2-3h)
- [ ] Multi-file Context (2-3h)

#### Fase 0: Preparação Dataset e Treinamento
- [ ] Coleta código de referência (20-30h)
- [ ] Processamento literatura técnica (15-20h)
- [ ] Preparação LeetCode dataset (10-15h)
- [ ] Pipeline AlphaZero (20-30h)
- [ ] Tokenização multi-linguagem (8-12h)
- [ ] Treinamento Code Agent (80-120h)
- [ ] Treinamento Review Agent (60-90h)

#### Fase 3: TDD + MFR + CoT + Proof Core
- [ ] Templates de prompt (8-12h)
- [ ] Geração automática de proofs (10-14h)
- [ ] Geração automática de testes (12-16h)
- [ ] Execução e validação (8-10h)
- [ ] Refinamento iterativo (6-8h)
- [ ] Integração core (4-6h)

#### Fase 3.5: Review Agent
- [ ] Review Agent Model (12-16h)
- [ ] Code Review Engine (8-10h)
- [ ] Test Generation Engine (6-8h)
- [ ] Debug Analysis Engine (6-8h)
- [ ] Quality Metrics (4-6h)
- [ ] Proof Verification Engine (4-6h)
- [ ] Integração colaborativa (4-6h)

#### Fase 4: Funcionalidades
- [ ] Code Completion (8-10h)
- [ ] Code Generation (6-8h)
- [ ] Code Refactoring (8-10h)
- [ ] Code Explanation (4-6h)

#### Fase 5: Design-to-Code (Opcional)
- [ ] Vision Processing (20-30h)
- [ ] Design Analysis (12-18h)
- [ ] Code Generation Frontend (8-12h)
- [ ] Visual Validation (8-12h)

#### Fase 6: Integração
- [ ] LSP Server (12-16h)
- [ ] Chat Interno (20-30h)

---

## 🎓 CONCLUSÃO

**Status Atual**: ~30-40% completo (especificação)

**MVP Funcional**: 346-498 horas (~9-12 semanas full-time)
- Preparação Dataset + Treinamento + Base LLM + Especialização + TDD+MFR+CoT+Proof + Review Agent básico

**Produto Completo**: 404-578 horas (~10-14 semanas full-time)
- MVP + Funcionalidades + Integração

**Produto Premium**: 452-658 horas (~11-16 semanas full-time)
- Todos os componentes incluindo Design-to-Code

**Bloqueadores Críticos:**
1. **Preparação Dataset e Treinamento** (213-317h) - Base fundamental para qualidade
2. **Base LLM** (33-45h) - Sem isso, nada funciona
3. **TDD+MFR+CoT+Proof Core** (48-66h) - Diferencial competitivo único
4. **Review Agent** (44-60h) - Garantia de qualidade através de validação colaborativa

**Diferenciais Únicos:**
1. **Treinamento Especializado**: Kernel Linux + Doom + CSAPP + LeetCode + AlphaZero
2. **Arquitetura Colaborativa**: Code Agent + Review Agent trabalhando em conjunto
3. **Metodologia Rigorosa**: TDD + MFR + CoT + Proof integrado e obrigatório

**Recomendação**: 
- **Fase 1**: Preparar dataset e treinar modelos em paralelo com desenvolvimento (213-317h)
- **Fase 2**: Implementar MVP funcional (Base LLM + Especialização + TDD+MFR+CoT+Proof + Review Agent básico)
- **Fase 3**: Expandir com funcionalidades e integração
- **Fase 4**: Adicionar Design-to-Code (opcional)

---

**Última Atualização**: 2024-12-29  
**Versão**: v3.1.0 (Elite System - Dual-Agent Architecture)  
**Metodologia Core**: TDD + MFR + CoT + Proof (Integrado e Obrigatório)  
**Arquitetura**: Architect + Auditor (Dual-Agent com LoRA Adapters)  
**Engine**: Qorus-IA v3.0 (C/CUDA Hybrid)  
**Base de Conhecimento**: Elite Repos (Linux/Doom) + Livros de Engenharia + AlphaZero
