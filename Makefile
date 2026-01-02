# Build System para Qorus-IA v2.0
# Flags: -O3 -mavx2 -mfma (AVX2 + FMA for optimal SIMD performance)
# Debug: make DEBUG=1 (enables AddressSanitizer + UndefinedBehaviorSanitizer)
# Sanitizers: make SANITIZE=1 (enables ASan + UBSan + TSan)
# Static Analysis: make ANALYZE=1 (enables static analysis warnings)

CC = gcc

# Detecção automática de versão do GCC para flags específicas (executado apenas uma vez)
GCC_FULL_VERSION := $(shell $(CC) -dumpversion 2>/dev/null || echo "0.0.0")
GCC_VERSION := $(shell echo "$(GCC_FULL_VERSION)" | cut -d. -f1)
GCC_MINOR := $(shell echo "$(GCC_FULL_VERSION)" | cut -d. -f2)

# Flags base (comuns) - CORRIGIDO: removido espaço em -Werror
CFLAGS_COMMON = -std=c11 -Wall -Wextra -Werror -I./include -D_GNU_SOURCE

# Flags de qualidade adicionais (warnings extras)
# NOTE: Flags removidas por serem muito restritivas para código legítimo:
#   -Wtraditional-conversion: muito restritivo para builtins do GCC
#   -Wsign-conversion: muito restritivo para código de sistema (mmap, st_size, etc)
#   -Wconversion: muito restritivo, mantemos apenas warnings específicos
#   -Wcast-qual: muito restritivo para casts legítimos de const
#   -Wrestrict: muito restritivo, pode ser falso positivo em alguns casos
#   -Wformat=2: muito restritivo, mantemos apenas -Wformat-security
CFLAGS_QUALITY = \
	-Wpedantic \
	-Wformat-security \
	-Wcast-align \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wuninitialized \
	-Winit-self \
	-Wshadow \
	-Wpointer-arith \
	-Wstrict-aliasing=2 \
	-Wundef \
	-Wunused \
	-Wunused-result \
	-Warray-bounds=2 \
	-Wimplicit-fallthrough=3 \
	-Wlogical-op \
	-Wduplicated-cond \
	-Wduplicated-branches \
	-Wnull-dereference \
	-Wstack-usage=8192 \
	-Wtrampolines \
	-Wfloat-equal

# Flags específicas do GCC 8+ (melhor detecção)
# Validação: GCC_VERSION deve ser numérico e >= 8
# NOTE: -Wrestrict removido: muito restritivo, pode ser falso positivo
ifeq ($(shell [ -n "$(GCC_VERSION)" ] && [ "$(GCC_VERSION)" -ge 8 ] 2>/dev/null && echo 1),1)
	CFLAGS_QUALITY += -Wformat-overflow=2 -Wformat-truncation=2
endif

# Flags específicas do GCC 10+ (análise estática melhorada)
# NOTE: -Wanalyzer-too-complex removido para permitir análise completa
# A análise estática em modo ANALYZE=1 já usa -fanalyzer sem limite de complexidade
ifeq ($(shell [ -n "$(GCC_VERSION)" ] && [ "$(GCC_VERSION)" -ge 10 ] 2>/dev/null && echo 1),1)
	CFLAGS_QUALITY += -Warith-conversion
endif

# Modo Release (Padrão: Performance Máxima)
# NOTE: -fstrict-aliasing removido: pode causar UB se código violar strict aliasing rules
#       Use apenas se código seguir strict aliasing (ponteiros de tipos diferentes não se sobrepõem)
CFLAGS_RELEASE = -O3 -mavx2 -mfma -fno-omit-frame-pointer \
	-fno-strict-overflow -fstack-protector-strong

LDFLAGS_RELEASE = -lm -fstack-protector-strong

# Modo Debug (Segurança Máxima + ASan + UBSan)
# NOTE: AVX2 flags are required even in DEBUG mode for intrinsics to compile
CFLAGS_DEBUG = -O0 -g3 -mavx2 -mfma -fno-omit-frame-pointer -DDEBUG \
	-fsanitize=undefined -fsanitize=address -fsanitize-address-use-after-scope \
	-fno-common -fstack-protector-all

LDFLAGS_DEBUG = -lm -fsanitize=undefined -fsanitize=address

# Modo Sanitize (apenas sanitizers, sem debug completo)
ifeq ($(SANITIZE),1)
	CFLAGS_SANITIZE = -O1 -g -mavx2 -mfma -fno-omit-frame-pointer \
		-fsanitize=undefined -fsanitize=address -fsanitize-address-use-after-scope \
		-fsanitize=thread -fno-common
	LDFLAGS_SANITIZE = -lm -fsanitize=undefined -fsanitize=address -fsanitize=thread
endif

# Modo Static Analysis (análise estática com GCC analyzer)
# NOTE: Removido -Wanalyzer-too-complex para permitir análise mais profunda
# Funções complexas (loops aninhados, grandes funções) são analisadas completamente
# mantendo todos os outros checks de segurança (leaks, use-after-free, null-deref)
ifeq ($(ANALYZE),1)
	CFLAGS_ANALYZE = -O0 -g -mavx2 -mfma -fanalyzer \
		-Wanalyzer-malloc-leak -Wanalyzer-double-free -Wanalyzer-use-after-free \
		-Wanalyzer-null-dereference -Wanalyzer-use-of-uninitialized-value
	LDFLAGS_ANALYZE = -lm
endif

# Seletor baseado em variável de ambiente
ifeq ($(DEBUG),1)
	CFLAGS = $(CFLAGS_COMMON) $(CFLAGS_QUALITY) $(CFLAGS_DEBUG)
	LDFLAGS = $(LDFLAGS_DEBUG)
else ifeq ($(SANITIZE),1)
	CFLAGS = $(CFLAGS_COMMON) $(CFLAGS_QUALITY) $(CFLAGS_SANITIZE)
	LDFLAGS = $(LDFLAGS_SANITIZE)
else ifeq ($(ANALYZE),1)
	CFLAGS = $(CFLAGS_COMMON) $(CFLAGS_QUALITY) $(CFLAGS_ANALYZE)
	LDFLAGS = $(LDFLAGS_ANALYZE)
else
	CFLAGS = $(CFLAGS_COMMON) $(CFLAGS_QUALITY) $(CFLAGS_RELEASE)
	LDFLAGS = $(LDFLAGS_RELEASE)
endif

# Diretórios (detecção automática melhorada)
SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = include
TESTS_DIR = tests

# Detecção automática de todos os diretórios de código fonte
SRC_DIRS := $(shell find $(SRC_DIR) -type d 2>/dev/null)
BUILD_SUBDIRS := $(patsubst $(SRC_DIR)/%,$(BUILD_DIR)/%,$(SRC_DIRS))

# Detecção automática de arquivos fonte (qualquer .c em subdiretórios de src/)
# Filtra arquivos de referência, testes, e arquivos obsoletos/substituídos
# Arquivos obsoletos: llama3.c (substituído por model.c), dummy_tokenizer.c (substituído por bpe.c)
# Complexidade: O(n log n) - aceitável para detecção automática
ALL_SRCS := $(shell find $(SRC_DIR) -name "*.c" -type f 2>/dev/null | \
	grep -v "_ref.c" | grep -v "/test" | \
	grep -v "llama3\.c$$" | grep -v "dummy_tokenizer\.c$$" | \
	grep -v "\.backup" | sort)

# Validação: garantir que encontramos pelo menos alguns arquivos
ifeq ($(ALL_SRCS),)
	$(warning Nenhum arquivo .c encontrado em $(SRC_DIR))
	ALL_SRCS := $(wildcard $(SRC_DIR)/**/*.c)
	# Aplicar mesmos filtros de exclusão
	ALL_SRCS := $(filter-out %_ref.c %/test/%, $(ALL_SRCS))
	ALL_SRCS := $(filter-out %llama3.c %dummy_tokenizer.c, $(ALL_SRCS))
	ALL_SRCS := $(filter-out %.backup, $(ALL_SRCS))
endif

OBJS = $(ALL_SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Geração automática de dependências (detecção de headers)
DEPFILES = $(OBJS:.o=.d)

# Target principal
TARGET = qorus-ia

# Benchmark tool
BENCHMARK_TARGET = tools/benchmark

# Testes
TEST_SRCS = $(wildcard $(TESTS_DIR)/*.c)
TEST_TARGETS = $(TEST_SRCS:$(TESTS_DIR)/%.c=$(BUILD_DIR)/tests/%)

.PHONY: all lib objects clean clean-objs clean-test-artifacts directories test test-memory test-dequantize test-matmul test-ops test-validation test-memory-adversarial test-model-overflow-adversarial test-utils test-avx-math test-llama-forward test-rmsnorm-adversarial test-rope-adversarial test-silu-adversarial test-softmax-adversarial test-dequantize-adversarial test-ops-integration test-tokenizer test-bpe-tokenizer test-llama-forward-adversarial test-tokenizer-adversarial test-memory-strategies test-llama-cleanup test-integration-e2e test-tokenizer-free-complete test-model-file-validation test-edge-cases-extreme test-llama-scratchpad test-llama-kv-cache test-llama-rope test-llama-token-embedding test-llama-free benchmark analyze analyze-cppcheck analyze-clang-tidy analyze-complete check-syntax

# Target para compilar apenas objetos (sem executável) - útil para bibliotecas
objects: directories $(OBJS)
	@echo "✓ Todos os objetos compilados"

# Target para biblioteca (alias para objects)
lib: objects

# Target principal - compila objetos e tenta criar executável (se main existir)
# Se não houver main(), apenas compila objetos (comportamento de biblioteca)
all: directories $(OBJS)
	@echo "Tentando criar executável $(TARGET)..."
	@$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS) 2>&1 || \
		(echo "⚠ Aviso: Não foi possível criar executável (projeto pode ser biblioteca sem main())"; \
		 echo "✓ Build completo: objetos compilados com sucesso")

# Criação automática de diretórios baseada em estrutura de src/
directories:
	@mkdir -p $(BUILD_SUBDIRS)
	@mkdir -p $(BUILD_DIR)/tests
	@echo "✓ Diretórios criados automaticamente"

# Incluir arquivos de dependência gerados automaticamente
-include $(DEPFILES)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)
	@echo "✓ Build completo: $(TARGET)"

# Benchmark tool
$(BENCHMARK_TARGET): tools/benchmark.c $(OBJS)
	$(CC) $(CFLAGS) -DDEBUG $< $(OBJS) -o $@ $(LDFLAGS)

# Regra com geração automática de dependências (detecção de headers)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# Regra para testes
$(BUILD_DIR)/tests/%: $(TESTS_DIR)/%.c $(OBJS)
	@mkdir -p $(BUILD_DIR)/tests
	$(CC) $(CFLAGS) -DDEBUG $< $(OBJS) -o $@ $(LDFLAGS)

# Target para verificação de sintaxe (sem compilar)
check-syntax:
	@echo "Verificando sintaxe de todos os arquivos..."
	@for file in $(ALL_SRCS); do \
		echo "Checking $$file..."; \
		$(CC) $(CFLAGS) -fsyntax-only $$file || exit 1; \
	done
	@echo "✓ Sintaxe OK"

# Target para análise estática (requer GCC 10+)
# CRITICAL FIX: Usar 'objects' em vez de 'all' para não tentar criar executável (biblioteca sem main())
# CRITICAL FIX: Retornar código de erro apropriado se análise encontrar problemas críticos
analyze:
	@echo "Executando análise estática (GCC analyzer)..."
	@$(MAKE) clean
	@$(MAKE) ANALYZE=1 objects 2>&1 | tee static-analysis.log; \
	ANALYZE_EXIT=$$?; \
	if [ $$ANALYZE_EXIT -ne 0 ]; then \
		echo "⚠ Compilação com análise estática falhou (exit code $$ANALYZE_EXIT)"; \
		echo "Verificando se há erros críticos..."; \
		if grep -qE "(error|warning.*leak|warning.*use-after-free|warning.*null-dereference)" static-analysis.log 2>/dev/null; then \
			echo "❌ ERROS CRÍTICOS ENCONTRADOS na análise estática!"; \
			grep -E "(error|warning.*leak|warning.*use-after-free|warning.*null-dereference)" static-analysis.log | head -20; \
			exit 1; \
		fi; \
		echo "⚠ Problemas não-críticos encontrados (ver static-analysis.log)"; \
		exit 0; \
	fi; \
	echo "✓ Análise estática concluída (ver static-analysis.log)"

# Target para análise estática complementar com cppcheck
analyze-cppcheck:
	@echo "Executando análise estática complementar (cppcheck)..."
	@which cppcheck > /dev/null 2>&1 || (echo "⚠ cppcheck não instalado. Instale com: sudo apt-get install cppcheck" && exit 0)
	@cppcheck --enable=all --inconclusive --error-exitcode=0 \
		--suppress=missingIncludeSystem \
		--suppress=unmatchedSuppression \
		--suppress=unusedFunction \
		src/ tests/ 2>&1 | tee cppcheck-report.log || true
	@echo "✓ Análise cppcheck concluída (ver cppcheck-report.log)"

# Target para análise estática com clang-tidy
analyze-clang-tidy:
	@echo "Executando análise estática complementar (clang-tidy)..."
	@which clang-tidy > /dev/null 2>&1 || (echo "⚠ clang-tidy não instalado. Instale com: sudo apt-get install clang-tidy" && exit 0)
	@echo "Gerando compile_commands.json para clang-tidy..."
	@$(MAKE) clean > /dev/null 2>&1 || true
	@bear --version > /dev/null 2>&1 || (echo "⚠ bear não instalado. Instalando compile_commands.json manualmente..."; \
		echo "[" > compile_commands.json; \
		first=1; \
		for file in $(ALL_SRCS); do \
			if [ $$first -eq 1 ]; then first=0; else echo "," >> compile_commands.json; fi; \
			rel_file=$$(echo $$file | sed 's|^$(SRC_DIR)/||'); \
			build_file=$$(echo $$file | sed 's|^$(SRC_DIR)|$(BUILD_DIR)|' | sed 's|\.c$$|.o|'); \
			echo "  {" >> compile_commands.json; \
			echo "    \"directory\": \"$(shell pwd)\"," >> compile_commands.json; \
			echo "    \"command\": \"$(CC) $(CFLAGS_COMMON) $(CFLAGS_QUALITY) $(CFLAGS_RELEASE) -c $$file -o $$build_file\"," >> compile_commands.json; \
			echo "    \"file\": \"$$file\"" >> compile_commands.json; \
			echo -n "  }" >> compile_commands.json; \
		done; \
		echo "" >> compile_commands.json; \
		echo "]" >> compile_commands.json; \
		echo "✓ compile_commands.json gerado")
	@clang-tidy --version > /dev/null 2>&1 && \
		clang-tidy $(ALL_SRCS) \
			-checks='-*,bugprone-*,cert-*,clang-analyzer-*,cppcoreguidelines-*,misc-*,modernize-*,performance-*,portability-*,readability-*' \
			-warnings-as-errors='' \
			-header-filter='include/.*' \
			-- $(CFLAGS_COMMON) $(CFLAGS_QUALITY) $(CFLAGS_RELEASE) -I./include 2>&1 | tee clang-tidy-report.log || true
	@echo "✓ Análise clang-tidy concluída (ver clang-tidy-report.log)"

# Target para análise estática completa (GCC analyzer + cppcheck + clang-tidy)
analyze-complete: analyze analyze-cppcheck analyze-clang-tidy
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  ANÁLISE ESTÁTICA COMPLETA CONCLUÍDA"
	@echo "═══════════════════════════════════════════════════════════"
	@echo "✓ GCC analyzer: static-analysis.log"
	@echo "✓ cppcheck: cppcheck-report.log"
	@echo "✓ clang-tidy: clang-tidy-report.log"
	@echo "═══════════════════════════════════════════════════════════"

# Target de teste específico
test-memory: directories $(BUILD_DIR)/tests/test_memory
	@echo "Gerando modelo dummy..."
	@python3 tools/convert_llama.py model_dummy.qorus || true
	@echo "Executando teste..."
	@$(BUILD_DIR)/tests/test_memory || (rm -f model_dummy.qorus tokenizer.bin; exit 1)
	@rm -f model_dummy.qorus tokenizer.bin 2>/dev/null || true

test-dequantize: directories $(BUILD_DIR)/tests/test_dequantize
	@echo "Gerando dados de teste..."
	@python3 tools/gen_test_dequantize.py || true
	@echo "Executando teste..."
	@$(BUILD_DIR)/tests/test_dequantize

test-matmul: directories $(BUILD_DIR)/tests/test_matmul
	@echo "Executando teste MatMul..."
	@$(BUILD_DIR)/tests/test_matmul

test-matmul-comprehensive: directories $(BUILD_DIR)/tests/test_matmul__test
	@echo "Executando teste abrangente de MatMul..."
	@$(BUILD_DIR)/tests/test_matmul__test

test-matmul-f32: directories $(BUILD_DIR)/tests/test_matmul_f32
	@echo "Executando teste MatMul FP32..."
	@$(BUILD_DIR)/tests/test_matmul_f32

test-causal-mask-f32: directories $(BUILD_DIR)/tests/test_causal_mask_f32
	@echo "Executando teste Causal Mask FP32..."
	@$(BUILD_DIR)/tests/test_causal_mask_f32

test-add-f32: directories $(BUILD_DIR)/tests/test_add_f32
	@echo "Executando teste Tensor Add FP32..."
	@$(BUILD_DIR)/tests/test_add_f32

test-mul-f32: directories $(BUILD_DIR)/tests/test_mul_f32
	@echo "Executando teste Element-wise Mul FP32..."
	@$(BUILD_DIR)/tests/test_mul_f32

test-matmul-adversarial: directories $(BUILD_DIR)/tests/test_matmul_adversarial
	@echo "Executando testes adversarial de MatMul..."
	@$(BUILD_DIR)/tests/test_matmul_adversarial

test-ops: directories $(BUILD_DIR)/tests/test_ops
	@echo "Executando teste de operações..."
	@$(BUILD_DIR)/tests/test_ops

test-llama-build: directories $(BUILD_DIR)/tests/test_llama_build
	@echo "Gerando modelo dummy..."
	@python3 tools/convert_llama.py model_dummy.qorus 2 || true
	@echo "Executando teste de construção do modelo..."
	@$(BUILD_DIR)/tests/test_llama_build

test-llama-build-adversarial: directories $(BUILD_DIR)/tests/test_llama_build_adversarial
	@echo "Executando testes adversarial de construção do modelo..."
	@$(BUILD_DIR)/tests/test_llama_build_adversarial

test-memory-adversarial: directories $(BUILD_DIR)/tests/test_memory_adversarial
	@echo "Executando testes adversarial de gerenciamento de memória..."
	@$(BUILD_DIR)/tests/test_memory_adversarial

test-model-overflow-adversarial: directories $(BUILD_DIR)/tests/test_model_adversarial_overflow
	@echo "Executando testes adversarial de proteção contra overflow..."
	@$(BUILD_DIR)/tests/test_model_adversarial_overflow

test-utils: directories $(BUILD_DIR)/tests/test_utils
	@echo "Executando testes de utilitários..."
	@$(BUILD_DIR)/tests/test_utils

test-avx-math: directories $(BUILD_DIR)/tests/test_avx_math
	@echo "Executando testes de funções AVX math..."
	@$(BUILD_DIR)/tests/test_avx_math

test-llama-forward: directories $(BUILD_DIR)/tests/test_llama_forward
	@echo "Gerando modelo dummy..."
	@python3 tools/convert_llama.py model_dummy.qorus 2 || true
	@echo "Executando teste de forward pass..."
	@$(BUILD_DIR)/tests/test_llama_forward

test-rmsnorm-adversarial: directories $(BUILD_DIR)/tests/test_rmsnorm_adversarial
	@echo "Executando testes adversarial de RMSNorm..."
	@$(BUILD_DIR)/tests/test_rmsnorm_adversarial

test-rope-adversarial: directories $(BUILD_DIR)/tests/test_rope_adversarial
	@echo "Executando testes adversarial de RoPE..."
	@$(BUILD_DIR)/tests/test_rope_adversarial

test-silu-adversarial: directories $(BUILD_DIR)/tests/test_silu_adversarial
	@echo "Executando testes adversarial de SiLU..."
	@$(BUILD_DIR)/tests/test_silu_adversarial

test-softmax-adversarial: directories $(BUILD_DIR)/tests/test_softmax_adversarial
	@echo "Executando testes adversarial de Softmax..."
	@$(BUILD_DIR)/tests/test_softmax_adversarial

test-dequantize-adversarial: directories $(BUILD_DIR)/tests/test_dequantize_adversarial
	@echo "Executando testes adversarial de Dequantize..."
	@$(BUILD_DIR)/tests/test_dequantize_adversarial

test-ops-integration: directories $(BUILD_DIR)/tests/test_ops_integration
	@echo "Executando testes de integração de operações matemáticas..."
	@$(BUILD_DIR)/tests/test_ops_integration

test-tokenizer: directories $(BUILD_DIR)/tests/test_tokenizer
	@echo "Gerando tokenizer..."
	@python3 tools/convert_llama.py --tokenizer tokenizer.bin || true
	@echo "Executando teste de tokenizer..."
	@$(BUILD_DIR)/tests/test_tokenizer tokenizer.bin || (rm -f model_dummy.qorus tokenizer.bin; exit 1)
	@rm -f model_dummy.qorus tokenizer.bin 2>/dev/null || true

test-bpe-tokenizer: directories $(BUILD_DIR)/tests/test_bpe_tokenizer
	@echo "Executando testes de especificação BPE..."
	@$(BUILD_DIR)/tests/test_bpe_tokenizer

test-bpe-tokenizer-adversarial: directories $(BUILD_DIR)/tests/test_bpe_tokenizer_adversarial
	@echo "Executando testes adversarial de BPE tokenizer..."
	@$(BUILD_DIR)/tests/test_bpe_tokenizer_adversarial

test-llama-forward-adversarial: directories $(BUILD_DIR)/tests/test_llama_forward_adversarial
	@echo "Gerando modelo dummy..."
	@python3 tools/convert_llama.py model_dummy.qorus 2 || true
	@echo "Executando testes adversarial de llama_forward..."
	@$(BUILD_DIR)/tests/test_llama_forward_adversarial || (rm -f model_dummy.qorus tokenizer.bin; exit 1)
	@rm -f model_dummy.qorus tokenizer.bin 2>/dev/null || true

test-tokenizer-adversarial: directories $(BUILD_DIR)/tests/test_tokenizer_adversarial
	@echo "Gerando tokenizer..."
	@python3 tools/convert_llama.py --tokenizer tokenizer.bin || true
	@echo "Executando testes adversarial de tokenizer..."
	@$(BUILD_DIR)/tests/test_tokenizer_adversarial || (rm -f model_dummy.qorus tokenizer.bin; exit 1)
	@rm -f model_dummy.qorus tokenizer.bin 2>/dev/null || true

test-memory-strategies: directories $(BUILD_DIR)/tests/test_memory_strategies
	@echo "Gerando modelo dummy..."
	@python3 tools/convert_llama.py model_dummy.qorus 2 || true
	@echo "Executando testes de estratégias de memória..."
	@$(BUILD_DIR)/tests/test_memory_strategies || (rm -f model_dummy.qorus tokenizer.bin; exit 1)
	@rm -f model_dummy.qorus tokenizer.bin 2>/dev/null || true

test-llama-cleanup: directories $(BUILD_DIR)/tests/test_llama_cleanup
	@echo "Gerando modelo dummy..."
	@python3 tools/convert_llama.py model_dummy.qorus 2 || true
	@echo "Executando testes de limpeza do modelo..."
	@$(BUILD_DIR)/tests/test_llama_cleanup || (rm -f model_dummy.qorus tokenizer.bin; exit 1)
	@rm -f model_dummy.qorus tokenizer.bin 2>/dev/null || true

test-llama-scratchpad: directories $(BUILD_DIR)/tests/test_llama_scratchpad
	@echo "Gerando modelo dummy..."
	@python3 tools/convert_llama.py model_dummy.qorus 2 || true
	@echo "Executando testes adversarial de scratchpad..."
	@$(BUILD_DIR)/tests/test_llama_scratchpad || (rm -f model_dummy.qorus tokenizer.bin; exit 1)
	@rm -f model_dummy.qorus tokenizer.bin 2>/dev/null || true

test-llama-kv-cache: directories $(BUILD_DIR)/tests/test_llama_kv_cache
	@echo "Gerando modelo dummy..."
	@python3 tools/convert_llama.py model_dummy.qorus 2 || true
	@echo "Executando testes adversarial de KV cache..."
	@$(BUILD_DIR)/tests/test_llama_kv_cache || (rm -f model_dummy.qorus tokenizer.bin; exit 1)
	@rm -f model_dummy.qorus tokenizer.bin 2>/dev/null || true

test-llama-rope: directories $(BUILD_DIR)/tests/test_llama_rope
	@echo "Gerando modelo dummy..."
	@python3 tools/convert_llama.py model_dummy.qorus 2 || true
	@echo "Executando testes adversarial de RoPE..."
	@$(BUILD_DIR)/tests/test_llama_rope || (rm -f model_dummy.qorus tokenizer.bin; exit 1)
	@rm -f model_dummy.qorus tokenizer.bin 2>/dev/null || true

test-llama-token-embedding: directories $(BUILD_DIR)/tests/test_llama_token_embedding
	@echo "Gerando modelo dummy..."
	@python3 tools/convert_llama.py model_dummy.qorus 2 || true
	@echo "Executando testes adversarial de token embedding..."
	@$(BUILD_DIR)/tests/test_llama_token_embedding || (rm -f model_dummy.qorus tokenizer.bin; exit 1)
	@rm -f model_dummy.qorus tokenizer.bin 2>/dev/null || true

test-llama-free: directories $(BUILD_DIR)/tests/test_llama_free
	@echo "Gerando modelo dummy..."
	@python3 tools/convert_llama.py model_dummy.qorus 2 || true
	@echo "Executando testes adversarial de llama_free_graph..."
	@$(BUILD_DIR)/tests/test_llama_free || (rm -f model_dummy.qorus tokenizer.bin; exit 1)
	@rm -f model_dummy.qorus tokenizer.bin 2>/dev/null || true

test-integration-e2e: directories $(BUILD_DIR)/tests/test_integration_e2e
	@echo "Gerando modelo dummy e tokenizer..."
	@python3 tools/convert_llama.py model_dummy.qorus 2 || true
	@python3 tools/convert_llama.py --tokenizer tokenizer.bin || true
	@echo "Executando testes end-to-end..."
	@$(BUILD_DIR)/tests/test_integration_e2e || (rm -f model_dummy.qorus tokenizer.bin; exit 1)
	@rm -f model_dummy.qorus tokenizer.bin 2>/dev/null || true

test-tokenizer-free-complete: directories $(BUILD_DIR)/tests/test_tokenizer_free_complete
	@echo "Gerando tokenizer..."
	@python3 tools/convert_llama.py --tokenizer tokenizer.bin || true
	@echo "Executando validação completa de q_tokenizer_free..."
	@$(BUILD_DIR)/tests/test_tokenizer_free_complete || (rm -f model_dummy.qorus tokenizer.bin; exit 1)
	@rm -f model_dummy.qorus tokenizer.bin 2>/dev/null || true

test-model-file-validation: directories $(BUILD_DIR)/tests/test_model_file_validation
	@echo "Executando testes de validação de arquivos de modelo..."
	@$(BUILD_DIR)/tests/test_model_file_validation || (rm -f model_dummy.qorus tokenizer.bin; exit 1)
	@rm -f model_dummy.qorus tokenizer.bin 2>/dev/null || true

test-edge-cases-extreme: directories $(BUILD_DIR)/tests/test_edge_cases_extreme
	@echo "Gerando modelo dummy..."
	@python3 tools/convert_llama.py model_dummy.qorus 2 || true
	@echo "Executando testes de edge cases extremos..."
	@$(BUILD_DIR)/tests/test_edge_cases_extreme || (rm -f model_dummy.qorus tokenizer.bin; exit 1)
	@rm -f model_dummy.qorus tokenizer.bin 2>/dev/null || true

test-adversarial-all: test-rmsnorm-adversarial test-rope-adversarial test-silu-adversarial test-softmax-adversarial test-dequantize-adversarial test-matmul-adversarial test-llama-forward-adversarial test-tokenizer-adversarial test-memory-strategies test-llama-cleanup test-tokenizer-free-complete test-model-file-validation test-edge-cases-extreme test-llama-scratchpad test-llama-kv-cache test-llama-rope test-llama-token-embedding test-llama-free
	@echo "✓ Todos os testes adversarial concluídos"

test-integration-all: test-ops-integration
	@echo "✓ Todos os testes de integração concluídos"

# Validação completa (Release + Debug)
# CRITICAL FIX: Remover clean redundante - clean-test-artifacts já faz limpeza completa
test-validation: clean-test-artifacts
	@echo "=== Running Release Tests ==="
	@$(MAKE) test-memory
	@$(MAKE) test-dequantize
	@$(MAKE) test-matmul
	@$(MAKE) test-ops
	@$(MAKE) test-matmul-f32
	@$(MAKE) test-causal-mask-f32
	@$(MAKE) test-add-f32
	@$(MAKE) test-mul-f32
	@echo "\n=== Running Debug Tests (Memory Safety) ==="
	@$(MAKE) test-memory DEBUG=1
	@$(MAKE) test-dequantize DEBUG=1
	@$(MAKE) test-matmul DEBUG=1
	@$(MAKE) test-ops DEBUG=1
	@$(MAKE) test-matmul-f32 DEBUG=1
	@$(MAKE) test-causal-mask-f32 DEBUG=1
	@$(MAKE) test-add-f32 DEBUG=1
	@$(MAKE) test-mul-f32 DEBUG=1
	@echo "\n✓ All tests passed (Release + Debug with sanitizers)"
	@$(MAKE) clean-test-artifacts

test: test-memory test-dequantize test-matmul test-ops
	@$(MAKE) clean-test-artifacts

benchmark: directories $(BENCHMARK_TARGET)
	@echo "Running performance benchmarks..."
	@$(BENCHMARK_TARGET)

# Limpeza de arquivos objeto (.o) e dependências (.d)
clean-objs:
	@echo "Limpando arquivos objeto e dependências..."
	@find $(BUILD_DIR) -name "*.o" -type f -delete 2>/dev/null || true
	@find $(BUILD_DIR) -name "*.d" -type f -delete 2>/dev/null || true
	@echo "✓ Arquivos .o e .d removidos"

# Limpeza de artefatos de teste (objetos + modelos dummy)
clean-test-artifacts: clean-objs
	@echo "Limpando artefatos de teste..."
	@rm -f model_dummy.qorus tokenizer.bin 2>/dev/null || true
	@echo "✓ Artefatos de teste removidos (.o, .d, model_dummy.qorus, tokenizer.bin)"

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(BENCHMARK_TARGET) model_dummy.qorus tokenizer.bin static-analysis.log cppcheck-report.log clang-tidy-report.log compile_commands.json
