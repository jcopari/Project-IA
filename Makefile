# Build System para Qorus-IA v2.0
# Flags: -O3 -mavx2 -mfma (AVX2 + FMA for optimal SIMD performance)
# Debug: make DEBUG=1 (enables AddressSanitizer + UndefinedBehaviorSanitizer)

CC = gcc

# Flags base (comuns)
CFLAGS_COMMON = -std=c11 -Wall -Wextra -I./include -D_GNU_SOURCE

# Modo Release (Padrão: Performance Máxima)
CFLAGS_RELEASE = -O3 -mavx2 -mfma -fno-omit-frame-pointer
LDFLAGS_RELEASE = -lm

# Modo Debug (Segurança Máxima + ASan + UBSan)
CFLAGS_DEBUG = -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer -DDEBUG
LDFLAGS_DEBUG = -lm -fsanitize=address,undefined

# Seletor baseado em variável de ambiente
ifeq ($(DEBUG),1)
    CFLAGS = $(CFLAGS_COMMON) $(CFLAGS_DEBUG)
    LDFLAGS = $(LDFLAGS_DEBUG)
else
    CFLAGS = $(CFLAGS_COMMON) $(CFLAGS_RELEASE)
    LDFLAGS = $(LDFLAGS_RELEASE)
endif

# Diretórios
SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = include
TESTS_DIR = tests

# Arquivos fonte
CORE_SRCS = $(wildcard $(SRC_DIR)/core/*.c)
OPS_AVX2_SRCS = $(filter-out %_ref.c, $(wildcard $(SRC_DIR)/ops/avx2/*.c))
OPS_CPU_SRCS = $(wildcard $(SRC_DIR)/ops/cpu/*.c)
MODELS_SRCS = $(wildcard $(SRC_DIR)/models/*.c)
TOKENIZER_SRCS = $(wildcard $(SRC_DIR)/tokenizer/*.c)

ALL_SRCS = $(CORE_SRCS) $(OPS_AVX2_SRCS) $(OPS_CPU_SRCS) $(MODELS_SRCS) $(TOKENIZER_SRCS)
OBJS = $(ALL_SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Target principal
TARGET = qorus-ia

# Benchmark tool
BENCHMARK_TARGET = tools/benchmark

# Testes
TEST_SRCS = $(wildcard $(TESTS_DIR)/*.c)
TEST_TARGETS = $(TEST_SRCS:$(TESTS_DIR)/%.c=$(BUILD_DIR)/tests/%)

.PHONY: all clean directories test test-memory test-dequantize test-matmul test-ops test-validation test-memory-adversarial test-llama3-overflow-adversarial test-utils test-avx-math test-llama-forward test-rmsnorm-adversarial test-rope-adversarial test-silu-adversarial test-softmax-adversarial test-dequantize-adversarial test-ops-integration benchmark

all: directories $(TARGET)

directories:
	@mkdir -p $(BUILD_DIR)/core
	@mkdir -p $(BUILD_DIR)/ops/avx2
	@mkdir -p $(BUILD_DIR)/ops/cpu
	@mkdir -p $(BUILD_DIR)/models
	@mkdir -p $(BUILD_DIR)/tokenizer
	@mkdir -p $(BUILD_DIR)/tests

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

# Benchmark tool
$(BENCHMARK_TARGET): tools/benchmark.c $(OBJS)
	$(CC) $(CFLAGS) -DDEBUG $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Regra para testes
$(BUILD_DIR)/tests/%: $(TESTS_DIR)/%.c $(OBJS)
	@mkdir -p $(BUILD_DIR)/tests
	$(CC) $(CFLAGS) -DDEBUG $< $(OBJS) -o $@ $(LDFLAGS)

# Target de teste específico
test-memory: directories $(BUILD_DIR)/tests/test_memory
	@echo "Gerando modelo dummy..."
	@python3 tools/convert_llama.py model_dummy.qorus || true
	@echo "Executando teste..."
	@$(BUILD_DIR)/tests/test_memory

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

test-llama3-overflow-adversarial: directories $(BUILD_DIR)/tests/test_llama3_adversarial_overflow
	@echo "Executando testes adversarial de proteção contra overflow..."
	@$(BUILD_DIR)/tests/test_llama3_adversarial_overflow

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

test-adversarial-all: test-rmsnorm-adversarial test-rope-adversarial test-silu-adversarial test-softmax-adversarial test-dequantize-adversarial test-matmul-adversarial
	@echo "✓ Todos os testes adversarial concluídos"

test-integration-all: test-ops-integration
	@echo "✓ Todos os testes de integração concluídos"

# Validação completa (Release + Debug)
test-validation: clean
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

test: test-memory test-dequantize test-matmul test-ops

benchmark: directories $(BENCHMARK_TARGET)
	@echo "Running performance benchmarks..."
	@$(BENCHMARK_TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(BENCHMARK_TARGET) model_dummy.qorus
