// ============================================================================
// EXAMPLE: Text Generation (FASE 4.2)
// ============================================================================
// Exemplo de uso do loop de geração completo
// ============================================================================

#include "qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.qorus> <tokenizer.bin> [prompt]\n", argv[0]);
        fprintf(stderr, "Example: %s model.qorus tokenizer.bin \"Hello, how are you?\"\n", argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* tokenizer_path = argv[2];
    const char* prompt = (argc > 3) ? argv[3] : "Hello";
    
    printf("Qorus-IA: Text Generation Example (FASE 4.2)\n");
    printf("==============================================\n\n");
    
    // Initialize context
    q_context ctx = {0};
    q_error_code err = q_init_memory(&ctx, model_path);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: Failed to load model: %s\n", q_strerror(err));
        return 1;
    }
    printf("✓ Model loaded\n");
    
    // Build model graph
    q_llama_model model = {0};
    err = llama_build_graph(&ctx, &model);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: Failed to build model graph: %s\n", q_strerror(err));
        q_free_memory(&ctx);
        return 1;
    }
    printf("✓ Model graph built\n");
    
    // Allocate KV cache
    // Tamanho: n_layers * n_kv_heads * max_seq_len * head_dim * 2 (K e V) * sizeof(float)
    uint32_t max_seq_len = model.config.max_seq_len;
    uint32_t n_layers = model.config.n_layers;
    uint32_t n_kv_heads = model.config.n_kv_heads;
    uint32_t head_dim = model.config.dim / model.config.n_heads;
    size_t kv_size = (size_t)n_layers * (size_t)n_kv_heads * (size_t)max_seq_len * 
                     (size_t)head_dim * 2 * sizeof(float);
    kv_size = Q_ALIGN_SIZE(kv_size);
    
    err = q_alloc_kv_cache(&ctx, kv_size);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: Failed to allocate KV cache: %s\n", q_strerror(err));
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    printf("✓ KV cache allocated\n");
    
    // Allocate arena
    size_t arena_size = 64 * 1024 * 1024; // 64 MB (ajustar conforme necessário)
    err = q_alloc_arena(&ctx, arena_size);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: Failed to allocate arena: %s\n", q_strerror(err));
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    printf("✓ Arena allocated\n");
    
    // Load tokenizer
    q_tokenizer tokenizer = {0};
    err = q_tokenizer_load(&tokenizer, tokenizer_path);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: Failed to load tokenizer: %s\n", q_strerror(err));
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    printf("✓ Tokenizer loaded\n\n");
    
    // Encode prompt
    uint32_t prompt_tokens[256];
    uint32_t num_prompt_tokens = 0;
    err = q_tokenizer_encode(&tokenizer, prompt, prompt_tokens, &num_prompt_tokens, 256, true, false);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: Failed to encode prompt: %s\n", q_strerror(err));
        q_tokenizer_free(&tokenizer);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    printf("Prompt: \"%s\"\n", prompt);
    printf("Prompt tokens (%u): ", num_prompt_tokens);
    for (uint32_t i = 0; i < num_prompt_tokens; i++) {
        printf("%u ", prompt_tokens[i]);
    }
    printf("\n\n");
    
    // Setup generation state
    uint32_t generated_tokens[256];
    q_generation_state gen_state = {
        .ctx = &ctx,
        .model = &model,
        .tokenizer = &tokenizer,
        .prompt_tokens = prompt_tokens,
        .num_prompt_tokens = num_prompt_tokens,
        .generated_tokens = generated_tokens,
        .num_generated_tokens = 0,
        .max_tokens = 50,  // Gerar até 50 tokens
        .temperature = 0.8f,
        .top_k = 40,
        .top_p = 0.9f,
        .current_pos = 0
    };
    
    // Generate text
    printf("Generating text...\n");
    err = q_generate(&gen_state);
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: Generation failed: %s\n", q_strerror(err));
        q_tokenizer_free(&tokenizer);
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    
    printf("✓ Generated %u tokens\n", gen_state.num_generated_tokens);
    printf("Generated tokens: ");
    for (uint32_t i = 0; i < gen_state.num_generated_tokens; i++) {
        printf("%u ", gen_state.generated_tokens[i]);
    }
    printf("\n\n");
    
    // Decode generated tokens
    char generated_text[2048];
    err = q_tokenizer_decode(&tokenizer, gen_state.generated_tokens, 
                             gen_state.num_generated_tokens, generated_text, sizeof(generated_text));
    if (err != Q_OK) {
        fprintf(stderr, "ERROR: Failed to decode tokens: %s\n", q_strerror(err));
    } else {
        printf("Generated text: \"%s\"\n\n", generated_text);
    }
    
    // Cleanup
    q_tokenizer_free(&tokenizer);
    llama_free_graph(&model);
    q_free_memory(&ctx);
    
    printf("✓ Generation complete!\n");
    return 0;
}

