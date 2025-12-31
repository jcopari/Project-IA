#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

// TDD Test for FASE 3.2: llama_build_graph()
// Tests model graph construction from mmap'd .qorus file

int main(void) {
    printf("=== Test: llama_build_graph() ===\n\n");
    
    q_context ctx = {0};
    llama_model model = {0};
    q_error_code ret;
    
    // Test 1: Load model file
    printf("1. Loading model file...\n");
    ret = q_init_memory(&ctx, "model_dummy.qorus");
    if (ret != Q_OK) {
        printf("   ERROR: Failed to load model (code: %d, %s)\n", ret, q_strerror(ret));
        printf("   Execute first: python3 tools/convert_llama.py model_dummy.qorus 2\n");
        return 1;
    }
    printf("   ✓ Model loaded successfully\n");
    printf("   Header: magic=0x%08X, n_layers=%u, dim=%u\n",
           ctx.header->magic, ctx.header->n_layers, ctx.header->dim);
    
    // Test 2: Allocate arena for model structures
    printf("\n2. Allocating arena...\n");
    ret = q_alloc_arena(&ctx, 1024 * 1024);  // 1MB arena
    if (ret != Q_OK) {
        printf("   ERROR: Failed to allocate arena (code: %d, %s)\n", ret, q_strerror(ret));
        q_free_memory(&ctx);
        return 1;
    }
    printf("   ✓ Arena allocated\n");
    
    // Test 3: Build model graph
    printf("\n3. Building model graph...\n");
    ret = llama_build_graph(&ctx, &model);
    if (ret != Q_OK) {
        printf("   ERROR: Failed to build graph (code: %d, %s)\n", ret, q_strerror(ret));
        q_free_memory(&ctx);
        return 1;
    }
    printf("   ✓ Model graph built successfully\n");
    
    // Test 4: Validate configuration
    printf("\n4. Validating configuration...\n");
    if (model.config.vocab_size == ctx.header->vocab_size &&
        model.config.dim == ctx.header->dim &&
        model.config.hidden_dim == ctx.header->hidden_dim &&
        model.config.n_layers == ctx.header->n_layers &&
        model.config.n_heads == ctx.header->n_heads &&
        model.config.n_kv_heads == ctx.header->n_kv_heads) {
        printf("   ✓ Configuration matches header\n");
    } else {
        printf("   ERROR: Configuration mismatch\n");
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    
    // Test 5: Validate token embeddings
    printf("\n5. Validating token embeddings...\n");
    if (model.token_embd != NULL) {
        if (model.token_embd->ne[0] == ctx.header->vocab_size &&
            model.token_embd->ne[1] == ctx.header->dim) {
            printf("   ✓ token_embd: [%u, %u]\n",
                   model.token_embd->ne[0], model.token_embd->ne[1]);
        } else {
            printf("   ERROR: token_embd shape mismatch\n");
            llama_free_graph(&model);
            q_free_memory(&ctx);
            return 1;
        }
    } else {
        printf("   ERROR: token_embd is NULL\n");
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    
    // Test 6: Validate output layer
    printf("\n6. Validating output layer...\n");
    if (model.output_norm != NULL && model.output != NULL) {
        if (model.output_norm->ne[0] == ctx.header->dim &&
            model.output->ne[0] == ctx.header->vocab_size &&
            model.output->ne[1] == ctx.header->dim) {
            printf("   ✓ output_norm: [%u]\n", model.output_norm->ne[0]);
            printf("   ✓ output: [%u, %u]\n",
                   model.output->ne[0], model.output->ne[1]);
        } else {
            printf("   ERROR: Output layer shape mismatch\n");
            llama_free_graph(&model);
            q_free_memory(&ctx);
            return 1;
        }
    } else {
        printf("   ERROR: output_norm or output is NULL\n");
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    
    // Test 7: Validate layers array
    printf("\n7. Validating layers array...\n");
    if (model.layers == NULL) {
        printf("   ERROR: layers array is NULL\n");
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    printf("   ✓ Layers array allocated: %u layers\n", model.config.n_layers);
    
    // Test 8: Validate each layer structure
    printf("\n8. Validating layer structures...\n");
    for (uint32_t i = 0; i < model.config.n_layers; i++) {
        llama_layer* layer = &model.layers[i];
        
        if (layer->layer_idx != i) {
            printf("   ERROR: Layer %u has incorrect layer_idx (%u)\n", i, layer->layer_idx);
            llama_free_graph(&model);
            q_free_memory(&ctx);
            return 1;
        }
        
        // Check all tensor pointers are non-NULL
        if (layer->attn_norm == NULL || layer->wq == NULL || layer->wk == NULL ||
            layer->wv == NULL || layer->wo == NULL || layer->ffn_norm == NULL ||
            layer->w_gate == NULL || layer->w_up == NULL || layer->w_down == NULL) {
            printf("   ERROR: Layer %u has NULL tensor pointers\n", i);
            llama_free_graph(&model);
            q_free_memory(&ctx);
            return 1;
        }
        
        // Validate attention norm shape
        if (layer->attn_norm->ne[0] != model.config.dim) {
            printf("   ERROR: Layer %u attn_norm shape mismatch\n", i);
            llama_free_graph(&model);
            q_free_memory(&ctx);
            return 1;
        }
        
        // Validate Q projection shape
        if (layer->wq->ne[0] != model.config.dim || layer->wq->ne[1] != model.config.dim) {
            printf("   ERROR: Layer %u wq shape mismatch\n", i);
            llama_free_graph(&model);
            q_free_memory(&ctx);
            return 1;
        }
        
        printf("   ✓ Layer %u: all tensors configured\n", i);
    }
    
    // Test 9: Validate data pointers point to mmap
    printf("\n9. Validating data pointers point to mmap...\n");
    uintptr_t mmap_start = (uintptr_t)ctx.weights_mmap;
    uintptr_t mmap_end = mmap_start + ctx.weights_size;
    
    if ((uintptr_t)model.token_embd->data < mmap_start ||
        (uintptr_t)model.token_embd->data >= mmap_end) {
        printf("   ERROR: token_embd->data outside mmap range\n");
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    printf("   ✓ token_embd->data points to mmap\n");
    
    // Test 10: Validate context pointer
    printf("\n10. Validating context pointer...\n");
    if (model.ctx == &ctx) {
        printf("   ✓ Context pointer set correctly\n");
    } else {
        printf("   ERROR: Context pointer mismatch\n");
        llama_free_graph(&model);
        q_free_memory(&ctx);
        return 1;
    }
    
    // Cleanup
    printf("\n11. Cleaning up...\n");
    llama_free_graph(&model);
    q_free_memory(&ctx);
    printf("   ✓ Cleanup successful\n");
    
    printf("\n=== All tests passed! ===\n");
    return 0;
}

