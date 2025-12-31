#include "../include/qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

// Teste de validação da FASE 1
// Valida: magic number, alinhamento de ponteiros, arena alloc

int main(void) {
    printf("=== Teste de Validação FASE 1 ===\n\n");
    
    q_context ctx = {0};
    int ret;
    
    // Teste 1: Gerar modelo dummy (deve ser feito via Python primeiro)
    printf("1. Carregando modelo dummy...\n");
    ret = q_init_memory(&ctx, "model_dummy.qorus");
    if (ret != Q_OK) {
        printf("   ERRO: Falha ao carregar modelo (código: %d)\n", ret);
        printf("   Execute primeiro: python3 tools/convert_llama.py model_dummy.qorus\n");
        return 1;
    }
    printf("   ✓ Modelo carregado com sucesso\n");
    
    // Teste 2: Verificar magic number
    printf("\n2. Verificando magic number...\n");
    if (ctx.header->magic == Q_MAGIC) {
        printf("   ✓ Magic number correto: 0x%08X\n", ctx.header->magic);
    } else {
        printf("   ERRO: Magic number incorreto: 0x%08X (esperado: 0x%08X)\n",
               ctx.header->magic, Q_MAGIC);
        q_free_memory(&ctx);
        return 1;
    }
    
    // Teste 3: Verificar alinhamento do header
    printf("\n3. Verificando alinhamento do header...\n");
    uintptr_t header_addr = (uintptr_t)ctx.header;
    if ((header_addr % Q_ALIGN) == 0) {
        printf("   ✓ Header alinhado a %d bytes: %p\n", Q_ALIGN, (void*)ctx.header);
    } else {
        printf("   ERRO: Header desalinhado: %p (offset: %zu)\n",
               (void*)ctx.header, header_addr % Q_ALIGN);
        q_free_memory(&ctx);
        return 1;
    }
    
    // Teste 4: Alocar arena
    printf("\n4. Alocando arena (512MB)...\n");
    ret = q_alloc_arena(&ctx, 512 * 1024 * 1024);
    if (ret != Q_OK) {
        printf("   ERRO: Falha ao alocar arena (código: %d)\n", ret);
        q_free_memory(&ctx);
        return 1;
    }
    printf("   ✓ Arena alocada: %zu bytes\n", ctx.scratch_size);
    
    // Teste 5: Verificar alinhamento da arena
    printf("\n5. Verificando alinhamento da arena...\n");
    uintptr_t arena_addr = (uintptr_t)ctx.scratch_buffer;
    if ((arena_addr % Q_ALIGN) == 0) {
        printf("   ✓ Arena alinhada a %d bytes: %p\n", Q_ALIGN, ctx.scratch_buffer);
    } else {
        printf("   ERRO: Arena desalinhada: %p (offset: %zu)\n",
               ctx.scratch_buffer, arena_addr % Q_ALIGN);
        q_free_memory(&ctx);
        return 1;
    }
    
    // Teste 6: Testar múltiplas alocações na arena
    printf("\n6. Testando alocações na arena...\n");
    void* ptrs[10];
    size_t sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    
    for (int i = 0; i < 10; i++) {
        ptrs[i] = q_arena_alloc(&ctx, sizes[i]);
        if (!ptrs[i]) {
            printf("   ERRO: Falha ao alocar %zu bytes\n", sizes[i]);
            q_free_memory(&ctx);
            return 1;
        }
        
        uintptr_t ptr_addr = (uintptr_t)ptrs[i];
        if ((ptr_addr % Q_ALIGN) != 0) {
            printf("   ERRO: Ponteiro desalinhado: %p (offset: %zu)\n",
                   ptrs[i], ptr_addr % Q_ALIGN);
            q_free_memory(&ctx);
            return 1;
        }
        
        printf("   ✓ Alocado %zu bytes em %p (alinhado)\n", sizes[i], ptrs[i]);
    }
    
    // Teste 7: Reset da arena
    printf("\n7. Testando reset da arena...\n");
    size_t head_before = ctx.scratch_head;
    q_arena_reset(&ctx);
    if (ctx.scratch_head == 0) {
        printf("   ✓ Arena resetada: head=%zu -> %zu\n", head_before, ctx.scratch_head);
    } else {
        printf("   ERRO: Reset falhou: head=%zu (esperado: 0)\n", ctx.scratch_head);
        q_free_memory(&ctx);
        return 1;
    }
    
    // Teste 8: Alocação após reset
    printf("\n8. Testando alocação após reset...\n");
    void* ptr_after_reset = q_arena_alloc(&ctx, 1024);
    if (ptr_after_reset && ((uintptr_t)ptr_after_reset % Q_ALIGN) == 0) {
        printf("   ✓ Alocação após reset funcionou: %p\n", ptr_after_reset);
    } else {
        printf("   ERRO: Alocação após reset falhou ou desalinhada\n");
        q_free_memory(&ctx);
        return 1;
    }
    
    // Teste 9: Informações do modelo
    printf("\n9. Informações do modelo:\n");
    printf("   - Version: %u\n", ctx.header->version);
    printf("   - Vocab Size: %u\n", ctx.header->vocab_size);
    printf("   - Dim: %u\n", ctx.header->dim);
    printf("   - Hidden Dim: %u\n", ctx.header->hidden_dim);
    printf("   - Layers: %u\n", ctx.header->n_layers);
    printf("   - Heads: %u\n", ctx.header->n_heads);
    printf("   - KV Heads: %u\n", ctx.header->n_kv_heads);
    printf("   - Max Seq Len: %u\n", ctx.header->max_seq_len);
    printf("   - Rope Freq Base: %f\n", ctx.header->rope_freq_base);
    
    // Cleanup
    printf("\n10. Limpando memória...\n");
    q_free_memory(&ctx);
    printf("   ✓ Memória liberada\n");
    
    printf("\n=== Todos os testes passaram! ===\n");
    return 0;
}

