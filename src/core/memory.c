#include "qorus.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <limits.h>

// === PLATFORM ABSTRACTION LAYER ===
#ifdef _WIN32
    #include <malloc.h>
    #define q_aligned_alloc(align, size) _aligned_malloc(size, align)
    #define q_aligned_free(ptr) _aligned_free(ptr)
#else
    #define q_aligned_alloc(align, size) aligned_alloc(align, size)
    #define q_aligned_free(ptr) free(ptr)
#endif

// Definições de compatibilidade multiplataforma
#ifndef MAP_POPULATE
#define MAP_POPULATE 0
#endif

#ifdef __APPLE__
#include <sys/mman.h>
#define madvise(addr, len, advice) posix_madvise(addr, len, advice)
#ifndef MADV_SEQUENTIAL
#define MADV_SEQUENTIAL 0
#endif
#ifndef MADV_WILLNEED
#define MADV_WILLNEED 0
#endif
#endif

#ifdef DEBUG
#define Q_ARENA_POISON_PATTERN 0xDEADBEEF
#define Q_ARENA_POISON_SIZE (64 * 1024)
#endif

// Validação de alinhamento de ponteiro (zero overhead em release)
static inline bool q_is_aligned(const void* ptr) {
    return ((uintptr_t)ptr % Q_ALIGN) == 0;
}

// Helper: Safe alignment calculation with overflow check
// Returns 0 on overflow, aligned size otherwise
// Time Complexity: O(1)
static inline size_t safe_align_size(size_t size) {
    // Check: Can we add (Q_ALIGN - 1) without overflow?
    if (__builtin_expect(size > SIZE_MAX - (Q_ALIGN - 1), 0)) {
        return 0;  // Overflow
    }
    return ((size + Q_ALIGN - 1) & ~(Q_ALIGN - 1));
}

// Inicializar memória com estratégia configurável (Tier 1: Mmap)
q_error_code q_init_memory_ex(q_context* restrict ctx, const char* model_path, q_mmap_strategy strategy) {
    Q_VALIDATE_PTR_OR_RETURN(ctx, Q_ERR_INVALID_ARG);
    Q_VALIDATE_PTR_OR_RETURN(model_path, Q_ERR_INVALID_ARG);
    
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        return Q_ERR_FILE_OPEN;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return Q_ERR_FILE_STAT;
    }
    size_t file_size = st.st_size;

    if (file_size < Q_HEADER_SIZE) {
        close(fd);
        return Q_ERR_FILE_TOO_SMALL;
    }

    // Mmap com flags seguras para portabilidade
    int flags = MAP_PRIVATE;
    
    // CRITICAL FIX: Tornar estratégia configurável
    // Q_MMAP_EAGER: Pré-carregar páginas (lento startup, rápida primeira inferência)
    // Q_MMAP_LAZY: Carregar sob demanda (rápido startup, page faults na primeira inferência)
    if (strategy == Q_MMAP_EAGER) {
        #ifdef __linux__
        flags |= MAP_POPULATE;  // Apenas Linux pré-carrega páginas
        #endif
    }
    // else: Q_MMAP_LAZY (padrão) - não usar MAP_POPULATE

    void* mmap_ptr = mmap(NULL, file_size, PROT_READ, flags, fd, 0);
    close(fd);

    if (mmap_ptr == MAP_FAILED) {
        return Q_ERR_MMAP_FAILED;
    }

    // Hints de performance (com fallback silencioso)
    // Sempre usar madvise para hints assíncronos (não bloqueia)
    #if defined(__linux__) || defined(__FreeBSD__)
    // Linux e FreeBSD suportam madvise completo
    madvise(mmap_ptr, file_size, MADV_SEQUENTIAL | MADV_WILLNEED);
    #elif defined(__APPLE__)
    // macOS: usar posix_madvise (já mapeado acima)
    posix_madvise(mmap_ptr, file_size, POSIX_MADV_SEQUENTIAL);
    posix_madvise(mmap_ptr, file_size, POSIX_MADV_WILLNEED);
    #else
    // Outros sistemas: ignorar silenciosamente
    (void)mmap_ptr; (void)file_size;
    #endif

    q_model_header* header = (q_model_header*)mmap_ptr;
    if (header->magic != Q_MAGIC) {
        munmap(mmap_ptr, file_size);
        return Q_ERR_INVALID_MAGIC;
    }

    Q_ASSERT_ALIGNED(header);

    ctx->weights_mmap = mmap_ptr;
    ctx->weights_size = file_size;
    ctx->header = header;

    return Q_OK;
}

// Wrapper para compatibilidade (padrão: LAZY para melhor UX)
q_error_code q_init_memory(q_context* restrict ctx, const char* model_path) {
    return q_init_memory_ex(ctx, model_path, Q_MMAP_LAZY);
}

// Alocar KV Cache (Tier 2: Persistent)
q_error_code q_alloc_kv_cache(q_context* restrict ctx, size_t kv_size) {
    Q_VALIDATE_PTR_OR_RETURN(ctx, Q_ERR_INVALID_ARG);
    
    // Prevent Memory Leak: Check if already allocated
    if (ctx->kv_buffer != NULL) {
        return Q_ERR_INVALID_ARG;
    }

    // Security: Safe alignment with overflow check
    size_t aligned_size = safe_align_size(kv_size);
    if (aligned_size == 0) {
        return Q_ERR_OVERFLOW;
    }
    
    // Use platform abstraction wrapper
    void* kv_buf = q_aligned_alloc(Q_ALIGN, aligned_size);
    if (!kv_buf) {
        return Q_ERR_ALLOC_FAILED;
    }

    // Zero-initialize (Security best practice)
    memset(kv_buf, 0, aligned_size);

    ctx->kv_buffer = kv_buf;
    ctx->kv_size = aligned_size;

    Q_ASSERT_ALIGNED(kv_buf);
    return Q_OK;
}

// Alocar Arena (Tier 3: Transient)
// CORREÇÃO 5: Inicializa scratch_base_offset para 0 (será atualizado após llama_build_graph)
q_error_code q_alloc_arena(q_context* restrict ctx, size_t arena_size) {
    Q_VALIDATE_PTR_OR_RETURN(ctx, Q_ERR_INVALID_ARG);
    
    // Prevent Memory Leak
    if (ctx->scratch_buffer != NULL) {
        return Q_ERR_INVALID_ARG;
    }
    
    // Security: Safe alignment with overflow check
    size_t aligned_size = safe_align_size(arena_size);
    if (aligned_size == 0) {
        return Q_ERR_OVERFLOW;
    }
    
    // Use platform abstraction wrapper
    void* arena_buf = q_aligned_alloc(Q_ALIGN, aligned_size);
    if (!arena_buf) {
        return Q_ERR_ALLOC_FAILED;
    }

    ctx->scratch_buffer = arena_buf;
    ctx->scratch_size = aligned_size;
    ctx->scratch_head = 0;  // Inicialmente alinhado
    ctx->scratch_base_offset = 0;  // CORREÇÃO 5: Inicializar (será atualizado após llama_build_graph)

    Q_ASSERT_ALIGNED(arena_buf);
    return Q_OK;
}

// Arena alloc (otimizado: branchless, zero overhead no caminho feliz)
// CRITICAL: Must call q_alloc_arena() before using this function
void* q_arena_alloc(q_context* restrict ctx, size_t size) {
    // Security: Validate context pointer (always active)
    if (__builtin_expect(ctx == NULL, 0)) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: q_arena_alloc: ctx is NULL at %s:%d\n", __FILE__, __LINE__);
        abort();
        #else
        return NULL;
        #endif
    }
    
    // Security: Validate arena is initialized (always active)
    if (__builtin_expect(ctx->scratch_buffer == NULL, 0)) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: q_arena_alloc: arena not initialized (call q_alloc_arena first)\n");
        abort();
        #else
        return NULL;
        #endif
    }
    
    // Security: Safe alignment with overflow check
    size_t aligned_size = safe_align_size(size);
    if (aligned_size == 0) {
        return NULL;  // Overflow in alignment
    }

    // Security: Check for overflow in addition (head + size)
    if (__builtin_expect(ctx->scratch_head > SIZE_MAX - aligned_size, 0)) {
        return NULL;  // Overflow in addition
    }

    size_t new_head = ctx->scratch_head + aligned_size;
    
    // Bounds check
    if (__builtin_expect(new_head > ctx->scratch_size, 0)) {
        return NULL;  // OOM
    }

    // OTIMIZAÇÃO CRÍTICA: Usar __builtin_assume_aligned baseado em invariante matemática
    // Invariante garantida: scratch_head é sempre múltiplo de Q_ALIGN
    // Prova: Base (scratch_head = 0) e Indução (scratch_head += Q_ALIGN_SIZE(size))
    // Isso elimina validação de alinhamento em runtime e permite otimizações do compilador
    // O compilador pode gerar instruções VMOVAPS (aligned store) sem preâmbulo de verificação
    void* base_ptr = __builtin_assume_aligned(ctx->scratch_buffer, Q_ALIGN);
    void* ptr = (uint8_t*)base_ptr + ctx->scratch_head;
    
    ctx->scratch_head = new_head; // Invariante mantida (new_head é múltiplo de Q_ALIGN)
    
    // Validação apenas em DEBUG para detectar bugs que quebrem a invariante
    #ifdef DEBUG
    if (new_head % Q_ALIGN != 0) {
        fprintf(stderr, "ERROR: q_arena_alloc: Invariante violada! new_head (%zu) not aligned to %d bytes\n", 
                new_head, Q_ALIGN);
        abort();
    }
    Q_ASSERT_ALIGNED(ptr);
    #endif

    return ptr;
}

// Reset arena (com poisoning seguro e otimizado)
// CORREÇÃO CRÍTICA: Validação de underflow antes da subtração
// CORREÇÃO 5: Resetar apenas para scratch_base_offset, não para 0
// Isso protege estruturas do modelo alocadas antes do scratchpad
void q_arena_reset(q_context* restrict ctx) {
    // Security: Validate context pointer
    if (__builtin_expect(ctx == NULL, 0)) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: q_arena_reset: ctx is NULL at %s:%d\n", __FILE__, __LINE__);
        abort();
        #endif
        return;
    }
    
    #ifdef DEBUG
    // Early return if arena not initialized
    if (ctx->scratch_buffer == NULL) {
        ctx->scratch_head = ctx->scratch_base_offset;
        return;
    }
    
    // SECURITY FIX: Prevent Integer Underflow
    // Validate invariant before subtraction to prevent silent corruption
    if (__builtin_expect(ctx->scratch_head < ctx->scratch_base_offset, 0)) {
        fprintf(stderr, "CRITICAL: Memory corruption in Arena! head(%zu) < base(%zu)\n",
                ctx->scratch_head, ctx->scratch_base_offset);
        abort(); // Fail fast to prevent exploiting the corrupted state
    }
    
    // Calculate poison_size safely (apenas a parte do scratchpad, não o modelo)
    size_t scratch_used = ctx->scratch_head - ctx->scratch_base_offset;
    size_t poison_size = scratch_used;
    if (poison_size > Q_ARENA_POISON_SIZE) {
        poison_size = Q_ARENA_POISON_SIZE;
    }
    if (poison_size + ctx->scratch_base_offset > ctx->scratch_size) {
        poison_size = (ctx->scratch_size > ctx->scratch_base_offset) ? 
                      (ctx->scratch_size - ctx->scratch_base_offset) : 0;
    }

    // Use memset for safety and performance (apenas na região do scratchpad)
    if (poison_size > 0) {
        uint8_t* scratch_start = (uint8_t*)ctx->scratch_buffer + ctx->scratch_base_offset;
        memset(scratch_start, 0xDE, poison_size);
    }
    
    if (scratch_used > Q_ARENA_POISON_SIZE) {
        fprintf(stderr, "WARNING: Arena reset com %zu bytes de scratchpad usados (possível vazamento?)\n", 
                scratch_used);
    }
    #endif
    
    // CORREÇÃO 5: Resetar para scratch_base_offset, não para 0
    ctx->scratch_head = ctx->scratch_base_offset;
}

void q_free_memory(q_context* restrict ctx) {
    // Security: Validate context pointer
    if (__builtin_expect(ctx == NULL, 0)) {
        #ifdef DEBUG
        fprintf(stderr, "ERROR: q_free_memory: ctx is NULL at %s:%d\n", __FILE__, __LINE__);
        abort();
        #endif
        return;
    }
    
    // Free in LIFO order (Last In, First Out)
    // Typical allocation order: q_init_memory() → q_alloc_kv_cache() → q_alloc_arena()
    // Therefore free order: arena → kv_cache → mmap
    
    // 1. Free arena (allocated last)
    if (ctx->scratch_buffer) {
        q_aligned_free(ctx->scratch_buffer); // Use platform abstraction wrapper
        ctx->scratch_buffer = NULL;
        ctx->scratch_size = 0;
        ctx->scratch_head = 0;
        ctx->scratch_base_offset = 0;  // CORREÇÃO 5: Resetar também
    }
    
    // 2. Free KV cache (allocated second)
    if (ctx->kv_buffer) {
        q_aligned_free(ctx->kv_buffer); // Use platform abstraction wrapper
        ctx->kv_buffer = NULL;
        ctx->kv_size = 0;
    }
    
    // 3. Free mmap (allocated first)
    if (ctx->weights_mmap) {
        munmap(ctx->weights_mmap, ctx->weights_size);
        ctx->weights_mmap = NULL;
        ctx->weights_size = 0;
        // Security: Clear header pointer (it points into the unmapped memory)
        ctx->header = NULL;
    }
}
