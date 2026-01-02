#include "qorus.h"

// Convert error code to human-readable string
// Returns: Pointer to static string (do not free)
// 
// SECURITY FIX: Switch-case instead of array lookup
// - Compiler warns if enum case is missing (-Wswitch)
// - No manual index alignment required
// - Compiler generates jump table (O(1)) for dense switches
// - Eliminates fragility of manual array bounds checking
const char* q_strerror(q_error_code err) {
    switch (err) {
        case Q_OK: return "Success";
        case Q_ERR_NULL_PTR: return "Null pointer argument";
        case Q_ERR_FILE_OPEN: return "Failed to open file";
        case Q_ERR_FILE_STAT: return "Failed to stat file";
        case Q_ERR_FILE_TOO_SMALL: return "File too small (corrupt header?)";
        case Q_ERR_MMAP_FAILED: return "mmap() failed";
        case Q_ERR_INVALID_MAGIC: return "Invalid file magic (not a Qorus file)";
        case Q_ERR_ALLOC_FAILED: return "Memory allocation failed";
        case Q_ERR_ARENA_OOM: return "Arena Out of Memory";
        case Q_ERR_INVALID_CONFIG: return "Invalid model configuration";
        case Q_ERR_INVALID_ARG: return "Invalid argument";
        case Q_ERR_ALIASING: return "Input/output aliasing detected";
        case Q_ERR_OVERFLOW: return "Integer overflow detected";
        case Q_ERR_MISALIGNED: return "Pointer not properly aligned";
        case Q_ERR_INVALID_DTYPE: return "Invalid data type";
        case Q_ERR_INVALID_SIZE: return "Invalid size";
        default: return "Unknown error";
    }
}
