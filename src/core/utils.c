#include "qorus.h"

// Lookup table para O(1) error string conversion
// Mapeia códigos de erro negativos para índices positivos do array
// Time Complexity: O(1) - Array lookup
// Space Complexity: O(1) - Constant space
static const char* const Q_ERROR_STRINGS[] = {
    [0] = "Success",
    [1] = "Null pointer argument",
    [2] = "Failed to open file",
    [3] = "Failed to stat file",
    [4] = "File too small (corrupt header?)",
    [5] = "mmap() failed",
    [6] = "Invalid file magic (not a Qorus file)",
    [7] = "Memory allocation failed",
    [8] = "Arena Out of Memory",
    [9] = "Invalid model configuration",
    [10] = "Invalid argument",
    [11] = "Input/output aliasing detected",
    [12] = "Integer overflow detected",
    [13] = "Pointer not properly aligned",
    [14] = "Invalid data type",
    [15] = "Invalid size"
};

// Convert error code to human-readable string
// Returns: Pointer to static string (do not free)
const char* q_strerror(q_error_code err) {
    // Convert negative error code to positive index
    // Q_OK (0) -> 0, Q_ERR_NULL_PTR (-1) -> 1, etc.
    int idx = -err;
    
    // Bounds check
    if (idx < 0 || idx >= (int)(sizeof(Q_ERROR_STRINGS)/sizeof(char*))) {
        return "Unknown error";
    }
    
    return Q_ERROR_STRINGS[idx];
}
