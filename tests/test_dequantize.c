#include "qorus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>

// Helper function to load binary file
static size_t load_file(const char* path, void** data) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    *data = malloc(size);
    if (!*data) {
        fclose(f);
        return 0;
    }
    
    size_t read = fread(*data, 1, size, f);
    fclose(f);
    
    return read == size ? size : 0;
}

// Helper function to load binary array (simple format: uint32_t count + float32[] data)
static int load_binary_array(const char* path, float** data, size_t* count) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    
    // Read shape (assuming 1D array, 4 bytes for count)
    uint32_t shape;
    if (fread(&shape, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return -1;
    }
    
    *count = shape;
    *data = malloc(*count * sizeof(float));
    if (!*data) {
        fclose(f);
        return -1;
    }
    
    if (fread(*data, sizeof(float), *count, f) != *count) {
        free(*data);
        fclose(f);
        return -1;
    }
    
    fclose(f);
    return 0;
}

// Hybrid error check (absolute + relative)
static bool check_error(float actual, float expected, float abs_tol, float rel_tol) {
    float abs_diff = fabsf(actual - expected);
    float rel_diff = abs_diff / (fabsf(expected) + 1e-8f);  // Avoid division by zero
    
    return (abs_diff <= abs_tol) || (rel_diff <= rel_tol);
}

static int test_dequantize_case(const char* block_path, const char* expected_path, int case_num) {
    printf("\n=== Test Case %d ===\n", case_num);
    
    // Load Q4_0 block
    void* block_data = NULL;
    size_t block_size = load_file(block_path, &block_data);
    if (block_size != 20) {
        printf("ERROR: Failed to load block file or wrong size (%zu bytes)\n", block_size);
        return -1;
    }
    
    // Load expected output
    float* expected = NULL;
    size_t expected_count = 0;
    if (load_binary_array(expected_path, &expected, &expected_count) != 0) {
        printf("ERROR: Failed to load expected output\n");
        free(block_data);
        return -1;
    }
    
    if (expected_count != 32) {
        printf("ERROR: Expected output must have 32 values, got %zu\n", expected_count);
        free(block_data);
        free(expected);
        return -1;
    }
    
    // Allocate aligned output buffer
    float* output = aligned_alloc(32, 32 * sizeof(float));
    if (!output) {
        printf("ERROR: Failed to allocate aligned output buffer\n");
        free(block_data);
        free(expected);
        return -1;
    }
    
    // Verify alignment
    if (((uintptr_t)output % 32) != 0) {
        printf("ERROR: Output buffer not aligned to 32 bytes\n");
        free(block_data);
        free(expected);
        free(output);
        return -1;
    }
    
    // Execute dequantization
    q_dequantize_q4_0_block_avx2_public((const q_block_q4_0*)block_data, output);
    
    // Validate results
    int errors = 0;
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    
    for (size_t i = 0; i < 32; i++) {
        float abs_diff = fabsf(output[i] - expected[i]);
        float rel_diff = abs_diff / (fabsf(expected[i]) + 1e-8f);
        
        if (abs_diff > max_abs_error) max_abs_error = abs_diff;
        if (rel_diff > max_rel_error) max_rel_error = rel_diff;
        
        if (!check_error(output[i], expected[i], Q_EPSILON_ABS_Q4_VAL, Q_EPSILON_REL_Q4_VAL)) {
            if (errors < 5) {  // Print first 5 errors
                printf("  Error at index %zu: got %.6f, expected %.6f "
                       "(abs_diff=%.6f, rel_diff=%.6f)\n",
                       i, output[i], expected[i], abs_diff, rel_diff);
            }
            errors++;
        }
    }
    
    printf("  Max absolute error: %.6f (tolerance: %.6f)\n", 
           max_abs_error, Q_EPSILON_ABS_Q4_VAL);
    printf("  Max relative error: %.6f (tolerance: %.6f)\n", 
           max_rel_error, Q_EPSILON_REL_Q4_VAL);
    printf("  Errors: %d / 32\n", errors);
    
    if (errors == 0) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED\n");
    }
    
    free(block_data);
    free(expected);
    free(output);
    
    return errors == 0 ? 0 : -1;
}

int main(void) {
    printf("=== Q4_0 Dequantization Test Suite ===\n");
    printf("Validating against Python Gold Standard\n");
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Test Case 1: Uniform values
    total_tests++;
    if (test_dequantize_case("test_data/dequantize_test1_block.bin",
                             "test_data/dequantize_test1_expected.bin", 1) == 0) {
        passed_tests++;
    }
    
    // Test Case 2: Random values
    total_tests++;
    if (test_dequantize_case("test_data/dequantize_test2_block.bin",
                             "test_data/dequantize_test2_expected.bin", 2) == 0) {
        passed_tests++;
    }
    
    // Test Case 3: Zero scale
    total_tests++;
    if (test_dequantize_case("test_data/dequantize_test3_block.bin",
                             "test_data/dequantize_test3_expected.bin", 3) == 0) {
        passed_tests++;
    }
    
    // Test Case 4: Large scale
    total_tests++;
    if (test_dequantize_case("test_data/dequantize_test4_block.bin",
                             "test_data/dequantize_test4_expected.bin", 4) == 0) {
        passed_tests++;
    }
    
    printf("\n=== Test Summary ===\n");
    printf("Passed: %d / %d\n", passed_tests, total_tests);
    
    if (passed_tests == total_tests) {
        printf("✓ All tests PASSED\n");
        return 0;
    } else {
        printf("✗ Some tests FAILED\n");
        return 1;
    }
}

