#!/usr/bin/env python3
"""
gen_forward_test_data.py - Generate Gold Standard test data for forward pass validation

Following MFR + CoT + Proof + TDD methodology:
- STEP 0: CoT - Problem Analysis
- STEP 0.5: Mathematical Proof
- STEP 1: Model Construction
- STEP 2: TDD - Write tests first

This script generates:
1. Input tokens [seq_len]
2. Expected logits [vocab_size] (using simplified reference implementation)
3. Intermediate outputs for component validation

Note: Full forward pass validation requires PyTorch reference implementation.
This script provides simplified validation for testing infrastructure.
"""

import numpy as np
import struct
import os
import sys

# Import tensor writing utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_llama import Q_ALIGN, align_size

# Tolerances for hybrid numerical validation
EPSILON_ABS = 1e-4      # Absolute error (relaxed for forward pass)
EPSILON_REL = 1e-3      # Relative error

def write_tensor_binary(filename, data):
    """
    Write tensor to binary file in simple format:
    - uint32_t: number of elements
    - float32[]: data (little-endian)
    """
    data_flat = np.ascontiguousarray(data.flatten(), dtype=np.float32)
    with open(filename, 'wb') as f:
        f.write(struct.pack('<I', len(data_flat)))
        f.write(data_flat.tobytes())

def write_tokens_binary(filename, tokens):
    """
    Write token IDs to binary file:
    - uint32_t: number of tokens
    - uint32_t[]: token IDs
    """
    tokens = np.asarray(tokens, dtype=np.uint32)
    with open(filename, 'wb') as f:
        f.write(struct.pack('<I', len(tokens)))
        f.write(tokens.tobytes())

def generate_simplified_forward_reference(
    tokens,
    vocab_size,
    dim,
    n_layers,
    n_heads,
    n_kv_heads,
    hidden_dim,
    max_seq_len,
    rope_theta,
    rms_norm_eps=1e-6,
    seed=42
):
    """
    Simplified forward pass reference implementation.
    
    This is a MINIMAL implementation for testing infrastructure.
    For full validation, use PyTorch reference model.
    
    Returns:
        logits: [vocab_size] output logits
    """
    np.random.seed(seed)
    seq_len = len(tokens)
    
    # Initialize random weights (simplified - not actual model weights)
    # In real validation, these would come from the model file
    
    # Step 1: Token embeddings (simplified)
    # In real model: x = token_embd[tokens]
    # For test: use random embeddings
    x = np.random.randn(seq_len, dim).astype(np.float32)
    
    # Step 2: Simplified forward through layers
    # In real model: full attention + MLP with residuals
    # For test: simple transformations
    for layer_idx in range(n_layers):
        # Simplified layer (not actual Transformer)
        x = x + 0.1 * np.random.randn(seq_len, dim).astype(np.float32)
        x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + rms_norm_eps)
    
    # Step 3: Final normalization
    x_final = x[-1]  # Last token only
    x_final = x_final / (np.linalg.norm(x_final) + rms_norm_eps)
    
    # Step 4: LM Head (simplified)
    # In real model: logits = x_final @ output.T
    # For test: random projection
    output_weight = np.random.randn(vocab_size, dim).astype(np.float32)
    logits = x_final @ output_weight.T
    
    return logits.astype(np.float32)

def generate_test_case(
    test_name,
    vocab_size=32000,
    dim=4096,
    n_layers=2,
    n_heads=32,
    n_kv_heads=8,
    hidden_dim=11008,
    max_seq_len=8192,
    rope_theta=500000.0,
    seq_len=4,
    seed=42
):
    """
    Generate a test case for forward pass validation.
    
    Args:
        test_name: Name of test case
        vocab_size: Vocabulary size
        dim: Model dimension
        n_layers: Number of layers
        n_heads: Number of attention heads
        n_kv_heads: Number of KV heads (GQA)
        hidden_dim: Hidden dimension for MLP
        max_seq_len: Maximum sequence length
        rope_theta: RoPE theta parameter
        seq_len: Sequence length for this test
        seed: Random seed
    
    Returns:
        dict with test data
    """
    np.random.seed(seed)
    
    # Generate input tokens
    tokens = np.random.randint(0, vocab_size, size=seq_len, dtype=np.uint32)
    
    # Generate expected logits (simplified reference)
    logits_expected = generate_simplified_forward_reference(
        tokens,
        vocab_size,
        dim,
        n_layers,
        n_heads,
        n_kv_heads,
        hidden_dim,
        max_seq_len,
        rope_theta,
        seed=seed
    )
    
    return {
        'name': test_name,
        'tokens': tokens,
        'logits_expected': logits_expected,
        'config': {
            'vocab_size': vocab_size,
            'dim': dim,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_kv_heads': n_kv_heads,
            'hidden_dim': hidden_dim,
            'max_seq_len': max_seq_len,
            'rope_theta': rope_theta,
            'seq_len': seq_len,
        }
    }

def main():
    """Generate test data for forward pass validation."""
    
    test_dir = "test_data/forward"
    os.makedirs(test_dir, exist_ok=True)
    
    print("=" * 70)
    print("Forward Pass Test Data Generator - Gold Standard (Simplified)")
    print("=" * 70)
    print(f"Directory: {os.path.abspath(test_dir)}")
    print(f"Note: This generates SIMPLIFIED reference data for infrastructure testing.")
    print(f"      For full validation, use PyTorch reference model.\n")
    
    # Test Case 1: Single token (incremental generation)
    print("Generating Test Case 1: Single token (incremental generation)...")
    test1 = generate_test_case(
        "single_token",
        seq_len=1,
        seed=42
    )
    write_tokens_binary(f"{test_dir}/test1_tokens.bin", test1['tokens'])
    write_tensor_binary(f"{test_dir}/test1_logits_expected.bin", test1['logits_expected'])
    print(f"  Tokens: {test1['tokens']}")
    print(f"  Logits shape: {test1['logits_expected'].shape}")
    print(f"  Logits range: [{test1['logits_expected'].min():.6f}, {test1['logits_expected'].max():.6f}]")
    
    # Test Case 2: Multiple tokens (prefill)
    print("\nGenerating Test Case 2: Multiple tokens (prefill)...")
    test2 = generate_test_case(
        "prefill",
        seq_len=4,
        seed=123
    )
    write_tokens_binary(f"{test_dir}/test2_tokens.bin", test2['tokens'])
    write_tensor_binary(f"{test_dir}/test2_logits_expected.bin", test2['logits_expected'])
    print(f"  Tokens: {test2['tokens']}")
    print(f"  Logits shape: {test2['logits_expected'].shape}")
    print(f"  Logits range: [{test2['logits_expected'].min():.6f}, {test2['logits_expected'].max():.6f}]")
    
    # Test Case 3: Longer sequence
    print("\nGenerating Test Case 3: Longer sequence...")
    test3 = generate_test_case(
        "long_sequence",
        seq_len=8,
        seed=456
    )
    write_tokens_binary(f"{test_dir}/test3_tokens.bin", test3['tokens'])
    write_tensor_binary(f"{test_dir}/test3_logits_expected.bin", test3['logits_expected'])
    print(f"  Tokens: {test3['tokens']}")
    print(f"  Logits shape: {test3['logits_expected'].shape}")
    print(f"  Logits range: [{test3['logits_expected'].min():.6f}, {test3['logits_expected'].max():.6f}]")
    
    # Save test configuration
    print("\nSaving test configuration...")
    config_file = f"{test_dir}/test_config.txt"
    with open(config_file, 'w') as f:
        f.write("Forward Pass Test Configuration\n")
        f.write("=" * 50 + "\n\n")
        for test_name in ['test1', 'test2', 'test3']:
            f.write(f"{test_name}:\n")
            f.write(f"  vocab_size: {test1['config']['vocab_size']}\n")
            f.write(f"  dim: {test1['config']['dim']}\n")
            f.write(f"  n_layers: {test1['config']['n_layers']}\n")
            f.write(f"  n_heads: {test1['config']['n_heads']}\n")
            f.write(f"  n_kv_heads: {test1['config']['n_kv_heads']}\n")
            f.write(f"  hidden_dim: {test1['config']['hidden_dim']}\n")
            f.write(f"  max_seq_len: {test1['config']['max_seq_len']}\n")
            f.write(f"  rope_theta: {test1['config']['rope_theta']}\n")
            f.write("\n")
        f.write(f"Tolerances:\n")
        f.write(f"  EPSILON_ABS: {EPSILON_ABS}\n")
        f.write(f"  EPSILON_REL: {EPSILON_REL}\n")
    
    print(f"  ✓ Configuration saved to {config_file}")
    print("\n" + "=" * 70)
    print("Test data generation complete!")
    print("=" * 70)
    print("\nNote: These are SIMPLIFIED reference outputs for infrastructure testing.")
    print("      For full validation against actual model, use PyTorch reference.")

if __name__ == "__main__":
    main()

