#!/usr/bin/env python3
"""
Python Gold Standard: Generate test data for Q4_0 dequantization.
Conforms to TDD methodology - tests define the interface.
"""

import numpy as np
import struct
import os

# Q4_0 block structure: 16 bytes of quantized data + 4 bytes scale
# Each byte contains 2 quantized values (4 bits each, range 0-15)
# Dequantization: value = (quantized - 8) * scale

def create_q4_0_block(values, scale):
    """
    Create a Q4_0 block from 32 float values.
    
    Args:
        values: Array of 32 floats
        scale: Scale factor for quantization
    
    Returns:
        bytes: 20-byte Q4_0 block
    """
    assert len(values) == 32, "Q4_0 block must contain exactly 32 values"
    
    # Quantize: q = round((value / scale) + 8), clamped to [0, 15]
    quantized = np.round(np.clip(values / scale + 8, 0, 15)).astype(np.uint8)
    
    # Pack into 16 bytes (2 values per byte)
    packed = bytearray(16)
    for i in range(16):
        # Lower 4 bits: quantized[i*2]
        # Upper 4 bits: quantized[i*2 + 1]
        packed[i] = (quantized[i*2 + 1] << 4) | quantized[i*2]
    
    # Create block: 16 bytes data + 4 bytes scale
    block = packed + struct.pack('<f', scale)
    return bytes(block)

def dequantize_q4_0_block(block):
    """
    Dequantize a Q4_0 block to 32 floats (Python reference).
    
    Args:
        block: 20-byte Q4_0 block
    
    Returns:
        np.ndarray: 32 float32 values
    """
    assert len(block) == 20, "Q4_0 block must be 20 bytes"
    
    # Extract scale (last 4 bytes)
    scale = struct.unpack('<f', block[16:20])[0]
    
    # Extract quantized values (first 16 bytes)
    packed = block[0:16]
    
    # Unpack nibbles
    quantized = np.zeros(32, dtype=np.uint8)
    for i in range(16):
        quantized[i*2] = packed[i] & 0x0F      # Lower 4 bits
        quantized[i*2 + 1] = (packed[i] >> 4) & 0x0F  # Upper 4 bits
    
    # Dequantize: value = (quantized - 8) * scale
    values = (quantized.astype(np.float32) - 8.0) * scale
    
    return values

def generate_test_cases():
    """Generate test cases for Q4_0 dequantization."""
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Test Case 1: Simple uniform values
    print("Generating test case 1: Uniform values...")
    values1 = np.full(32, 1.5, dtype=np.float32)
    scale1 = 0.1
    block1 = create_q4_0_block(values1, scale1)
    expected1 = dequantize_q4_0_block(block1)
    
    with open(f"{test_dir}/dequantize_test1_block.bin", "wb") as f:
        f.write(block1)
    # Save as simple binary format: uint32_t count + float32[] data
    with open(f"{test_dir}/dequantize_test1_expected.bin", "wb") as f:
        f.write(struct.pack('<I', len(expected1)))
        f.write(expected1.tobytes())
    print(f"  Scale: {scale1}, Expected range: [{expected1.min():.6f}, {expected1.max():.6f}]")
    
    # Test Case 2: Random values
    print("Generating test case 2: Random values...")
    np.random.seed(42)
    values2 = np.random.randn(32).astype(np.float32) * 2.0
    scale2 = np.abs(values2).max() / 7.0  # Scale to fit in [-8, 7] range
    block2 = create_q4_0_block(values2, scale2)
    expected2 = dequantize_q4_0_block(block2)
    
    with open(f"{test_dir}/dequantize_test2_block.bin", "wb") as f:
        f.write(block2)
    with open(f"{test_dir}/dequantize_test2_expected.bin", "wb") as f:
        f.write(struct.pack('<I', len(expected2)))
        f.write(expected2.tobytes())
    print(f"  Scale: {scale2:.6f}, Expected range: [{expected2.min():.6f}, {expected2.max():.6f}]")
    
    # Test Case 3: Edge case - zero scale
    print("Generating test case 3: Zero scale...")
    values3 = np.random.randn(32).astype(np.float32)
    scale3 = 0.0
    block3 = create_q4_0_block(values3, scale3)
    expected3 = dequantize_q4_0_block(block3)
    
    with open(f"{test_dir}/dequantize_test3_block.bin", "wb") as f:
        f.write(block3)
    with open(f"{test_dir}/dequantize_test3_expected.bin", "wb") as f:
        f.write(struct.pack('<I', len(expected3)))
        f.write(expected3.tobytes())
    print(f"  Scale: {scale3}, All values should be 0.0")
    
    # Test Case 4: Large scale
    print("Generating test case 4: Large scale...")
    values4 = np.linspace(-10.0, 10.0, 32, dtype=np.float32)
    scale4 = 2.0
    block4 = create_q4_0_block(values4, scale4)
    expected4 = dequantize_q4_0_block(block4)
    
    with open(f"{test_dir}/dequantize_test4_block.bin", "wb") as f:
        f.write(block4)
    with open(f"{test_dir}/dequantize_test4_expected.bin", "wb") as f:
        f.write(struct.pack('<I', len(expected4)))
        f.write(expected4.tobytes())
    print(f"  Scale: {scale4}, Expected range: [{expected4.min():.6f}, {expected4.max():.6f}]")
    
    print(f"\n✓ Generated {4} test cases in {test_dir}/")

if __name__ == "__main__":
    generate_test_cases()

