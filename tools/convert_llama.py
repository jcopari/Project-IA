#!/usr/bin/env python3
"""
Script Python: GGUF -> Qorus Binary (Zero-Parse)
Versão otimizada com zero-copy e validação de contiguidade.

Conforme FASE 1 - Passo 1.2
"""

import struct
import numpy as np
import os
import sys

Q_ALIGN = 64
Q_MAGIC = 0x514F5231  # 'QOR1'
Q_HEADER_SIZE = 64

def align_size(size):
    """Alinhamento de tamanho para múltiplo de Q_ALIGN."""
    return (size + Q_ALIGN - 1) & ~(Q_ALIGN - 1)

def write_header(f, config):
    """Escreve header de 64 bytes."""
    header = struct.pack(
        '<IIIIIIIIIf6I',  # Little-endian, 9 uint32_t + 1 float + 6 uint32_t reservados
        Q_MAGIC,
        config.get('version', 1),
        config.get('vocab_size', 32000),
        config.get('dim', 4096),
        config.get('hidden_dim', 11008),
        config.get('n_layers', 32),
        config.get('n_heads', 32),
        config.get('n_kv_heads', 8),  # GQA
        config.get('max_seq_len', 8192),
        config.get('rope_freq_base', 500000.0),
        *([0] * 6)  # 6 reservados (24 bytes)
    )
    
    assert len(header) == Q_HEADER_SIZE, f"Header size mismatch: {len(header)}"
    f.write(header)
    
    # Verificar alinhamento
    pos = f.tell()
    assert pos % Q_ALIGN == 0, f"Header not aligned: {pos}"

def write_tensor(f, name, data):
    """Escreve tensor com alinhamento garantido e zero-copy quando possível."""
    pos = f.tell()
    
    # Calcular padding necessário
    padding = (Q_ALIGN - (pos % Q_ALIGN)) % Q_ALIGN
    
    # Escrever padding
    if padding > 0:
        f.write(b'\x00' * padding)
    
    # Validação de contiguidade e endianness
    if not data.flags['C_CONTIGUOUS']:
        print(f"WARNING: {name} não é C-contíguo, forçando cópia")
        data = np.ascontiguousarray(data, dtype=data.dtype)
    
    if data.dtype.byteorder not in ('<', '='):
        print(f"WARNING: {name} é big-endian, convertendo")
        data = data.astype(data.dtype.newbyteorder('<'))
    
    # Zero-copy: usar memoryview para evitar cópia intermediária
    # Isso funciona porque data já é C-contíguo e little-endian
    f.write(memoryview(data))
    
    # Verificar alinhamento do próximo tensor
    new_pos = f.tell()
    assert new_pos % Q_ALIGN == 0, \
        f"Tensor {name} desalinhado. Pos: {new_pos}"
    
    # Log informativo
    print(f"Wrote {name:<30} | {str(data.dtype):<8} | "
          f"Shape: {str(data.shape):<15} | Offset: {pos+padding:08x} | "
          f"Size: {data.nbytes} bytes")

def calculate_q4_0_size(rows, cols):
    """Calculate size in bytes for Q4_0 tensor [rows, cols] where cols must be multiple of 32."""
    assert cols % 32 == 0, f"cols ({cols}) must be multiple of 32 for Q4_0"
    blocks_per_row = cols // 32
    bytes_per_block = 20  # 16 bytes qs + 4 bytes scale
    return rows * blocks_per_row * bytes_per_block

def generate_dummy_model(output_path, n_layers=2):
    """Gera modelo dummy completo para validação (FASE 3).
    
    Layout do arquivo:
    1. Header (64 bytes)
    2. token_embd.weight [vocab_size, dim] (FP32)
    3. output_norm.weight [dim] (FP32)
    4. output.weight [vocab_size, dim] (FP32)
    5. Para cada layer i (0..n_layers-1):
       - layers.{i}.attn_norm.weight [dim] (FP32)
       - layers.{i}.wq.weight [dim, dim] (Q4_0)
       - layers.{i}.wk.weight [dim, n_kv_heads * head_dim] (Q4_0)
       - layers.{i}.wv.weight [dim, n_kv_heads * head_dim] (Q4_0)
       - layers.{i}.wo.weight [dim, dim] (Q4_0)
       - layers.{i}.ffn_norm.weight [dim] (FP32)
       - layers.{i}.w_gate.weight [dim, hidden_dim] (Q4_0)
       - layers.{i}.w_up.weight [dim, hidden_dim] (Q4_0)
       - layers.{i}.w_down.weight [hidden_dim, dim] (Q4_0)
    """
    config = {
        'version': 1,
        'vocab_size': 32000,
        'dim': 4096,
        'hidden_dim': 11008,
        'n_layers': n_layers,
        'n_heads': 32,
        'n_kv_heads': 8,
        'max_seq_len': 8192,
        'rope_freq_base': 500000.0,
    }
    
    # Ensure dim is multiple of 32 for Q4_0 compatibility
    assert config['dim'] % 32 == 0, f"dim ({config['dim']}) must be multiple of 32"
    assert config['hidden_dim'] % 32 == 0, f"hidden_dim ({config['hidden_dim']}) must be multiple of 32"
    
    head_dim = config['dim'] // config['n_heads']
    kv_dim = config['n_kv_heads'] * head_dim
    
    print("Generating dummy model (FASE 3 layout)...")
    print(f"Config: vocab_size={config['vocab_size']}, dim={config['dim']}, "
          f"n_layers={config['n_layers']}, hidden_dim={config['hidden_dim']}")
    print(f"  head_dim={head_dim}, kv_dim={kv_dim}")
    
    with open(output_path, 'wb') as f:
        # Escrever header
        write_header(f, config)
        
        print("\nWriting tensors...")
        
        # 1. Token embeddings [vocab_size, dim] (FP32)
        embd = np.random.randn(config['vocab_size'], config['dim']).astype(np.float32)
        write_tensor(f, 'token_embd.weight', embd)
        
        # 2. Output normalization [dim] (FP32)
        output_norm = np.random.randn(config['dim']).astype(np.float32)
        write_tensor(f, 'output_norm.weight', output_norm)
        
        # 3. Output projection [vocab_size, dim] (FP32)
        output = np.random.randn(config['vocab_size'], config['dim']).astype(np.float32)
        write_tensor(f, 'output.weight', output)
        
        # 4. Layers
        for layer_idx in range(config['n_layers']):
            print(f"\n  Layer {layer_idx}:")
            
            # Attention norm [dim] (FP32)
            attn_norm = np.random.randn(config['dim']).astype(np.float32)
            write_tensor(f, f'layers.{layer_idx}.attn_norm.weight', attn_norm)
            
            # Q projection [dim, dim] (Q4_0) - simulated as FP32 for now
            # In real implementation, this would be Q4_0 blocks
            wq = np.random.randn(config['dim'], config['dim']).astype(np.float32)
            write_tensor(f, f'layers.{layer_idx}.wq.weight', wq)
            
            # K projection [dim, kv_dim] (Q4_0)
            wk = np.random.randn(config['dim'], kv_dim).astype(np.float32)
            write_tensor(f, f'layers.{layer_idx}.wk.weight', wk)
            
            # V projection [dim, kv_dim] (Q4_0)
            wv = np.random.randn(config['dim'], kv_dim).astype(np.float32)
            write_tensor(f, f'layers.{layer_idx}.wv.weight', wv)
            
            # Output projection [dim, dim] (Q4_0)
            wo = np.random.randn(config['dim'], config['dim']).astype(np.float32)
            write_tensor(f, f'layers.{layer_idx}.wo.weight', wo)
            
            # FFN norm [dim] (FP32)
            ffn_norm = np.random.randn(config['dim']).astype(np.float32)
            write_tensor(f, f'layers.{layer_idx}.ffn_norm.weight', ffn_norm)
            
            # Gate projection [dim, hidden_dim] (Q4_0)
            w_gate = np.random.randn(config['dim'], config['hidden_dim']).astype(np.float32)
            write_tensor(f, f'layers.{layer_idx}.w_gate.weight', w_gate)
            
            # Up projection [dim, hidden_dim] (Q4_0)
            w_up = np.random.randn(config['dim'], config['hidden_dim']).astype(np.float32)
            write_tensor(f, f'layers.{layer_idx}.w_up.weight', w_up)
            
            # Down projection [hidden_dim, dim] (Q4_0)
            w_down = np.random.randn(config['hidden_dim'], config['dim']).astype(np.float32)
            write_tensor(f, f'layers.{layer_idx}.w_down.weight', w_down)
    
    file_size = os.path.getsize(output_path)
    print(f"\n✓ Generated dummy model: {output_path}")
    print(f"  Total size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    print(f"  Header: {Q_HEADER_SIZE} bytes")
    print(f"  Tensors: {file_size - Q_HEADER_SIZE:,} bytes")
    print(f"  Layers: {config['n_layers']}")

if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else "model_dummy.qorus"
    n_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    generate_dummy_model(output_path, n_layers=n_layers)
