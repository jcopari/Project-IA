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

def pad_vocab_size(vocab_size):
    """Garante que vocab_size seja múltiplo de 32 para compatibilidade com Q4_0."""
    remainder = vocab_size % 32
    if remainder == 0:
        return vocab_size, 0
    padding = 32 - remainder
    padded_size = vocab_size + padding
    return padded_size, padding

def write_tensor(f, name, data, pad_rows=False):
    """Escreve tensor com alinhamento garantido e zero-copy quando possível.
    
    Args:
        f: File handle
        name: Tensor name (for logging)
        data: NumPy array
        pad_rows: Se True, adiciona padding nas linhas para múltiplo de 32 (para vocab_size)
    """
    pos = f.tell()
    
    # Calcular padding necessário para alinhamento de arquivo
    padding = (Q_ALIGN - (pos % Q_ALIGN)) % Q_ALIGN
    
    # Escrever padding de arquivo
    if padding > 0:
        f.write(b'\x00' * padding)
    
    # Padding de linhas (para vocab_size - CRITICAL FIX)
    original_shape = data.shape
    if pad_rows and len(data.shape) >= 1:
        original_rows = data.shape[0]
        padded_rows, row_padding = pad_vocab_size(original_rows)
        
        if row_padding > 0:
            # Adicionar padding com zeros nas linhas
            padding_shape = (row_padding,) + data.shape[1:]
            padding_data = np.zeros(padding_shape, dtype=data.dtype)
            data = np.vstack([data, padding_data])
            print(f"  PADDED {name}: {original_rows} → {padded_rows} rows (+{row_padding} padding)")
    
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
    
    # CRITICAL FIX: Garantir vocab_size múltiplo de 32 (para compatibilidade com Q4_0)
    original_vocab_size = config['vocab_size']
    padded_vocab_size, vocab_padding = pad_vocab_size(original_vocab_size)
    if vocab_padding > 0:
        print(f"WARNING: vocab_size {original_vocab_size} não é múltiplo de 32. "
              f"Será preenchido para {padded_vocab_size} no arquivo.")
        # Manter original_vocab_size para gerar dados, mas usar padded_vocab_size no header
        config_for_header = config.copy()
        config_for_header['vocab_size'] = padded_vocab_size
    else:
        config_for_header = config
    
    # Ensure dim is multiple of 32 for Q4_0 compatibility
    assert config['dim'] % 32 == 0, f"dim ({config['dim']}) must be multiple of 32"
    assert config['hidden_dim'] % 32 == 0, f"hidden_dim ({config['hidden_dim']}) must be multiple of 32"
    
    head_dim = config['dim'] // config['n_heads']
    kv_dim = config['n_kv_heads'] * head_dim
    
    print("Generating dummy model (FASE 3 layout)...")
    print(f"Config: vocab_size={config['vocab_size']} (original), dim={config['dim']}, "
          f"n_layers={config['n_layers']}, hidden_dim={config['hidden_dim']}")
    if vocab_padding > 0:
        print(f"  vocab_size no arquivo: {padded_vocab_size} (com padding)")
    print(f"  head_dim={head_dim}, kv_dim={kv_dim}")
    
    with open(output_path, 'wb') as f:
        # Escrever header com vocab_size padded
        write_header(f, config_for_header)
        
        print("\nWriting tensors...")
        
        # 1. Token embeddings [vocab_size, dim] (FP32)
        # CRITICAL: Usar vocab_size original para gerar dados, padding será adicionado automaticamente
        embd = np.random.randn(original_vocab_size, config['dim']).astype(np.float32)
        write_tensor(f, 'token_embd.weight', embd, pad_rows=True)
        
        # 2. Output normalization [dim] (FP32)
        output_norm = np.random.randn(config['dim']).astype(np.float32)
        write_tensor(f, 'output_norm.weight', output_norm)
        
        # 3. Output projection [vocab_size, dim] (FP32)
        # CRITICAL: Esta é a camada crítica que precisa de padding para Q4_0
        output = np.random.randn(original_vocab_size, config['dim']).astype(np.float32)
        write_tensor(f, 'output.weight', output, pad_rows=True)
        
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

def write_tokenizer(tokenizer_path, vocab_size=32000):
    """Export tokenizer to binary format for Qorus-IA.
    
    Creates a minimal tokenizer with:
    - Vocab: 256 base tokens (bytes 0-255) + special tokens
    - BPE merges: Empty (simplified for now)
    - Special tokens: BOS, EOS, PAD
    
    Args:
        tokenizer_path: Output path for tokenizer binary file
        vocab_size: Vocabulary size parameter (used for special token IDs, not actual vocab size)
    """
    TOKENIZER_MAGIC = 0x51544B52  # 'QTKR'
    TOKENIZER_VERSION = 1
    
    # Base vocabulary: 256 bytes (0-255)
    BASE_VOCAB_SIZE = 256
    
    # Special token IDs (after base vocab)
    BOS_TOKEN_ID = BASE_VOCAB_SIZE
    EOS_TOKEN_ID = BASE_VOCAB_SIZE + 1
    PAD_TOKEN_ID = BASE_VOCAB_SIZE + 2
    
    # Calculate actual vocab size (base + special tokens)
    actual_vocab_size = BASE_VOCAB_SIZE + 3  # 256 base + 3 special = 259
    
    print(f"Writing tokenizer to {tokenizer_path}...")
    print(f"  Base vocab: {BASE_VOCAB_SIZE} tokens (bytes 0-255)")
    print(f"  Special tokens: BOS={BOS_TOKEN_ID}, EOS={EOS_TOKEN_ID}, PAD={PAD_TOKEN_ID}")
    print(f"  Total vocab size: {actual_vocab_size}")
    
    with open(tokenizer_path, 'wb') as f:
        # Write header (32 bytes)
        header = struct.pack('<IIIIIIII',
            TOKENIZER_MAGIC,      # magic
            TOKENIZER_VERSION,    # version
            actual_vocab_size,    # vocab_size
            0,                    # num_merges (simplified: no BPE merges for now)
            BOS_TOKEN_ID,        # bos_token_id
            EOS_TOKEN_ID,        # eos_token_id
            PAD_TOKEN_ID,        # pad_token_id
            0                     # reserved
        )
        assert len(header) == 32, f"Header size mismatch: {len(header)}"
        f.write(header)
        
        # Write vocab: Base tokens (bytes 0-255)
        for i in range(256):
            token_bytes = bytes([i])  # Single byte token
            length = len(token_bytes)
            assert length == 1, f"Base token length must be 1, got {length}"
            f.write(struct.pack('B', length))
            f.write(token_bytes)
        
        # Write special tokens
        special_tokens = [
            (BOS_TOKEN_ID, b"<|begin_of_text|>"),
            (EOS_TOKEN_ID, b"<|end_of_text|>"),
            (PAD_TOKEN_ID, b"<|finetune_right_pad_id|>"),
        ]
        
        for token_id, token_bytes in special_tokens:
            length = len(token_bytes)
            assert length > 0 and length <= 255, f"Token length invalid: {length}"
            f.write(struct.pack('B', length))
            f.write(token_bytes)
        
        # No BPE merges for now (simplified)
        # Merges would be written here if implemented
        
    file_size = os.path.getsize(tokenizer_path)
    print(f"✓ Wrote tokenizer: {tokenizer_path}")
    print(f"  Total size: {file_size:,} bytes")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--tokenizer":
        # Generate tokenizer only
        tokenizer_path = sys.argv[2] if len(sys.argv) > 2 else "tokenizer.bin"
        vocab_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32000
        write_tokenizer(tokenizer_path, vocab_size)
    else:
        # Generate model (default)
        output_path = sys.argv[1] if len(sys.argv) > 1 else "model_dummy.qorus"
        n_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        generate_dummy_model(output_path, n_layers=n_layers)
