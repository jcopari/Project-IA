# Flow.md - Fluxo Completo de Execução do Forward Pass

Este documento mapeia **TODAS** as funções executadas durante o forward pass do Qorus-IA, mostrando de onde vem cada dado e para onde vai, função por função.

---

## 1. ENTRADA: `llama_forward`

**Localização:** `src/models/model.c:1435`

**Chamada por:** Código externo (testes, aplicação)

**Parâmetros de Entrada:**
- `model`: Ponteiro para estrutura do modelo (contém pesos, configuração)
- `ctx`: Contexto de memória (arena, KV cache)
- `tokens`: Array de token IDs `[seq_len]` (entrada do usuário)
- `seq_len`: Número de tokens (1 para geração incremental, >1 para prefill)
- `pos`: Posição absoluta na sequência (0 para primeira chamada)
- `logits`: Buffer de saída `[vocab_size]` (alocado pelo chamador)

**Validações:**
- Verifica ponteiros não-NULL
- Verifica `seq_len > 0` e `seq_len <= max_seq_len`
- Verifica `pos < max_seq_len`
- Verifica arena e KV cache alocados

**Alocações na Arena:**
- `x`: Buffer `[seq_len, dim]` para embeddings de tokens
- `layer_out`: Buffer `[seq_len, dim]` para saída de cada camada
- `x_final`: Buffer `[seq_len, dim]` para normalização final

**Fluxo de Execução:**

### 1.1 Token Embedding Lookup
```
llama_forward
  └─> token_embedding_lookup(model->token_embd, tokens, seq_len, x)
      └─> [memcpy] Copia embeddings de token_embd->data para x
```

**Dados:**
- **Entrada:** `tokens[seq_len]` (IDs), `model->token_embd->data[vocab_size, dim]` (embeddings)
- **Saída:** `x[seq_len, dim]` (embeddings dos tokens)

### 1.2 Loop pelas Camadas (n_layers vezes)
```
llama_forward
  └─> Para cada layer l de 0 até n_layers-1:
      └─> llama_layer_forward(&model->layers[l], ctx, &model->config, x, layer_out, l, seq_len, pos)
          └─> [Após retorno] Swap: x = layer_out, layer_out = x
```

**Dados:**
- **Entrada:** `x[seq_len, dim]` (saída da camada anterior ou embeddings iniciais)
- **Saída:** `layer_out[seq_len, dim]` (entrada para próxima camada)

### 1.3 Normalização Final
```
llama_forward
  └─> q_rmsnorm_f32_avx2(x, model->output_norm->data, x_final, dim, rms_norm_eps)
```

**Dados:**
- **Entrada:** `x[seq_len, dim]` (saída da última camada), `model->output_norm->data[dim]` (pesos)
- **Saída:** `x_final[seq_len, dim]` (normalizado)

### 1.4 LM Head Projection
```
llama_forward
  └─> q_matmul_f32_avx2(&last_token_tensor, &output_t_tensor, &logits_tensor, ctx)
```

**Dados:**
- **Entrada:** `last_token[1, dim]` (último token de x_final), `model->output->data[vocab_size, dim]` (pesos transpostos)
- **Saída:** `logits[1, vocab_size]` (probabilidades de cada token)

**Retorno:** `Q_OK` em sucesso, código de erro em caso de falha

---

## 2. CAMADA: `llama_layer_forward`

**Localização:** `src/models/model.c:1319`

**Chamada por:** `llama_forward` (loop pelas camadas)

**Parâmetros de Entrada:**
- `layer`: Ponteiro para estrutura da camada (pesos Q4_0, normas)
- `ctx`: Contexto de memória (arena)
- `config`: Configuração do modelo (dimensões, etc.)
- `x`: Input `[seq_len, dim]` (da camada anterior ou embeddings)
- `output`: Output `[seq_len, dim]` (para próxima camada)
- `layer_idx`: Índice da camada (0..n_layers-1)
- `seq_len`: Comprimento da sequência
- `pos`: Posição absoluta na sequência

**Alocações na Arena:**
- `attn_out`: Buffer `[seq_len, dim]` para saída da atenção
- `mlp_out`: Buffer `[seq_len, dim]` para saída do MLP
- `x_norm`: Buffer `[seq_len, dim]` para normalização pré-attention
- `x_norm_mlp`: Buffer `[seq_len, dim]` para normalização pré-MLP

**Fluxo de Execução:**

### 2.1 Attention Block
```
llama_layer_forward
  └─> llama_attention_forward(layer, ctx, config, x, attn_out, layer_idx, seq_len, pos)
      └─> [Retorna attn_out[seq_len, dim]]
```

**Dados:**
- **Entrada:** `x[seq_len, dim]`
- **Saída:** `attn_out[seq_len, dim]`

### 2.2 Residual Connection (Attention)
```
llama_layer_forward
  └─> q_add_f32_avx2(&x_tensor, &attn_tensor, &x_residual)
```

**Dados:**
- **Entrada:** `x[seq_len, dim]` (original), `attn_out[seq_len, dim]` (atenção)
- **Saída:** `x_norm[seq_len, dim]` (x + attn_out, reutiliza buffer x_norm)

### 2.3 Pre-MLP Normalization
```
llama_layer_forward
  └─> q_rmsnorm_f32_avx2(x_norm, layer->ffn_norm->data, x_norm_mlp, dim, rms_norm_eps)
```

**Dados:**
- **Entrada:** `x_norm[seq_len, dim]` (residual), `layer->ffn_norm->data[dim]` (pesos)
- **Saída:** `x_norm_mlp[seq_len, dim]` (normalizado)

### 2.4 MLP Block
```
llama_layer_forward
  └─> llama_mlp_forward(layer, ctx, config, x_norm_mlp, mlp_out, seq_len)
      └─> [Retorna mlp_out[seq_len, dim]]
```

**Dados:**
- **Entrada:** `x_norm_mlp[seq_len, dim]`
- **Saída:** `mlp_out[seq_len, dim]`

### 2.5 Residual Connection (MLP)
```
llama_layer_forward
  └─> q_add_f32_avx2(&x_residual, &mlp_tensor, &output_tensor)
```

**Dados:**
- **Entrada:** `x_norm[seq_len, dim]` (residual após attention), `mlp_out[seq_len, dim]` (MLP)
- **Saída:** `output[seq_len, dim]` (x_norm + mlp_out)

**Retorno:** `Q_OK` em sucesso

---

## 3. ATTENTION: `llama_attention_forward`

**Localização:** `src/models/model.c:715`

**Chamada por:** `llama_layer_forward`

**Parâmetros de Entrada:**
- `layer`: Estrutura da camada (pesos wq, wk, wv, wo, attn_norm)
- `ctx`: Contexto (arena, KV cache)
- `config`: Configuração (dimensões, n_heads, n_kv_heads, rope_theta)
- `x`: Input `[seq_len, dim]`
- `output`: Output `[seq_len, dim]`
- `layer_idx`: Índice da camada
- `seq_len`: Comprimento da sequência
- `pos`: Posição absoluta

**Alocações na Arena:**
- `x_norm`: Buffer `[seq_len, dim]` para normalização
- `q_buf`: Buffer `[seq_len, dim]` para projeções Q
- `k_buf`: Buffer `[seq_len, n_kv_heads * head_dim]` para projeções K
- `v_buf`: Buffer `[seq_len, n_kv_heads * head_dim]` para projeções V
- `attn_out`: Buffer `[seq_len, dim]` para saída temporária
- `cos_buf`: Buffer `[head_dim]` para cos RoPE
- `sin_buf`: Buffer `[head_dim]` para sin RoPE
- `q_rope_buf`: Buffer `[seq_len, dim]` para Q após RoPE
- `k_rope_buf`: Buffer `[seq_len, n_kv_heads * head_dim]` para K após RoPE
- `scores_buf`: Buffer `[seq_len, seq_len]` para attention scores
- `q_heads`: Buffer `[n_heads, seq_len, head_dim]` reorganizado
- `k_heads`: Buffer `[n_kv_heads, seq_len, head_dim]` reorganizado
- `v_heads`: Buffer `[n_kv_heads, seq_len, head_dim]` reorganizado
- `attn_head_buf`: Buffer `[seq_len, head_dim]` para saída de cada head
- `k_t_buf`: Buffer `[head_dim, seq_len]` para K transposto

**Fluxo de Execução:**

### 3.1 Pre-Attention Normalization
```
llama_attention_forward
  └─> q_rmsnorm_f32_avx2(x, layer->attn_norm->data, x_norm, dim, rms_norm_eps)
```

**Dados:**
- **Entrada:** `x[seq_len, dim]`, `layer->attn_norm->data[dim]`
- **Saída:** `x_norm[seq_len, dim]`

### 3.2 Q Projection (para cada token)
```
llama_attention_forward
  └─> Para cada i de 0 até seq_len-1:
      └─> q_gemv_q4_f32_avx2(layer->wq, x_row, q_row)
```

**Dados:**
- **Entrada:** `x_norm[i, dim]` (linha i), `layer->wq->data[M=dim, N=dim]` (pesos Q4_0)
- **Saída:** `q_buf[i, dim]` (linha i)

**Chamadas Internas:**
- `q_process_block_avx2()` (múltiplas vezes por linha, processa blocos de 32)

### 3.3 K Projection (para cada token)
```
llama_attention_forward
  └─> Para cada i de 0 até seq_len-1:
      └─> q_gemv_q4_f32_avx2(layer->wk, x_row, k_row)
```

**Dados:**
- **Entrada:** `x_norm[i, dim]` (linha i), `layer->wk->data[M=dim, N=n_kv_heads*head_dim]` (pesos Q4_0)
- **Saída:** `k_buf[i, n_kv_heads*head_dim]` (linha i)

### 3.4 V Projection (para cada token)
```
llama_attention_forward
  └─> Para cada i de 0 até seq_len-1:
      └─> q_gemv_q4_f32_avx2(layer->wv, x_row, v_row)
```

**Dados:**
- **Entrada:** `x_norm[i, dim]` (linha i), `layer->wv->data[M=dim, N=n_kv_heads*head_dim]` (pesos Q4_0)
- **Saída:** `v_buf[i, n_kv_heads*head_dim]` (linha i)

### 3.5 RoPE Application (para cada token e cada head)
```
llama_attention_forward
  └─> Para cada t de 0 até seq_len-1:
      └─> generate_rope_cos_sin(rope_theta, head_dim, token_pos, cos_buf, sin_buf)
          └─> [Usa cosf(), sinf()] Calcula cos/sin para cada par
      └─> Para cada Q head h de 0 até n_heads-1:
          └─> q_rope_f32_avx2(q_head, cos_buf, sin_buf, q_head_out, head_dim)
      └─> Para cada K head h de 0 até n_kv_heads-1:
          └─> q_rope_f32_avx2(k_head, cos_buf, sin_buf, k_head_out, head_dim)
```

**Dados:**
- **Entrada Q:** `q_buf[t, h*head_dim : (h+1)*head_dim]`, `cos_buf[head_dim]`, `sin_buf[head_dim]`
- **Saída Q:** `q_rope_buf[t, h*head_dim : (h+1)*head_dim]`
- **Entrada K:** `k_buf[t, h*head_dim : (h+1)*head_dim]`, `cos_buf[head_dim]`, `sin_buf[head_dim]`
- **Saída K:** `k_rope_buf[t, h*head_dim : (h+1)*head_dim]`

### 3.6 KV Cache Update (para cada token e cada KV head)
```
llama_attention_forward
  └─> Para cada t de 0 até seq_len-1:
      └─> Para cada KV head h de 0 até n_kv_heads-1:
          └─> get_kv_cache_ptr(ctx, config, layer_idx, h, cache_pos, true)  // K
              └─> [memcpy] Copia k_rope_buf para KV cache
          └─> get_kv_cache_ptr(ctx, config, layer_idx, h, cache_pos, false) // V
              └─> [memcpy] Copia v_buf para KV cache
```

**Dados:**
- **Entrada K:** `k_rope_buf[t, h*head_dim : (h+1)*head_dim]`
- **Destino K:** `ctx->kv_buffer[layer_idx, h, cache_pos, K]`
- **Entrada V:** `v_buf[t, h*head_dim : (h+1)*head_dim]`
- **Destino V:** `ctx->kv_buffer[layer_idx, h, cache_pos, V]`

### 3.7 Reorganização Q/K/V (reshape para layout por head)
```
llama_attention_forward
  └─> [memcpy] Reorganiza q_rope_buf[seq_len, dim] -> q_heads[n_heads, seq_len, head_dim]
  └─> [memcpy] Reorganiza k_rope_buf[seq_len, n_kv_heads*head_dim] -> k_heads[n_kv_heads, seq_len, head_dim]
  └─> [memcpy] Reorganiza v_buf[seq_len, n_kv_heads*head_dim] -> v_heads[n_kv_heads, seq_len, head_dim]
```

**Dados:**
- **Entrada:** `q_rope_buf`, `k_rope_buf`, `v_buf`
- **Saída:** `q_heads`, `k_heads`, `v_heads` (layout reorganizado)

### 3.8 Attention Computation (para cada query head)
```
llama_attention_forward
  └─> Para cada query head qh de 0 até n_heads-1:
      └─> [Transpose manual] k_heads[kv_head_idx] -> k_t_buf[head_dim, seq_len]
      └─> q_matmul_f32_avx2(&q_head_tensor, &k_t_tensor, &scores_tensor, ctx)
          └─> transpose_blocked() [se necessário]
          └─> [AVX2 MatMul] Q @ K^T -> scores
      └─> [Scalar loop] scores *= 1/sqrt(head_dim)
      └─> q_causal_mask_f32_avx2(&scores_tensor, -1e9f)
      └─> Para cada linha i de 0 até seq_len-1:
          └─> q_softmax_f32_avx2(&scores_buf[i*seq_len], &probs_buf[i*seq_len], seq_len)
              └─> horizontal_max_avx() [encontra máximo]
              └─> exp_approx_avx() [calcula exp]
              └─> horizontal_sum_avx() [soma]
              └─> [Normaliza] probs /= sum
      └─> q_matmul_f32_avx2(&probs_tensor, &v_head_tensor, &attn_head_tensor, ctx)
          └─> transpose_blocked() [se necessário]
          └─> [AVX2 MatMul] probs @ V -> attn_head
      └─> [memcpy] Concatena attn_head_buf para output[seq_len, dim]
```

**Dados:**
- **Entrada Q:** `q_heads[qh, seq_len, head_dim]`
- **Entrada K:** `k_heads[kv_head_idx, seq_len, head_dim]` (transposto para `k_t_buf[head_dim, seq_len]`)
- **Saída Scores:** `scores_buf[seq_len, seq_len]` (Q @ K^T)
- **Saída Probs:** `probs_buf[seq_len, seq_len]` (softmax(scores))
- **Entrada V:** `v_heads[kv_head_idx, seq_len, head_dim]`
- **Saída Attn:** `attn_head_buf[seq_len, head_dim]` (probs @ V)
- **Saída Final:** `output[seq_len, dim]` (concatenação de todas as heads)

### 3.9 Output Projection (para cada token)
```
llama_attention_forward
  └─> Para cada i de 0 até seq_len-1:
      └─> q_gemv_q4_f32_avx2(layer->wo, attn_row, out_row)
```

**Dados:**
- **Entrada:** `output[i, dim]` (após concatenação), `layer->wo->data[M=dim, N=dim]` (pesos Q4_0)
- **Saída:** `attn_out[i, dim]` (reutiliza buffer output)

**Retorno:** `Q_OK` em sucesso

---

## 4. MLP: `llama_mlp_forward`

**Localização:** `src/models/model.c:1164`

**Chamada por:** `llama_layer_forward`

**Parâmetros de Entrada:**
- `layer`: Estrutura da camada (pesos w_gate, w_up, w_down)
- `ctx`: Contexto (arena)
- `config`: Configuração (dim, hidden_dim)
- `x`: Input `[seq_len, dim]`
- `output`: Output `[seq_len, dim]`
- `seq_len`: Comprimento da sequência

**Alocações na Arena:**
- `gate_buf`: Buffer `[seq_len, hidden_dim]` para gate projection
- `up_buf`: Buffer `[seq_len, hidden_dim]` para up projection
- `mul_buf`: Buffer `[seq_len, hidden_dim]` para gate * up
- `gate_silu`: Buffer `[seq_len, hidden_dim]` para SiLU(gate)

**Fluxo de Execução:**

### 4.1 Gate Projection (para cada token)
```
llama_mlp_forward
  └─> Para cada i de 0 até seq_len-1:
      └─> q_gemv_q4_f32_avx2(layer->w_gate, x_row, gate_row)
```

**Dados:**
- **Entrada:** `x[i, dim]`, `layer->w_gate->data[M=dim, N=hidden_dim]` (pesos Q4_0)
- **Saída:** `gate_buf[i, hidden_dim]`

### 4.2 Up Projection (para cada token)
```
llama_mlp_forward
  └─> Para cada i de 0 até seq_len-1:
      └─> q_gemv_q4_f32_avx2(layer->w_up, x_row, up_row)
```

**Dados:**
- **Entrada:** `x[i, dim]`, `layer->w_up->data[M=dim, N=hidden_dim]` (pesos Q4_0)
- **Saída:** `up_buf[i, hidden_dim]`

### 4.3 SiLU Activation
```
llama_mlp_forward
  └─> q_silu_f32_avx2(gate_buf, gate_silu, seq_len * hidden_dim)
      └─> Para cada elemento:
          └─> exp_approx_avx() [calcula exp(-x)]
          └─> sigmoid = 1 / (1 + exp(-x))
          └─> silu = x * sigmoid
```

**Dados:**
- **Entrada:** `gate_buf[seq_len * hidden_dim]`
- **Saída:** `gate_silu[seq_len * hidden_dim]`

### 4.4 Element-wise Multiply
```
llama_mlp_forward
  └─> q_mul_f32_avx2(&gate_silu_flat, &up_flat, &mul_tensor)
```

**Dados:**
- **Entrada:** `gate_silu[seq_len * hidden_dim]`, `up_buf[seq_len * hidden_dim]`
- **Saída:** `mul_buf[seq_len * hidden_dim]` (gate_silu * up)

### 4.5 Down Projection (para cada token)
```
llama_mlp_forward
  └─> Para cada i de 0 até seq_len-1:
      └─> q_gemv_q4_f32_avx2(layer->w_down, mul_row, out_row)
```

**Dados:**
- **Entrada:** `mul_buf[i, hidden_dim]`, `layer->w_down->data[M=hidden_dim, N=dim]` (pesos Q4_0)
- **Saída:** `output[i, dim]`

**Retorno:** `Q_OK` em sucesso

---

## 5. KERNELS AVX2 - Detalhamento Completo

### 5.1 `q_rmsnorm_f32_avx2`

**Localização:** `src/ops/avx2/rmsnorm.c:15`

**Chamada por:**
- `llama_attention_forward` (pre-attention norm)
- `llama_layer_forward` (pre-MLP norm)
- `llama_forward` (final norm)

**Parâmetros:**
- `x`: Input `[N]`, 32-byte aligned
- `weight`: Pesos `[N]`, 32-byte aligned
- `output`: Output `[N]`, 32-byte aligned
- `N`: Tamanho (deve ser múltiplo de 8)
- `eps`: Epsilon para estabilidade numérica

**Algoritmo:**
1. Calcula `sum_sq = sum(x^2)` usando AVX2
2. Calcula `scale = weight / sqrt(mean(x^2) + eps)`
3. Multiplica `output = x * scale` usando AVX2

**Dados:**
- **Entrada:** `x[N]`, `weight[N]`
- **Saída:** `output[N]`

---

### 5.2 `q_gemv_q4_f32_avx2`

**Localização:** `src/ops/avx2/matmul.c:90`

**Chamada por:**
- `llama_attention_forward` (projeções Q, K, V, wo)
- `llama_mlp_forward` (projeções w_gate, w_up, w_down)

**Parâmetros:**
- `weights`: Matriz Q4_0 `[M, N]`, N deve ser múltiplo de 32
- `input`: Vetor F32 `[N]`, 32-byte aligned
- `output`: Vetor F32 `[M]`, 32-byte aligned

**Algoritmo:**
1. Para cada linha i de 0 até M-1:
   - Processa blocos de 32 elementos usando `q_process_block_avx2()`
   - Desquantiza Q4_0 on-the-fly (em registradores)
   - Calcula dot product usando FMA AVX2
   - Reduz horizontalmente para escalar
   - Armazena em `output[i]`

**Função Interna:** `q_process_block_avx2()`
- **Localização:** `src/ops/avx2/matmul.c:35`
- **Parâmetros:** `block` (Q4_0 block), `input_ptr` (32 floats), `acc` (acumulador), `low_mask`
- **Algoritmo:**
  1. Carrega scale do bloco
  2. Carrega dados quantizados (16 bytes)
  3. Separa nibbles (low/high)
  4. Intercala para restaurar ordem
  5. Processa 4 batches de 8 elementos usando FMA
  6. Retorna acumulador atualizado

**Dados:**
- **Entrada:** `weights->data[M*blocks_per_row]` (Q4_0), `input[N]`
- **Saída:** `output[M]`

---

### 5.3 `q_matmul_f32_avx2`

**Localização:** `src/ops/avx2/matmul_fp32.c:54`

**Chamada por:**
- `llama_attention_forward` (Q @ K^T, probs @ V)
- `llama_forward` (LM Head: last_token @ output^T)

**Parâmetros:**
- `A`: Matriz F32 `[M, K]`, 32-byte aligned
- `B`: Matriz F32 `[K, N]`, 32-byte aligned
- `C`: Matriz F32 `[M, N]` (output), 32-byte aligned
- `ctx`: Contexto (para alocar B transposto)

**Algoritmo:**
1. Valida dimensões e tipos
2. Aloca buffer temporário para B^T `[N, K]` na arena
3. Transpõe B usando `transpose_blocked()` (se necessário)
4. Para cada linha i de 0 até M-1:
   - Para cada coluna j de 0 até N-1:
     - Calcula dot product: `C[i,j] = sum(A[i,k] * B_T[j,k])` usando AVX2
     - Processa blocos de 32 elementos
     - Reduz horizontalmente para escalar

**Função Interna:** `transpose_blocked()`
- **Localização:** `src/ops/avx2/matmul_fp32.c:29`
- **Parâmetros:** `src`, `dst`, `rows`, `cols`, `src_stride`, `dst_stride`
- **Algoritmo:** Transpõe matriz em blocos de 32x32 para cache efficiency

**Dados:**
- **Entrada:** `A->data[M, K]`, `B->data[K, N]`
- **Temporário:** `B_T_data[N, K]` (B transposto)
- **Saída:** `C->data[M, N]`

---

### 5.4 `q_rope_f32_avx2`

**Localização:** `src/ops/avx2/rope.c:23`

**Chamada por:** `llama_attention_forward` (aplicação de RoPE em Q e K)

**Parâmetros:**
- `x`: Input `[N]`, N deve ser par e múltiplo de 8, 32-byte aligned
- `cos`: Cos values `[N/2]`, 32-byte aligned
- `sin`: Sin values `[N/2]`, 32-byte aligned
- `output`: Output `[N]`, 32-byte aligned
- `N`: Tamanho (deve ser par e múltiplo de 8)

**Algoritmo:**
1. Processa pares (x[2i], x[2i+1]) usando AVX2
2. Para cada par:
   - `output[2i] = x[2i] * cos[i] - x[2i+1] * sin[i]`
   - `output[2i+1] = x[2i] * sin[i] + x[2i+1] * cos[i]`

**Dados:**
- **Entrada:** `x[N]`, `cos[N/2]`, `sin[N/2]`
- **Saída:** `output[N]`

---

### 5.5 `q_causal_mask_f32_avx2`

**Localização:** `src/ops/avx2/causal_mask_fp32.c:16`

**Chamada por:** `llama_attention_forward` (antes do softmax)

**Parâmetros:**
- `scores`: Matriz `[seq_len, seq_len]` (modificado in-place), 32-byte aligned
- `mask_value`: Valor para posições mascaradas (geralmente -1e9f)

**Algoritmo:**
1. Para cada linha i de 0 até seq_len-1:
   - Para cada coluna j de i+1 até seq_len-1:
     - `scores[i, j] = mask_value` (mascara posições futuras)

**Dados:**
- **Entrada/Saída:** `scores->data[seq_len, seq_len]` (modificado in-place)

---

### 5.6 `q_softmax_f32_avx2`

**Localização:** `src/ops/avx2/softmax.c:17`

**Chamada por:** `llama_attention_forward` (por linha da matriz de scores)

**Parâmetros:**
- `x`: Input `[N]`, 32-byte aligned
- `output`: Output `[N]`, 32-byte aligned
- `N`: Tamanho (qualquer tamanho, com tail handling)

**Algoritmo:**
1. Encontra máximo usando AVX2 + `horizontal_max_avx()`
2. Calcula `exp(x - max)` usando `exp_approx_avx()` (vectorizado + tail escalar)
3. Calcula soma usando AVX2 + `horizontal_sum_avx()`
4. Normaliza: `output = exp / sum`

**Funções Auxiliares:**
- `horizontal_max_avx()`: Reduz vetor AVX2 para máximo escalar
- `horizontal_sum_avx()`: Reduz vetor AVX2 para soma escalar
- `exp_approx_avx()`: Aproximação polinomial de exp usando Horner's method

**Dados:**
- **Entrada:** `x[N]`
- **Saída:** `output[N]` (probabilidades normalizadas, soma = 1.0)

---

### 5.7 `q_silu_f32_avx2`

**Localização:** `src/ops/avx2/silu.c:17`

**Chamada por:** `llama_mlp_forward` (ativação SwiGLU)

**Parâmetros:**
- `x`: Input `[N]`, 32-byte aligned (se N >= 8)
- `output`: Output `[N]`, 32-byte aligned (se N >= 8)
- `N`: Tamanho (qualquer tamanho, com tail handling)

**Algoritmo:**
1. Para elementos vectorizados (N >= 8):
   - Calcula `exp(-x)` usando `exp_approx_avx()`
   - Calcula `sigmoid = 1 / (1 + exp(-x))`
   - Calcula `silu = x * sigmoid` usando AVX2
2. Para tail (N % 8 != 0):
   - Usa implementação escalar com `expf()`

**Dados:**
- **Entrada:** `x[N]`
- **Saída:** `output[N]`

---

### 5.8 `q_add_f32_avx2`

**Localização:** `src/ops/avx2/add_fp32.c:17`

**Chamada por:**
- `llama_layer_forward` (residual connections)

**Parâmetros:**
- `a`: Tensor `[N]`, 32-byte aligned, contíguo
- `b`: Tensor `[N]`, 32-byte aligned, contíguo
- `output`: Tensor `[N]`, 32-byte aligned, contíguo (pode alias a ou b)

**Algoritmo:**
1. Processa 32 elementos por iteração usando AVX2 (4x unrolling)
2. `output[i] = a[i] + b[i]` para cada elemento
3. Tail handling para elementos restantes

**Dados:**
- **Entrada:** `a->data[N]`, `b->data[N]`
- **Saída:** `output->data[N]`

---

### 5.9 `q_mul_f32_avx2`

**Localização:** `src/ops/avx2/mul_fp32.c:15`

**Chamada por:** `llama_mlp_forward` (gate * up)

**Parâmetros:**
- `a`: Tensor `[N]`, 32-byte aligned, contíguo
- `b`: Tensor `[N]`, 32-byte aligned, contíguo
- `output`: Tensor `[N]`, 32-byte aligned, contíguo (pode alias a ou b)

**Algoritmo:**
1. Processa 32 elementos por iteração usando AVX2 (4x unrolling)
2. `output[i] = a[i] * b[i]` para cada elemento
3. Tail handling para elementos restantes

**Dados:**
- **Entrada:** `a->data[N]`, `b->data[N]`
- **Saída:** `output->data[N]`

---

## 6. FUNÇÕES HELPER

### 6.1 `token_embedding_lookup`

**Localização:** `src/models/model.c:640`

**Chamada por:** `llama_forward`

**Parâmetros:**
- `token_embd`: Tensor de embeddings `[vocab_size, dim]`
- `tokens`: Array de token IDs `[seq_len]`
- `seq_len`: Número de tokens
- `output`: Buffer de saída `[seq_len, dim]`

**Algoritmo:**
1. Para cada token i:
   - Valida `tokens[i] < vocab_size`
   - Copia `token_embd->data[tokens[i] * dim : (tokens[i] + 1) * dim]` para `output[i * dim : (i + 1) * dim]`

**Dados:**
- **Entrada:** `token_embd->data[vocab_size, dim]`, `tokens[seq_len]`
- **Saída:** `output[seq_len, dim]`

---

### 6.2 `get_kv_cache_ptr`

**Localização:** `src/models/model.c:558`

**Chamada por:** `llama_attention_forward` (atualização do KV cache)

**Parâmetros:**
- `ctx`: Contexto (contém kv_buffer)
- `config`: Configuração (dimensões)
- `layer_idx`: Índice da camada
- `kv_head_idx`: Índice do KV head
- `pos`: Posição na sequência
- `is_key`: true para K, false para V

**Algoritmo:**
1. Calcula offset no KV buffer:
   - `layer_offset = layer_idx * (n_kv_heads * max_seq_len * head_dim * 2)`
   - `head_offset = kv_head_idx * (max_seq_len * head_dim * 2)`
   - `pos_offset = pos * (head_dim * 2)`
   - `kv_offset = is_key ? 0 : head_dim`
2. Retorna ponteiro: `kv_buffer + layer_offset + head_offset + pos_offset + kv_offset`

**Retorno:** Ponteiro para `float[head_dim]` no KV cache

---

### 6.3 `generate_rope_cos_sin`

**Localização:** `src/models/model.c:604`

**Chamada por:** `llama_attention_forward` (geração de tabelas RoPE)

**Parâmetros:**
- `rope_theta`: Base theta para RoPE
- `head_dim`: Dimensão do head (deve ser par)
- `pos`: Posição absoluta na sequência
- `cos_buf`: Buffer de saída `[head_dim]`
- `sin_buf`: Buffer de saída `[head_dim]`

**Algoritmo:**
1. Para cada par i de 0 até head_dim/2-1:
   - Calcula frequência: `freq_exp = -2.0f * i / head_dim`
   - Calcula theta: `theta = rope_theta^freq_exp * pos`
   - Calcula cos/sin: `c = cosf(theta)`, `s = sinf(theta)`
   - Duplica para layout AVX2: `cos_buf[2i] = cos_buf[2i+1] = c`, `sin_buf[2i] = sin_buf[2i+1] = s`

**Dados:**
- **Entrada:** Parâmetros escalares
- **Saída:** `cos_buf[head_dim]`, `sin_buf[head_dim]`

---

## 7. FUNÇÕES AUXILIARES AVX2 (inline)

### 7.1 `exp_approx_avx`

**Localização:** `src/ops/avx2/avx_math.h:10`

**Chamada por:**
- `q_softmax_f32_avx2`
- `q_silu_f32_avx2`

**Parâmetros:**
- `x`: Vetor AVX2 de 8 floats

**Algoritmo:**
1. Clamp para [-5, 5]
2. Aproximação polinomial usando Horner's method:
   - `result = ((((c5*x + c4)*x + c3)*x + c2)*x + c1)*x + c0`
3. Garante não-negativo
4. Aplica máscaras para valores extremos

**Retorno:** Vetor AVX2 `__m256` com 8 valores de exp aproximados

---

### 7.2 `horizontal_sum_avx`

**Localização:** `src/ops/avx2/avx_math.h:64`

**Chamada por:**
- `q_softmax_f32_avx2`
- `q_gemv_q4_f32_avx2` (redução final)

**Parâmetros:**
- `vec`: Vetor AVX2 `__m256` com 8 floats

**Algoritmo:**
1. Extrai low e high (128-bit cada)
2. Soma low + high
3. Reduz horizontalmente para escalar

**Retorno:** Soma escalar dos 8 elementos

---

### 7.3 `horizontal_max_avx`

**Localização:** `src/ops/avx2/avx_math.h:76`

**Chamada por:** `q_softmax_f32_avx2`

**Parâmetros:**
- `vec`: Vetor AVX2 `__m256` com 8 floats

**Algoritmo:**
1. Extrai low e high (128-bit cada)
2. Máximo de low e high
3. Reduz horizontalmente para escalar

**Retorno:** Máximo escalar dos 8 elementos

---

## 8. RESUMO DO FLUXO DE DADOS

### 8.1 Fluxo Principal (High-Level)

```
tokens[seq_len]
  └─> token_embedding_lookup()
      └─> x[seq_len, dim]

x[seq_len, dim]
  └─> [Loop n_layers vezes]
      └─> llama_layer_forward()
          └─> Attention Block
              └─> attn_out[seq_len, dim]
          └─> Residual: x + attn_out
          └─> MLP Block
              └─> mlp_out[seq_len, dim]
          └─> Residual: x + mlp_out
      └─> x = layer_out (swap buffers)

x[seq_len, dim]
  └─> RMSNorm Final
      └─> x_final[seq_len, dim]

x_final[seq_len, dim]
  └─> LM Head (last token only)
      └─> logits[1, vocab_size]
```

### 8.2 Fluxo de Dados por Operação

**Token Embedding:**
- `token_embd[vocab_size, dim]` + `tokens[seq_len]` → `x[seq_len, dim]`

**Attention (por camada):**
- `x[seq_len, dim]` → RMSNorm → `x_norm[seq_len, dim]`
- `x_norm` → Q/K/V projections → `q_buf`, `k_buf`, `v_buf`
- `q_buf`, `k_buf` → RoPE → `q_rope_buf`, `k_rope_buf`
- `k_rope_buf`, `v_buf` → KV Cache (persistente)
- `q_rope_buf`, `k_rope_buf` → Attention scores → `scores[seq_len, seq_len]`
- `scores` → Causal mask → `scores` (modificado)
- `scores` → Softmax → `probs[seq_len, seq_len]`
- `probs`, `v_buf` → Attention output → `attn_head[seq_len, head_dim]` (por head)
- `attn_head` (concatenação) → Output projection → `attn_out[seq_len, dim]`

**MLP (por camada):**
- `x_norm[seq_len, dim]` → Gate/Up projections → `gate_buf`, `up_buf[seq_len, hidden_dim]`
- `gate_buf` → SiLU → `gate_silu[seq_len, hidden_dim]`
- `gate_silu`, `up_buf` → Mul → `mul_buf[seq_len, hidden_dim]`
- `mul_buf` → Down projection → `mlp_out[seq_len, dim]`

**Residual Connections:**
- Attention: `x[seq_len, dim]` + `attn_out[seq_len, dim]` → `x_norm[seq_len, dim]`
- MLP: `x_norm[seq_len, dim]` + `mlp_out[seq_len, dim]` → `output[seq_len, dim]`

---

## 9. MEMÓRIA E ALOCAÇÕES

### 9.1 Estruturas Persistentes (alocadas em `llama_build_graph`)
- Modelo: `llama_model` (na arena)
- Camadas: `llama_layer[]` (na arena)
- Tensor views: `q_tensor*` para todos os pesos (na arena, apontam para mmap)
- Pesos Q4_0: mmap'd do arquivo `.qorus` (não alocados, zero-copy)

### 9.2 Buffers Temporários (alocados na arena durante forward pass)
- `x`: Embeddings `[seq_len, dim]`
- `layer_out`: Saída de camada `[seq_len, dim]`
- `x_final`: Normalização final `[seq_len, dim]`
- Por camada (Attention):
  - `x_norm`, `q_buf`, `k_buf`, `v_buf`, `attn_out`: `[seq_len, dim]` cada
  - `cos_buf`, `sin_buf`: `[head_dim]` cada
  - `q_rope_buf`: `[seq_len, dim]`
  - `k_rope_buf`: `[seq_len, n_kv_heads * head_dim]`
  - `scores_buf`: `[seq_len, seq_len]`
  - `q_heads`: `[n_heads, seq_len, head_dim]`
  - `k_heads`, `v_heads`: `[n_kv_heads, seq_len, head_dim]` cada
  - `attn_head_buf`: `[seq_len, head_dim]`
  - `k_t_buf`: `[head_dim, seq_len]`
- Por camada (MLP):
  - `gate_buf`, `up_buf`, `mul_buf`, `gate_silu`: `[seq_len, hidden_dim]` cada
- Por MatMul:
  - `B_T_data`: `[N, K]` (temporário para transposição)

### 9.3 KV Cache (alocado separadamente, persistente)
- `kv_buffer`: `[n_layers, n_kv_heads, max_seq_len, head_dim, 2]` (K e V)
- Acessado via `get_kv_cache_ptr()`
- Atualizado durante forward pass, persiste entre chamadas

---

## 10. NOTAS IMPORTANTES

1. **Zero-Copy**: Pesos Q4_0 são mmap'd diretamente do arquivo, não copiados para memória heap.

2. **Arena Allocation**: Todos os buffers temporários são alocados na arena (scratch buffer), que é reutilizada entre chamadas. Estruturas do modelo (tensor views) ficam no início da arena e não são resetadas.

3. **Alinhamento**: Todos os buffers usados com AVX2 devem estar alinhados a 32 bytes (garantido por `Q_ALIGN_SIZE` e `q_arena_alloc`).

4. **GQA (Grouped Query Attention)**: Múltiplos query heads compartilham um único KV head. O mapeamento é: `kv_head_idx = qh / (n_heads / n_kv_heads)`.

5. **RoPE**: Aplicado por token e por head, usando cos/sin gerados dinamicamente para cada posição.

6. **Causal Masking**: Aplicado antes do softmax, mascara posições futuras (j > i) com valor grande negativo.

7. **Tail Handling**: Kernels AVX2 processam elementos em múltiplos de 8 (ou 32), com fallback escalar para elementos restantes.

8. **Quantização Q4_0**: Pesos são desquantizados on-the-fly durante `q_gemv_q4_f32_avx2`, nunca armazenados em FP32.

---

**Última atualização:** Baseado no código atual do Qorus-IA v2.0

