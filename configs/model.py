from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Dimensions
    hidden_size:       int   = 896
    num_layers:        int   = 24
    num_q_heads:       int   = 14
    num_kv_heads:      int   = 2
    head_dim:          int   = 64
    intermediate_size: int   = 4864

    # Tokenizer / vocab
    vocab_size:        int   = 151936   # Qwen3 vocab
    max_seq_len:       int   = 2048

    # Critical: saves 136M params by sharing embedding & lm_head weights
    tie_word_embeddings: bool = True

    # Regularization
    rms_norm_eps:      float = 1e-6
    rope_base:         float = 10000.0