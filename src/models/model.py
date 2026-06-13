import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from configs.model import ModelConfig
from src.models.decoder_block import DecoderBlock


class ThalamModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            DecoderBlock(
                hidden_size=config.hidden_size,
                num_q_heads=config.num_q_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
            )
            for _ in range(config.num_layers)
        ])

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        # input_ids: (B, S)
        x = self.embedding(input_ids)       # (B, S, hidden_size)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.norm(x)
        if return_hidden:
            return x                        # (B, S, hidden_size)
        return self.lm_head(x)              # (B, S, vocab_size)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    config = ModelConfig()
    model = ThalamModel(config)
    print(f"Parameters: {model.count_parameters() / 1e6:.1f}M")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device, dtype=torch.bfloat16)
    ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
    logits = model(ids)
    print(f"Logits shape: {logits.shape}")  # (2, 16, 151936)