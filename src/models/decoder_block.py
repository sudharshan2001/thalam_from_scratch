import torch
import torch.nn as nn

from src.models.attention import QwenGQAWithFlashRoPE
from src.models.swiglu import SwiGLU


class DecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 896,
        num_q_heads: int = 14,
        num_kv_heads: int = 2,
        head_dim: int = 64,
        intermediate_size: int = 4864,
    ):
        super().__init__()

        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attn  = QwenGQAWithFlashRoPE(hidden_size, num_q_heads, num_kv_heads, head_dim)

        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.mlp   = SwiGLU(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention + residual
        x = x + self.attn(self.norm1(x))
        # Pre-norm MLP + residual
        x = x + self.mlp(self.norm2(x))
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 16, 896, device=device, dtype=torch.bfloat16)
    block = DecoderBlock().to(device, dtype=torch.bfloat16)
    out = block(x)
    print(out.shape)  # (2, 16, 896)