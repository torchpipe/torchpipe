import torch
from typing import Tuple


import torch
from typing import Tuple


class LlamaRotaryEmbedding(torch.nn.Module):
    """Generates rotary embeddings for Llama2's attention mechanism.
    
    Args:
        base (float): Frequency base for RoPE. Default: 10000.0
        dim (int): Embedding dimension (must be even). Default: 128
        device (torch.device): Target device. Default: 'cuda'
    """
    
    def __init__(self, base: float = 10000.0, dim: int = 128, device: torch.device = torch.device('cuda')):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        
        # 核心计算：inv_freq 的维度应为 [1, dim//2, 1]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq.view(1, -1, 1))  # Shape: [1, dim//2, 1]
        
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成 RoPE 的 cos/sin 嵌入
        
        Args:
            x: 输入张量（仅用于获取 dtype 和 device）
            position_ids: 位置ID，形状 [batch_size, seq_len]
            
        Returns:
            Tuple[cos_emb, sin_emb]: 形状均为 [batch_size, seq_len, dim]
        """
        batch_size, seq_len = position_ids.shape
        
        inv_freq_expanded = self.inv_freq.expand(batch_size, -1, -1)
        
        pos_expanded = position_ids.unsqueeze(-1).to(torch.float32)  # Shape: [batch_size, seq_len, 1]
        
        freqs = torch.matmul(inv_freq_expanded, pos_expanded.transpose(1, 2))  # [batch_size, dim//2, seq_len]
        freqs = freqs.transpose(1, 2)  # [batch_size, seq_len, dim//2]
        
        emb = torch.cat([freqs, freqs], dim=-1)  # [batch_size, seq_len, dim]
        
        cos_emb = emb.cos().to(x.dtype)
        sin_emb = emb.sin().to(x.dtype)
        
        return cos_emb, sin_emb
    
rope = LlamaRotaryEmbedding(base=10000.0, dim=128, device=torch.device('cuda'))

def generate_cos_sin_attention_mask(
    sequence_length: int,
    target_length: int,
    is_prefill: bool = True,
    device: torch.device = torch.device('cuda')
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成符合 Llama2 需求的三个关键张量
    
    Args:
        sequence_length: 当前序列长度
        target_length: 目标序列长度
        is_prefill: 是否为预填充阶段
        device: 输出设备
        
    Returns:
        Tuple[cos, sin, mask]:
            - cos: [1, target_length, 128]
            - sin: [1, target_length, 128]
            - mask: [1, 1, sequence_length, target_length]
    """
    
    if is_prefill:
        assert  sequence_length == target_length, "For prefill, sequence_length must equal target_length right now"
        position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)  # [1, seq_len]
    else:
        assert sequence_length == 1, "For decode, sequence_length must be 1 right now"
        position_ids = torch.arange(target_length - sequence_length, target_length, device=device).unsqueeze(0)
    
    # 生成注意力掩码
    if is_prefill:
        causal_mask = torch.full((1, 1, sequence_length, target_length), 
                               torch.finfo(torch.float16).min, 
                               dtype=torch.float16, device=device)
        if sequence_length > 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
    else:
        causal_mask = torch.zeros((1, 1, sequence_length, target_length), 
                                dtype=torch.float16, device=device)
    
    cos, sin = rope(causal_mask, position_ids)
    
    return cos, sin, causal_mask

# Example usage:
if __name__ == "__main__":
    # Prefill example
    cos_pf, sin_pf, mask_pf = generate_cos_sin_attention_mask(
        sequence_length=5, target_length=5, is_prefill=True
    )
    
    # Decode example
    cos_decode, sin_decode, mask_decode = generate_cos_sin_attention_mask(
        sequence_length=1, target_length=6, is_prefill=False
    )
    
    print(f"Prefill shapes: cos={cos_pf.shape}, mask={mask_pf.shape}")
    print(f"Decode shapes: cos={cos_decode.shape}, mask={mask_decode.shape}")
    print(f'mask_decode: {mask_decode}')
