import torch
import torch.nn as nn
from torchtune.modules.transformer import TransformerDecoder

def qwen(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    intermediate_dim: int,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-5,
    rope_base: int = 10000,
    scale_factor: int = 1,
) -> TransformerDecoder:
    """
    创建一个Qwen Transformer解码器
    
    Args:
        vocab_size: 词汇表大小
        num_layers: transformer层数
        num_heads: 注意力头数
        num_kv_heads: KV注意力头数
        embed_dim: 嵌入维度
        max_seq_len: 最大序列长度
        intermediate_dim: FFN中间层维度
        attn_dropout: 注意力dropout率
        norm_eps: Layer normalization的epsilon值
        rope_base: RoPE位置编码的base值
        scale_factor: 缩放因子
    """
    return TransformerDecoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        intermediate_dim=intermediate_dim,
        attn_dropout=attn_dropout,
        norm_eps=norm_eps,
        rope_base=rope_base,
        scale_factor=scale_factor,
    ) 