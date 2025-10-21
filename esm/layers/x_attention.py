import math
import functools

import einops
import torch
import torch.nn.functional as F
from torch import nn

from esm.layers.rotary import RotaryEmbedding, TritonRotaryEmbedding

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func  # type: ignore
except (ImportError, RuntimeError):
    flash_attn_varlen_qkvpacked_func = None


class LoRALinearQKV(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
    ):
        super().__init__()
        assert out_dim == in_dim * 3

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = bias

        # Setup weight and bias
        linear = nn.Linear(in_features=in_dim,
                           out_features=out_dim,
                           bias=self.use_bias)
        weight = linear.weight
        bias = linear.bias if self.use_bias else None
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter(
            "bias",
            nn.Parameter(bias) if bias is not None else None)

        # https://github.com/meta-pytorch/torchtune/blob/main/recipes/configs/qwen3/0.6B_lora_single_device.yaml
        # lora_attn_modules: ['q_proj', 'v_proj', 'output_proj']
        # apply_lora_to_mlp: True
        # lora_rank: 32  # higher increases accuracy and memory
        # lora_alpha: 64  # usually alpha=2*rank
        # lora_dropout: 0.0
        self.rank = 32
        self.alpha = 64

        def make_lora_pair(in_dim, rank):
            a = nn.Linear(in_dim, rank, bias=False)
            b = nn.Linear(rank, in_dim, bias=False)
            return a, b

        self.q_lora_a, self.q_lora_b = make_lora_pair(in_dim, self.rank)
        self.v_lora_a, self.v_lora_b = make_lora_pair(in_dim, self.rank)
        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119
        _lora_a_init_params(self.q_lora_a)
        _lora_b_init_params(self.q_lora_b)

        _lora_a_init_params(self.v_lora_a)
        _lora_b_init_params(self.v_lora_b)

    def adapter_params(self) -> list[str]:
        """
        Return a list of strings corresponding to the names of the ``nn.Parameter`` s in
        the model coming from the adapter.

        For LoRA this means lora_a.weight and lora_b.weight.
        """
        # NOTE: this function has to be updated if the names of "lora_a" and "lora_b"
        # in this module change.
        adapter_params = [
            "q_lora_a.weight",
            "q_lora_b.weight",
            "v_lora_a.weight",
            "v_lora_b.weight",
        ]
        return adapter_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            torch.Tensor: output tensor with shape ``(..., out_dim)``

        """
        out = F.linear(x, self.weight, self.bias)
        q, k, v = out.split(self.in_dim, dim=-1)

        q_lora_out = self.q_lora_a(x)
        q_lora_out = (self.alpha / self.rank) * self.q_lora_b(q_lora_out)

        v_lora_out = self.v_lora_a(x)
        v_lora_out = (self.alpha / self.rank) * self.v_lora_b(v_lora_out)

        return torch.cat([q + q_lora_out, k, v + v_lora_out], -1)


def _lora_a_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA A weight to Kaiming uniform.
    """
    nn.init.kaiming_uniform_(x.weight, a=math.sqrt(5))


def _lora_b_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA B weight to zeros.
    """
    nn.init.zeros_(x.weight)


class x_MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 bias: bool = False,
                 qk_layernorm: bool = True):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            LoRALinearQKV(d_model, d_model * 3, bias=bias))
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.rotary = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x, seq_id):
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD, key_BLD = (
            self.q_ln(query_BLD).to(query_BLD.dtype),
            self.k_ln(key_BLD).to(query_BLD.dtype),
        )
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

        reshaper = functools.partial(einops.rearrange,
                                     pattern="b s (h d) -> b h s d",
                                     h=self.n_heads)

        query_BHLD, key_BHLD, value_BHLD = map(reshaper,
                                               (query_BLD, key_BLD, value_BLD))

        if seq_id is not None:
            # Where True, enable participation in attention.
            mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            mask_BHLL = mask_BLL.unsqueeze(1)

            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, mask_BHLL)
        else:
            # Shortcut, if we don't use attention biases then torch
            # will autoselect flashattention as the implementation
            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD)

        context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")

        return self.out_proj(context_BLD)


class x_FlashMultiHeadAttention(x_MultiHeadAttention):

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 bias: bool = False,
                 qk_layernorm: bool = True):
        super().__init__(d_model=d_model,
                         n_heads=n_heads,
                         bias=bias,
                         qk_layernorm=qk_layernorm)

        # Flash attention rotary.
        self.rotary = TritonRotaryEmbedding(d_model // n_heads)

    def forward(self, x, seq_id):
        assert seq_id.dtype == torch.bool

        seqlens = seq_id.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32),
                           (1, 0))
        max_seqlen = seqlens.max().item()

        qkv_ND3 = self.layernorm_qkv(x)

        query_ND, key_ND, value_ND = torch.chunk(qkv_ND3, 3, dim=-1)
        query_ND, key_ND = (
            self.q_ln(query_ND).to(query_ND.dtype),
            self.k_ln(key_ND).to(query_ND.dtype),
        )

        qkv_N3D = torch.stack([query_ND, key_ND, value_ND], dim=1)
        qkv_N3HD = einops.rearrange(qkv_N3D,
                                    pattern="n a (h d) -> n a h d",
                                    h=self.n_heads)
        qkv_N3HD = self.rotary(qkv_N3HD, cu_seqlens, max_seqlen)

        context_NHD = flash_attn_varlen_qkvpacked_func(  # type: ignore
            qkv_N3HD,
            cu_seqlens,
            max_seqlen,
            softmax_scale=self.d_head**-0.5)
        context_ND = einops.rearrange(context_NHD, "n h d -> n (h d)")

        return self.out_proj(context_ND)
