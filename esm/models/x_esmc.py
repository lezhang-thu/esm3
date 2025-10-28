import attr
import torch
import torch.nn as nn
from attr import dataclass

try:
    from flash_attn.bert_padding import pad_input, unpad_input  # type:ignore

    is_flash_attn_available = True
except ImportError:
    pad_input = None
    unpad_input = None
    is_flash_attn_available = False

from esm.layers.regression_head import RegressionHead
from esm.layers.transformer_stack import TransformerStack
from esm.sdk.api import (
    ESMCInferenceClient,
    ESMProtein,
    ESMProteinTensor,
)
from esm.tokenization import EsmSequenceTokenizer
from esm.utils import encoding
from esm.utils.constants.models import ESMC_600M
from esm.utils.decoding import decode_sequence
from esm.utils.misc import stack_variable_length_tensors

#class TwoHeadMu(nn.Module):
#
#    def __init__(self, d_model: int):
#        super().__init__()
#        # Shared trunk before splitting
#        self.shared = nn.Sequential(
#            nn.LayerNorm(d_model, bias=False),
#            nn.Linear(d_model, 3 * d_model),
#        )
#
#        # Separate output heads for IC50 and IC80
#        self.mu50_out = nn.Sequential(
#            nn.PReLU(num_parameters=1, init=0.25),
#            nn.Linear(d_model, 1),
#        )
#        self.mu80_out = nn.Sequential(
#            nn.PReLU(num_parameters=1, init=0.25),
#            nn.Linear(d_model, 1),
#        )
#        self.mu_ID50 = nn.Sequential(
#            nn.PReLU(num_parameters=1, init=0.25),
#            nn.Linear(d_model, 1),
#        )
#
#    def forward(self, x: torch.Tensor) -> torch.Tensor:
#        """
#        Args:
#            x: [B, d_model] input (e.g., CLS embedding)
#        Returns:
#            mu: [B, 2] tensor, where mu[:, 0] = IC50, mu[:, 1] = IC80
#        """
#        # Shared transformation
#        t = self.shared(x)  # [B, 2 * d_model]
#
#        # Split into two [B, d_model] chunks
#        mu50_h, mu80_h, muID50_h = torch.chunk(t, 3, dim=-1)
#
#        # Apply independent output heads
#        mu50 = self.mu50_out(mu50_h)  # [B, 1]
#        mu80 = self.mu80_out(mu80_h)  # [B, 1]
#        muID50 = self.mu_ID50(muID50_h)  # [B, 1]
#
#        # Concatenate results → [B, 2]
#        mu = torch.cat([mu50, mu80, muID50], dim=-1)
#        return mu


class ESMC(nn.Module, ESMCInferenceClient):
    """
    ESMC model implementation.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads in the transformer layers.
        n_layers (int): The number of transformer layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        tokenizer: EsmSequenceTokenizer,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.embed = nn.Embedding(64, d_model)

        self._use_flash_attn = is_flash_attn_available and use_flash_attn
        self.transformer = TransformerStack(
            d_model,
            n_heads,
            None,
            n_layers,
            n_layers_geom=0,
            use_flash_attn=self._use_flash_attn,
        )

        self.sequence_head = RegressionHead(d_model, 64)
        self.tokenizer = tokenizer

        #self.mu = TwoHeadMu(d_model)
        self.mu = nn.Sequential(
            nn.LayerNorm(d_model, bias=False),
            nn.Linear(d_model, 3),
        )
        #self.logsigma = nn.Sequential(
        #    nn.LayerNorm(d_model, bias=False),
        #    nn.Linear(d_model, 2),
        #)

    @classmethod
    def from_pretrained(cls,
                        model_name: str = ESMC_600M,
                        device: torch.device | None = None):
        from esm.pretrained import load_local_model

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        model = load_local_model(model_name, device=device)
        if device.type != "cpu":
            model = model.to(torch.bfloat16)
        assert isinstance(model, ESMC)
        return model

    @property
    def device(self):
        return next(self.parameters()).device

    def _tokenize(self, sequence: list[str]) -> torch.Tensor:
        pad = self.tokenizer.pad_token_id
        assert pad is not None
        return stack_variable_length_tensors(
            [
                encoding.tokenize_sequence(
                    x, self.tokenizer, add_special_tokens=True)
                for x in sequence
            ],
            constant_value=pad,
        ).to(next(self.parameters()).device)

    def forward(
        self,
        sequence_tokens: torch.Tensor | None = None,
    ):
        # For EMSC, a boolean mask is created in place of sequence_id if not specified.
        sequence_id = sequence_tokens != self.tokenizer.pad_token_id

        x = self.embed(sequence_tokens)

        B, L = x.shape[:2]

        # If sequence_id looks like a mask.
        if self._use_flash_attn:
            assert (
                sequence_id.dtype == torch.bool
            ), "sequence_id must be a boolean mask if Flash Attention is used"
            assert sequence_id.shape == (B, L)
            assert unpad_input is not None
            x, indices, *_ = unpad_input(  # type: ignore
                x, sequence_id)
        else:
            indices = None

        x, _, hiddens = self.transformer(x, sequence_id=sequence_id)

        if self._use_flash_attn:
            assert indices is not None
            assert pad_input is not None
            x = pad_input(x, indices, B, L)  # Back to [B, L, D]
            hiddens = [
                # Back to [[B, L, D], ...]
                pad_input(h, indices, B, L) for h in hiddens
            ]

        # Stack hidden states into a [n_layers, B, L, D] matrix.
        hiddens = torch.stack(hiddens, dim=0)  # type: ignore
        return hiddens

    def encode(self, input: ESMProtein) -> ESMProteinTensor:
        input = attr.evolve(input)  # Make a copy
        sequence_tokens = None

        if input.sequence is not None:
            sequence_tokens = self._tokenize(input.sequence)
        return ESMProteinTensor(sequence=sequence_tokens).to(self.device)

    def predict(
        self,
        seq: ESMProteinTensor,
    ):
        hiddens = self.forward(sequence_tokens=seq.sequence)
        # [n_layers, B, L, D]
        cls_vec = hiddens[-1, :, 0, :]
        #return self.mu(cls_vec), self.logsigma(cls_vec)
        return self.mu(cls_vec), None
