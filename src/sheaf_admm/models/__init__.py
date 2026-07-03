"""Neural building blocks: shared layers, per-agent encoders/decoders, MPNN.

The encoder maps each agent's local view to its consensus vector ``h`` and the
local-objective parameters consumed by the x-solvers; the decoder maps final
agent states back to logits. Both are **shared across agents** (the parent
flattens the ``N`` agent axis into the batch). The MPNN module is the recurrent
message-passing ablation of the ADMM coordinator and reuses the same
encoder/decoder.
"""

from __future__ import annotations

from .config import ModelConfig, model_config_from_dict
from .decoder import (
    ClassificationDecoder,
    ConcatMLPDecoderV2,
    SudokuDecoder,
    create_decoder,
)
from .encoder import MLPEncoder, MLPEncoderV2, SudokuEncoder, create_encoder
from .layers import (
    CommHead,
    GeLUMLP,
    MLPBlock,
    MLPMixerBlock,
    RMSNorm,
    SwiGLU,
    rms_norm,
)
from .mpnn import DirectionalGGNN, DirectionalGGNNCell, GraphClassificationHead
from .mpnn_model import MPNNModel
from .sheaf_model import SheafADMMModel

__all__ = [
    # config + full models
    "ModelConfig",
    "model_config_from_dict",
    "SheafADMMModel",
    "MPNNModel",
    # shared layers
    "RMSNorm",
    "rms_norm",
    "MLPBlock",
    "MLPMixerBlock",
    "SwiGLU",
    "GeLUMLP",
    "CommHead",
    # encoders
    "MLPEncoder",
    "MLPEncoderV2",
    "SudokuEncoder",
    "create_encoder",
    # decoders
    "ConcatMLPDecoderV2",
    "SudokuDecoder",
    "ClassificationDecoder",
    "create_decoder",
    # mpnn baseline
    "DirectionalGGNN",
    "DirectionalGGNNCell",
    "GraphClassificationHead",
]
