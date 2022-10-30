from functools import partial
from typing import Type, Set

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
# from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.t5.modeling_t5 import T5Block

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)

try:
    from torch.distributed.fsdp import MixedPrecision
except ImportError:
    logger.info("MixedPrecision Class need torch>=1.12.0")

"""
Refer to https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/,
and https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html.
"""


# def torch_fsdp_initialize_default(model,
#                                   device,
#                                   cpu_offload=False,
#                                   min_num_params: int = 1e8):
#     my_auto_wrap_policy = partial(default_auto_wrap_policy,
#                                   min_num_params=min_num_params)
#
#     fsdp_model = FullyShardedDataParallel(
#         model,
#         fsdp_auto_wrap_policy=my_auto_wrap_policy,
#         cpu_offload=CPUOffload(offload_params=cpu_offload)
#     )
#
#     if not cpu_offload:
#         fsdp_model.to(device)
#
#     return fsdp_model

def transformer_auto_wrap_policy(
        module: nn.Module,
        recurse: bool,
        unwrapped_params: int,
        transformer_layer_cls: Set[Type[nn.Module]],
) -> bool:
    """
    A convenient auto wrap policy for transformer models. If the submodule
    is an instance of transformer_layer_cls, the submodule will be wrapped
    as a FSDP unit. Otherwise, all the other remainder submodules are wrapped
    by the outermost FSDP unit. Right now, FSDP requires submodules that share
    weights to be wrapped in the same FSDP unit, this auto wrap policy can
    conviniently wrap the shared embeddings into the same FSDP unit for transformer
    models. In the near future, FSDP will support submodules that share weights
    to be wrapped in the separated FSDP units.

    Return if a module should be wrapped during FSDP auto wrapping.

    The first three parameters are required by :func:`_recursive_wrap`.


    Args:
       module (nn.Module):
           The module to be considered in this decision.
       recurse (bool):
           Indicate if this is called to make a decision on whether we
           should recurse down a subgraph of the module structure.
           If False, it means this function is called to make a decision
           on whether we should wrap the said module.
       unwrapped_params (int):
           The number of parameters yet to be wrapped in this module.

       transformer_layer_cls (int):
           Submodules with one of the `transformer_layer_cls` names
           will be wrapped as seperated FSDP units
    """
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap for the leaf node or reminder
        return isinstance(module, tuple(transformer_layer_cls))


def transformer_fsdp_initialize_torch1_11(model,
                                          device,
                                          cpu_offload=False):
    wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={T5Block})

    fsdp_model = FullyShardedDataParallel(
        model,
        fsdp_auto_wrap_policy=wrap_policy,
        cpu_offload=CPUOffload(offload_params=True) if cpu_offload else None,
    )

    if not cpu_offload:
        fsdp_model.to(device)

    return fsdp_model


def transformer_fsdp_initialize_default(model,
                                        device,
                                        fp16_bfloat16: bool = False,
                                        param_fp16: bool = True,
                                        reduce_fp16: bool = False,
                                        cpu_offload=False):
    wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={T5Block})

    if fp16_bfloat16:
        fp16_dtype = torch.bfloat16
    else:
        fp16_dtype = torch.float16

    param_dtype = fp16_dtype if param_fp16 else None
    reduce_dtype = fp16_dtype if reduce_fp16 else None

    fsdp_model = FullyShardedDataParallel(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype),
        cpu_offload=CPUOffload(offload_params=True) if cpu_offload else None,
    )

    if not cpu_offload:
        fsdp_model.to(device)

    return fsdp_model
