# Copyright (c) OpenMMLab. All rights reserved.
"""ROCm backend for AMD GPUs."""

from .op_backend import (
    ROCmOpsBackend,
    is_rocm_available,
    get_rocm_arch,
    get_rocm_device_info,
)

__all__ = [
    'ROCmOpsBackend',
    'is_rocm_available',
    'get_rocm_arch',
    'get_rocm_device_info',
]
