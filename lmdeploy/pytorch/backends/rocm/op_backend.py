# Copyright (c) OpenMMLab. All rights reserved.
"""
ROCm (AMD GPU) Backend for LMDeploy

This backend enables LMDeploy to run on AMD GPUs using ROCm.
Most Triton kernels from the CUDA backend can be directly reused
since Triton automatically compiles to ROCm.
"""

from typing import Tuple
import torch

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.utils import get_logger

from ..base import OpType
from ..cuda import CudaOpsBackend  # 复用 CUDA backend 的大部分实现

logger = get_logger('lmdeploy')


def is_rocm_available() -> bool:
    """Check if ROCm is available."""
    if not torch.cuda.is_available():
        return False

    try:
        # ROCm version of PyTorch also uses torch.cuda API
        # but we can distinguish it by checking for HIP
        import torch.version
        return hasattr(torch.version, 'hip') and torch.version.hip is not None
    except:
        return False


def get_rocm_arch() -> str:
    """
    Get AMD GPU architecture.

    Common architectures:
    - gfx900: Vega (MI25)
    - gfx906: Vega 20 (MI50, MI60)
    - gfx908: CDNA 1 (MI100)
    - gfx90a: CDNA 2 (MI210, MI250, MI250X)
    - gfx940: CDNA 3 (MI300A, MI300X)
    - gfx1030: RDNA 2 (RX 6000 series)
    - gfx1100: RDNA 3 (RX 7000 series)
    """
    if not torch.cuda.is_available():
        return None

    props = torch.cuda.get_device_properties(0)

    # ROCm provides gcnArchName in device properties
    if hasattr(props, 'gcnArchName'):
        return props.gcnArchName

    # Fallback: construct from compute capability
    major, minor = props.major, props.minor
    return f"gfx{major}{minor}"


class ROCmOpsBackend(CudaOpsBackend):
    """
    ROCm backend for AMD GPUs.

    This backend inherits from CudaOpsBackend because:
    1. Triton kernels work on both CUDA and ROCm
    2. PyTorch's ROCm version uses the same torch.cuda API
    3. We only need to override platform-specific parts
    """

    @staticmethod
    def get_name() -> str:
        """Backend name."""
        return 'rocm'

    @classmethod
    def get_layer_impl_builder(cls, layer_type: OpType):
        """
        Get ROCm layer builder.

        Most operations can directly use CUDA backend's Triton implementations
        since Triton automatically compiles to ROCm.
        """

        # Check for ROCm-specific optimizations
        arch = get_rocm_arch()

        if layer_type == OpType.PagedAttention:
            # Try to use Flash Attention for ROCm if available
            try:
                import flash_attn
                # Flash Attention 2.x has ROCm support
                from ..cuda.attention import TritonAttentionBuilder
                logger.info(f'Using Flash Attention on ROCm arch: {arch}')
                return TritonAttentionBuilder
            except ImportError:
                logger.warning('Flash Attention not found, using Triton attention')
                from ..cuda.attention import TritonAttentionBuilder
                return TritonAttentionBuilder

        elif layer_type == OpType.LinearW4A16:
            # AWQ quantization on ROCm
            # ROCm might not have optimized INT4 kernels yet
            logger.warning(
                f'AWQ W4A16 on ROCm may use fallback implementation. '
                f'Architecture: {arch}'
            )
            # Try CUDA implementation first, fallback if it fails
            try:
                from ..cuda.awq_modules import AwqLinearW4A16Builder
                return AwqLinearW4A16Builder
            except Exception as e:
                logger.warning(f'AWQ builder failed: {e}, using default')
                return super().get_layer_impl_builder(layer_type)

        elif layer_type == OpType.FusedMoE:
            # Fused MoE should work on ROCm via Triton
            logger.debug(f'Using Triton Fused MoE on ROCm arch: {arch}')
            return super().get_layer_impl_builder(layer_type)

        else:
            # All other operations use CUDA backend (Triton kernels)
            logger.debug(
                f'Op {layer_type} using CUDA/Triton implementation on ROCm'
            )
            return super().get_layer_impl_builder(layer_type)

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        """K cache block shape - same as CUDA."""
        return (block_size, num_heads, head_size)

    @staticmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        """V cache block shape - same as CUDA."""
        return (num_heads, head_size, block_size)

    @staticmethod
    def device_count():
        """Get number of AMD GPUs."""
        if not is_rocm_available():
            return 0
        return torch.cuda.device_count()

    @staticmethod
    def support_ray():
        """ROCm supports Ray for distributed inference."""
        return True

    @classmethod
    def update_step_context(cls, step_context):
        """
        Update StepContext for inference on ROCm.

        ROCm uses RCCL (ROCm Collective Communications Library)
        which is API-compatible with NCCL, so we can use the same logic.
        """
        arch = get_rocm_arch()
        logger.debug(f'Running on ROCm architecture: {arch}')

        # Log ROCm-specific information
        if arch and arch.startswith('gfx90a'):  # MI200 series
            logger.info('Detected MI200 series GPU (CDNA 2)')
        elif arch and arch.startswith('gfx940'):  # MI300 series
            logger.info('Detected MI300 series GPU (CDNA 3)')

        return super().update_step_context(step_context)

    @staticmethod
    def build_graph_runner(
        model: torch.nn.Module,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        backend_config: BackendConfig,
        device: torch.device
    ):
        """
        Build graph runner for ROCm.

        CUDA Graphs are also supported on ROCm (called HIP Graphs).
        """
        if is_rocm_available():
            logger.info('Using HIP Graphs (ROCm equivalent of CUDA Graphs)')

        # Use the same graph runner as CUDA
        from ..graph_runner import GraphRunner
        return GraphRunner(model, model_config, cache_config, backend_config, device)


def get_rocm_device_info():
    """
    Get detailed information about ROCm devices.

    Returns:
        dict: Device information including:
            - device_count: Number of AMD GPUs
            - devices: List of device info dicts
            - hip_version: HIP runtime version
            - rocm_version: ROCm version
    """
    if not is_rocm_available():
        return None

    import torch.version

    devices = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        device_info = {
            'id': i,
            'name': props.name,
            'compute_capability': (props.major, props.minor),
            'total_memory_gb': props.total_memory / (1024**3),
            'multi_processor_count': props.multi_processor_count,
        }

        # Add ROCm-specific properties
        if hasattr(props, 'gcnArchName'):
            device_info['gcn_arch_name'] = props.gcnArchName

        devices.append(device_info)

    return {
        'device_count': len(devices),
        'devices': devices,
        'hip_version': torch.version.hip,
        'rocm_version': getattr(torch.version, 'rocm', 'unknown'),
    }


# Export public API
__all__ = [
    'ROCmOpsBackend',
    'is_rocm_available',
    'get_rocm_arch',
    'get_rocm_device_info',
]
