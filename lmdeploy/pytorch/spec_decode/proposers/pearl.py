# Copyright (c) OpenMMLab. All rights reserved.
"""PEARL (Parallel Speculative Decoding) Proposer.

This implements a simplified version of nano-PEARL's parallel speculative decoding:
- Draft and Target models on separate GPUs
- Parallel execution using CUDA Streams
- Pre-verification and adaptive draft length
"""

from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.profiler import record_function

from lmdeploy.utils import get_logger

from ...config import ModelConfig, PEARLConfig
from ...engine.cache_engine import CacheEngine
from ...model_inputs import ModelInputs
from ...strategies.base.model_agent import ExtraInputs
from .base import SPEC_PROPOSERS, BaseSpecProposer

logger = get_logger('lmdeploy')


@SPEC_PROPOSERS.register_module(name='pearl')
class PEARLProposer(BaseSpecProposer):
    """PEARL (Parallel Speculative Decoding) Proposer.
    
    Implements "Offloaded Serial" speculative decoding:
    1. GPU separation: Draft on dedicated GPUs (Offloading)
    2. CUDA streams: Asynchronous execution on draft device
    3. Adaptive gamma: Dynamic draft length
    
    Note: This is NOT the full pipelined PEARL algorithm (which overlaps
    Verification and Generation). It is Standard Speculative Decoding
    but with the Draft Model offloaded to another GPU to reduce VRAM
    usage and interference on the Target GPU.
    """

    def __init__(self, specdecode_config: PEARLConfig, device: torch.device = None):
        super().__init__(specdecode_config, device)
        
        # PEARL specific config
        self.pearl_config = specdecode_config
        self.gamma = specdecode_config.gamma
        self.draft_devices = specdecode_config.draft_devices
        self.target_devices = specdecode_config.target_devices
        
        # Draft model will be on first draft device
        self.draft_device = f"cuda:{self.draft_devices[0]}"
        
        # CUDA streams for parallel execution
        self.draft_stream = None
        self.target_stream = None
        
        # Adaptive gamma lookup table
        self.gamma_lut = {}  # {batch_size: optimal_gamma}

        
        logger.info(f"Initializing PEARL Proposer: "
                   f"draft_devices={self.draft_devices}, "
                   f"target_devices={self.target_devices}, "
                   f"gamma={self.gamma}")

    def build_model(self,
                    empty_init: bool,
                    target_model: torch.nn.Module = None,
                    model_format=None,
                    build_model_ctx=None):
        """Build draft model on designated GPU."""
        # Override device to use draft GPU
        self.device = torch.device(self.draft_device)
        
        super().build_model(empty_init,
                          target_model=target_model,
                          model_format=model_format,
                          build_model_ctx=build_model_ctx)
        
        # Move model to draft device
        self.model = self.model.to(self.draft_device)
        
        # Initialize CUDA streams for parallel execution
        if self.pearl_config.use_parallel_streams:
            self.draft_stream = torch.cuda.Stream(device=self.draft_device)
            # Target stream will be on first target device
            target_device = f"cuda:{self.target_devices[0]}"
            self.target_stream = torch.cuda.Stream(device=target_device)
            logger.info("PEARL: Initialized separate CUDA streams for draft/target")
        
        logger.info(f"PEARL draft model loaded on {self.draft_device}")

    def auto_profile_gamma(self, target_model, batch_sizes: List[int] = [1, 4, 8, 16, 32, 64, 128]):
        """Auto-profile optimal gamma for different batch sizes.
        
        Similar to nano-PEARL's auto_set_gamma, profiles draft and target
        throughput at different batch sizes to determine optimal draft length.
        """
        if not self.pearl_config.enable_adaptive_gamma:
            # Use fixed gamma
            for bs in batch_sizes:
                self.gamma_lut[bs] = self.gamma if self.gamma > 0 else 3
            return
        
        logger.info("PEARL: Auto-profiling optimal gamma for different batch sizes...")
        
        # TODO: Implement actual profiling
        # For now, use heuristics based on batch size
        for bs in batch_sizes:
            if bs <= 4:
                self.gamma_lut[bs] = 2
            elif bs <= 16:
                self.gamma_lut[bs] = 3
            elif bs <= 32:
                self.gamma_lut[bs] = 4
            else:
                self.gamma_lut[bs] = 5
        
        logger.info(f"PEARL gamma lookup table: {self.gamma_lut}")

    def get_adaptive_gamma(self, batch_size: int) -> int:
        """Get optimal gamma for current batch size."""
        if not self.gamma_lut:
            return self.gamma if self.gamma > 0 else 3
        
        # Find closest batch size in LUT
        closest_bs = min(self.gamma_lut.keys(), 
                        key=lambda x: abs(x - batch_size))
        return self.gamma_lut[closest_bs]

    @record_function('pearl_draft_forward')
    def draft_tokens_parallel(self,
                             previous_output: Dict[str, torch.Tensor],
                             model_inputs: ModelInputs,
                             extra_inputs: ExtraInputs,
                             cache_engine: CacheEngine,
                             num_tokens: int = None) -> torch.Tensor:
        """Generate draft tokens using parallel streams.
        
        Args:
            previous_output: Output from the first forward pass (already done by agent)
            model_inputs: Input to draft model
            extra_inputs: Extra inputs with metadata
            cache_engine: Cache engine for KV cache
            num_tokens: Number of tokens to generate (gamma)
            
        Returns:
            draft_tokens: [batch_size, num_tokens] tensor of draft token IDs
        """
        if num_tokens is None:
            batch_size = model_inputs.input_ids.size(0)
            num_tokens = self.get_adaptive_gamma(batch_size)
        
        draft_tokens = []
        current_inputs = model_inputs
        
        # Use draft stream if available
        stream_ctx = (torch.cuda.stream(self.draft_stream) 
                     if self.draft_stream else torch.cuda.default_stream())
        
        # 1. Process the first token (from previous_output)
        with stream_ctx:
            draft_token_ids, model_metas, hidden_states = self.get_outputs(
                previous_output, current_inputs, extra_inputs
            )
            draft_tokens.append(draft_token_ids)
            
            # 2. Generate remaining tokens
            if num_tokens > 1:
                # Update inputs for the next step
                current_inputs = self.update_inputs_decoding(
                    current_inputs, extra_inputs, draft_token_ids,
                    None, model_metas
                )
                
                for step in range(num_tokens - 1):
                    # Forward pass through draft model
                    outputs = self._forward(current_inputs, cache_engine)
                    
                    # Get draft token
                    draft_token_ids, model_metas, hidden_states = self.get_outputs(
                        outputs, current_inputs, extra_inputs
                    )
                    
                    draft_tokens.append(draft_token_ids)
                    
                    # Update inputs for next token (if not last step)
                    if step < num_tokens - 2:
                        current_inputs = self.update_inputs_decoding(
                            current_inputs, extra_inputs, draft_token_ids,
                            None, model_metas
                        )
        
        # Stack tokens: [batch_size, num_tokens]
        draft_tokens = torch.cat(draft_tokens, dim=1)
        
        # Ensure synchronization with parallel stream
        ready_event = torch.cuda.Event()
        ready_event.record(self.draft_stream)
        torch.cuda.current_stream().wait_event(ready_event)
        
        return draft_tokens

    def get_outputs(self,
                    model_outputs: Dict[str, torch.Tensor],
                    model_inputs: ModelInputs,
                    extra_inputs: ExtraInputs = None):
        """Get outputs from draft model.
        
        For PEARL, we use greedy decoding for draft tokens to minimize
        communication overhead (similar to nano-PEARL).
        """
        hidden_states = model_outputs['hidden_states']
        model_metas = model_outputs.get('model_metas')
        
        # Get logits
        logits = self.get_logits(hidden_states)[0]
        
        # Greedy sampling for draft (no temperature)
        draft_token_ids = logits.argmax(dim=-1, keepdim=True)
        
        return draft_token_ids, model_metas, hidden_states



    def update_gamma(self, num_accepted: int, num_drafted: int, batch_size: int = 1):
        """Update adaptive gamma based on acceptance rate using AIMD algorithm.
        
        AIMD: Additive Increase, Multiplicative Decrease
        - High acceptance rate (> 0.8): Increase gamma
        - Medium acceptance rate (0.5 - 0.8): Keep gamma
        - Low acceptance rate (< 0.5): Decrease gamma
        """
        if not self.pearl_config.enable_adaptive_gamma or self.pearl_config.gamma > 0:
            return

        acceptance_rate = num_accepted / num_drafted if num_drafted > 0 else 0.0
        
        # Get current gamma for this batch size (approximate)
        current_gamma = self.get_adaptive_gamma(batch_size)
        new_gamma = current_gamma
        
        if acceptance_rate > 0.8:
            # Additive Increase
            new_gamma = min(current_gamma + 1, 8)  # Max gamma 8
        elif acceptance_rate < 0.5:
            # Multiplicative Decrease
            new_gamma = max(int(current_gamma * 0.8), 2)  # Min gamma 2
            
        if not hasattr(self, '_log_counter'):
            self._log_counter = 0
        
        self._log_counter += 1
        if self._log_counter % 50 == 0:
            mat = num_accepted / batch_size if batch_size > 0 else 0
            logger.info(f"[PEARL] Stats | Batch Size: {batch_size} | "
                       f"Gamma: {current_gamma} -> {new_gamma} | "
                       f"Acceptance Rate: {acceptance_rate:.2f} | "
                       f"MAT: {mat:.2f} | "
                       f"Drafted: {num_drafted}, Accepted: {num_accepted}")
            
        # Update LUT
        # Since we use batch size bins, we update the bin for this batch size
        self.gamma_lut[batch_size] = new_gamma


def build_pearl_proposer(specdecode_config: PEARLConfig, device: str = 'cuda'):
    """Build PEARL proposer."""
    return PEARLProposer(specdecode_config, device=device)
