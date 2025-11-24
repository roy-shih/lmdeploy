"""Test sampling penalty kernel (corrected version)."""
import torch


def test_repetition_penalty_triton():
    """Test Triton repetition penalty kernel (two-phase implementation)."""
    print("Testing Triton Repetition Penalty Kernel (Two-Phase)...")
    
    # Setup
    batch_size = 4
    vocab_size = 100
    seq_len = 16
    
    # Create test data
    scores = torch.rand(batch_size, vocab_size, dtype=torch.float32, device='cuda')
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device='cuda')
    penalties = 1.0 + torch.rand(batch_size, dtype=torch.float32, device='cuda')
    
    # Clone for comparison
    scores_triton = scores.clone()
    scores_pytorch = scores.clone()
    
    # Triton implementation
    from lmdeploy.pytorch.kernels.cuda.sampling_penalty import apply_repetition_penalty
    apply_repetition_penalty(scores_triton, input_ids, penalties, penalty_type='multiplicative')
    
    # PyTorch reference implementation
    for i in range(batch_size):
        seen_tokens = set()
        for j in range(seq_len):
            tid = input_ids[i, j].item()
            if tid < vocab_size:
                seen_tokens.add(tid)
        
        for tid in seen_tokens:
            logit = scores_pytorch[i, tid].item()
            penalty = penalties[i].item()
            if logit < 0:
                scores_pytorch[i, tid] = logit * penalty
            else:
                scores_pytorch[i, tid] = logit / penalty
    
    # Compare
    max_diff = (scores_triton - scores_pytorch).abs().max().item()
    print(f"Max difference: {max_diff}")
    
    if max_diff < 1e-5:
        print("✅ Test PASSED")
        return True
    else:
        print(f"❌ Test FAILED: max_diff={max_diff}")
        # Print some debug info
        print(f"Triton sample: {scores_triton[0, :5]}")
        print(f"PyTorch sample: {scores_pytorch[0, :5]}")
        return False


def test_additive_penalty():
    """Test additive penalty mode."""
    print("\nTesting Additive Penalty...")
    
    batch_size = 2
    vocab_size = 50
    seq_len = 8
    
    scores = torch.rand(batch_size, vocab_size, dtype=torch.float32, device='cuda')
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device='cuda')
    penalties = torch.tensor([0.5, 1.0], dtype=torch.float32, device='cuda')
    
    scores_triton = scores.clone()
    scores_ref = scores.clone()
    
    # Triton
    from lmdeploy.pytorch.kernels.cuda.sampling_penalty import apply_repetition_penalty
    apply_repetition_penalty(scores_triton, input_ids, penalties, penalty_type='additive')
    
    # Reference
    for i in range(batch_size):
        seen_tokens = set()
        for j in range(seq_len):
            tid = input_ids[i, j].item()
            if tid < vocab_size:
                seen_tokens.add(tid)
        
        for tid in seen_tokens:
            scores_ref[i, tid] = scores_ref[i, tid] - penalties[i]
    
    max_diff = (scores_triton - scores_ref).abs().max().item()
    print(f"Max difference: {max_diff}")
    
    if max_diff < 1e-5:
        print("✅ Test PASSED")
        return True
    else:
        print(f"❌ Test FAILED")
        return False


def test_duplicate_tokens():
    """Test that duplicate tokens are handled correctly (only penalized once)."""
    print("\nTesting Duplicate Token Handling...")
    
    batch_size = 1
    vocab_size = 10
    seq_len = 6
    
    scores = torch.ones(batch_size, vocab_size, dtype=torch.float32, device='cuda')
    # Token 5 appears 3 times, should only be penalized once
    input_ids = torch.tensor([[5, 5, 5, 3, 7, 3]], dtype=torch.long, device='cuda')
    penalties = torch.tensor([2.0], dtype=torch.float32, device='cuda')
    
    scores_triton = scores.clone()
    
    from lmdeploy.pytorch.kernels.cuda.sampling_penalty import apply_repetition_penalty
    apply_repetition_penalty(scores_triton, input_ids, penalties, penalty_type='multiplicative')
    
    # Expected: tokens 3, 5, 7 should be penalized (divided by 2.0)
    # Token 5 appears 3 times but should only be penalized once
    expected = scores.clone()
    expected[0, 3] = 1.0 / 2.0
    expected[0, 5] = 1.0 / 2.0
    expected[0, 7] = 1.0 / 2.0
    
    max_diff = (scores_triton - expected).abs().max().item()
    print(f"Max difference: {max_diff}")
    print(f"Token 5 (appeared 3x): {scores_triton[0, 5].item():.4f} (expected: {expected[0, 5].item():.4f})")
    
    if max_diff < 1e-5:
        print("✅ Test PASSED - Duplicates handled correctly")
        return True
    else:
        print(f"❌ Test FAILED")
        return False


if __name__ == '__main__':
    if torch.cuda.is_available():
        test_repetition_penalty_triton()
        test_additive_penalty()
        test_duplicate_tokens()
    else:
        print("CUDA not available, skipping tests")
