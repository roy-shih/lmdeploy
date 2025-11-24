"""Test ban bad words kernel."""
import torch


def test_ban_bad_words_basic():
    """Test basic bad words banning."""
    print("Testing Ban Bad Words Kernel (Basic)...")
    
    batch_size = 3
    vocab_size = 100
    max_bad_words = 5
    
    # Create test logits
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device='cuda')
    original_logits = logits.clone()
    
    # Bad words: [batch_size, max_bad_words]
    bad_words = torch.tensor([
        [10, 20, 30, -1, -1],  # Batch 0: ban tokens 10, 20, 30
        [5, 15, 25, 35, -1],   # Batch 1: ban tokens 5, 15, 25, 35
        [50, -1, -1, -1, -1],  # Batch 2: ban token 50
    ], dtype=torch.int32, device='cuda')
    
    # Mask for valid bad words
    bad_words_mask = bad_words >= 0
    
    # Apply ban
    from lmdeploy.pytorch.kernels.cuda.ban_bad_words import ban_bad_words
    result = ban_bad_words(logits, bad_words, bad_words_mask)
    
    # Verify
    # Batch 0
    assert result[0, 10] == -float('inf'), f"Token 10 should be banned in batch 0"
    assert result[0, 20] == -float('inf'), f"Token 20 should be banned in batch 0"
    assert result[0, 30] == -float('inf'), f"Token 30 should be banned in batch 0"
    assert result[0, 0] == original_logits[0, 0], f"Token 0 should not be affected in batch 0"
    
    # Batch 1
    assert result[1, 5] == -float('inf'), f"Token 5 should be banned in batch 1"
    assert result[1, 15] == -float('inf'), f"Token 15 should be banned in batch 1"
    assert result[1, 25] == -float('inf'), f"Token 25 should be banned in batch 1"
    assert result[1, 35] == -float('inf'), f"Token 35 should be banned in batch 1"
    
    # Batch 2
    assert result[2, 50] == -float('inf'), f"Token 50 should be banned in batch 2"
    assert result[2, 0] == original_logits[2, 0], f"Token 0 should not be affected in batch 2"
    
    print("✅ Test PASSED")
    return True


def test_ban_bad_words_vs_pytorch():
    """Test Triton kernel vs PyTorch reference."""
    print("\nTesting Ban Bad Words vs PyTorch...")
    
    batch_size = 4
    vocab_size = 50
    max_bad_words = 3
    
    # Create test data
    logits_triton = torch.randn(batch_size, vocab_size, dtype=torch.float32, device='cuda')
    logits_pytorch = logits_triton.clone()
    
    bad_words = torch.randint(0, vocab_size, (batch_size, max_bad_words), dtype=torch.int32, device='cuda')
    bad_words_mask = torch.ones_like(bad_words, dtype=torch.bool)
    
    # Triton version
    from lmdeploy.pytorch.kernels.cuda.ban_bad_words import ban_bad_words
    ban_bad_words(logits_triton, bad_words, bad_words_mask)
    
    # PyTorch reference (from logits_process.py)
    filter_value = -float('inf')
    filtered_scores = logits_pytorch.gather(1, bad_words.long())
    filtered_scores[bad_words_mask] = filter_value
    logits_pytorch.scatter_(1, bad_words.long(), filtered_scores)
    
    # Compare
    if torch.equal(logits_triton, logits_pytorch):
        print("✅ Test PASSED - Triton matches PyTorch")
        return True
    else:
        # Check if differences are only in non-banned tokens (shouldn't happen)
        diff = (logits_triton - logits_pytorch).abs()
        max_diff = diff.max().item()
        print(f"❌ Test FAILED - max_diff={max_diff}")
        return False


def test_ban_bad_words_edge_cases():
    """Test edge cases."""
    print("\nTesting Edge Cases...")
    
    batch_size = 2
    vocab_size = 20
    max_bad_words = 3
    
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device='cuda')
    
    # Edge case: out of vocab range bad words should be ignored
    bad_words = torch.tensor([
        [5, 100, -1],  # 100 is out of range, should be ignored
        [10, 15, -5],  # -5 is negative, should be ignored
    ], dtype=torch.int32, device='cuda')
    
    bad_words_mask = torch.tensor([
        [True, True, False],  # Only first two are valid
        [True, True, True],   # All marked valid but -5 should be ignored by kernel
    ], dtype=torch.bool, device='cuda')
    
    original_logits = logits.clone()
    
    from lmdeploy.pytorch.kernels.cuda.ban_bad_words import ban_bad_words
    result = ban_bad_words(logits, bad_words, bad_words_mask)
    
    # Verify
    assert result[0, 5] == -float('inf'), "Token 5 should be banned"
    assert result[0, 10] == original_logits[0, 10], "Token 10 should not be affected (out of range)"
    assert result[1, 10] == -float('inf'), "Token 10 should be banned"
    assert result[1, 15] == -float('inf'), "Token 15 should be banned"
    
    print("✅ Test PASSED - Edge cases handled correctly")
    return True


if __name__ == '__main__':
    if torch.cuda.is_available():
        test_ban_bad_words_basic()
        test_ban_bad_words_vs_pytorch()
        test_ban_bad_words_edge_cases()
    else:
        print("CUDA not available, skipping tests")
