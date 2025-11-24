"""Test stop criteria kernels."""
import torch


def test_length_criterion():
    """Test length criterion kernel."""
    print("Testing Length Criterion Kernel...")
    
    batch_size = 4
    
    # Create test data
    finished = torch.zeros(batch_size, dtype=torch.bool, device='cuda')
    sequence_lengths = torch.tensor([10, 15, 20, 25], dtype=torch.int32, device='cuda')
    max_lengths = torch.tensor([20, 15, 20, 30], dtype=torch.int32, device='cuda')
    
    # Expected: batch 1 (15>=15) and batch 2 (20>=20) should be marked finished
    expected = torch.tensor([False, True, True, False], dtype=torch.bool, device='cuda')
    
    # Run kernel
    from lmdeploy.pytorch.kernels.cuda.stop_criteria import check_length_criterion
    result = check_length_criterion(finished, sequence_lengths, max_lengths)
    
    # Verify
    if torch.equal(result, expected):
        print("✅ Test PASSED")
        print(f"  Result: {result.tolist()}")
        return True
    else:
        print(f"❌ Test FAILED")
        print(f"  Expected: {expected.tolist()}")
        print(f"  Got:      {result.tolist()}")
        return False


def test_stop_words_criterion():
    """Test stop words criterion kernel."""
    print("\nTesting Stop Words Criterion Kernel...")
    
    batch_size = 3
    max_seq_len = 10
    max_stop_words = 4
    
    # Create test data
    finished = torch.zeros(batch_size, dtype=torch.bool, device='cuda')
    
    # Output IDs: [max_seq_len, batch_size]
    output_ids = torch.randint(0, 100, (max_seq_len, batch_size), dtype=torch.int32, device='cuda')
    
    # Set specific last tokens
    current_step = 5
    output_ids[current_step, 0] = 10  # Batch 0: token 10
    output_ids[current_step, 1] = 20  # Batch 1: token 20
    output_ids[current_step, 2] = 99  # Batch 2: token 99 (not a stop word)
    
    # Stop words: [batch_size, max_stop_words]
    stop_words = torch.full((batch_size, max_stop_words), -1, dtype=torch.int32, device='cuda')
    stop_words[0, 0] = 10  # Batch 0 has stop word 10
    stop_words[0, 1] = 11
    stop_words[1, 0] = 15  # Batch 1 has stop word 15 (won't match)
    stop_words[1, 1] = 16
    stop_words[2, 0] = 50  # Batch 2 has stop word 50 (won't match)
    
    # Number of valid stop words per batch
    stop_words_len = torch.tensor([2, 2, 1], dtype=torch.int32, device='cuda')
    
    # Expected: only batch 0 should be marked finished (token 10 matches stop word 10)
    expected = torch.tensor([True, False, False], dtype=torch.bool, device='cuda')
    
    # Run kernel
    from lmdeploy.pytorch.kernels.cuda.stop_criteria import check_stop_words_criterion
    result = check_stop_words_criterion(finished, output_ids, stop_words, stop_words_len, current_step)
    
    # Verify
    if torch.equal(result, expected):
        print("✅ Test PASSED")
        print(f"  Result: {result.tolist()}")
        return True
    else:
        print(f"❌ Test FAILED")
        print(f"  Expected: {expected.tolist()}")
        print(f"  Got:      {result.tolist()}")
        return False


def test_combined_criteria():
    """Test that OR logic works correctly when combining criteria."""
    print("\nTesting Combined Criteria (OR logic)...")
    
    batch_size = 4
    
    # Start with some already finished
    finished = torch.tensor([False, True, False, False], dtype=torch.bool, device='cuda')
    
    # Length check
    sequence_lengths = torch.tensor([10, 15, 20, 25], dtype=torch.int32, device='cuda')
    max_lengths = torch.tensor([20, 15, 20, 30], dtype=torch.int32, device='cuda')
    
    from lmdeploy.pytorch.kernels.cuda.stop_criteria import check_length_criterion
    result = check_length_criterion(finished, sequence_lengths, max_lengths)
    
    # Expected: batch 1 (already finished), batch 2 (reached max length)
    expected = torch.tensor([False, True, True, False], dtype=torch.bool, device='cuda')
    
    if torch.equal(result, expected):
        print("✅ Test PASSED - OR logic works correctly")
        print(f"  Result: {result.tolist()}")
        return True
    else:
        print(f"❌ Test FAILED")
        print(f"  Expected: {expected.tolist()}")
        print(f"  Got:      {result.tolist()}")
        return False


if __name__ == '__main__':
    if torch.cuda.is_available():
        test_length_criterion()
        test_stop_words_criterion()
        test_combined_criteria()
    else:
        print("CUDA not available, skipping tests")
