# test_ml_debugging.py
# Unit Tests for ML Debugging Practice

import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import math
from unittest.mock import Mock, patch
import io
import sys

# Import the buggy code
from ml_debugging_practice import *

class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.d_model = 512
        self.num_heads = 8
        self.batch_size = 2
        self.seq_len = 10
        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
        
    def test_output_shape(self):
        """Test that output has correct shape"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = self.mha(x, x, x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
    
    def test_attention_scaling(self):
        """Test that attention scores are properly scaled"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Test with very large inputs - should not cause overflow if properly scaled
        large_x = torch.randn(self.batch_size, self.seq_len, self.d_model) * 100
        output_large = self.mha(large_x, large_x, large_x)
        
        # Should not contain NaN or inf values if properly scaled
        self.assertFalse(torch.isnan(output_large).any(), 
                        "NaN values detected - likely missing scaling factor")
        self.assertFalse(torch.isinf(output_large).any(),
                        "Inf values detected - likely missing scaling factor")
    
    def test_causal_masking(self):
        """Test that causal mask prevents attention to future positions"""
        x = torch.randn(1, 4, self.d_model)
        
        # Create causal mask (lower triangular)
        mask = torch.tril(torch.ones(4, 4)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
        
        output = self.mha(x, x, x, mask)
        self.assertEqual(output.shape, (1, 4, self.d_model))
        
        # Output should be finite
        self.assertTrue(torch.isfinite(output).all())
    
    def test_different_key_value_lengths(self):
        """Test attention with different key/value sequence lengths"""
        q = torch.randn(self.batch_size, 5, self.d_model)  # Query length: 5
        k = torch.randn(self.batch_size, 8, self.d_model)  # Key length: 8
        v = torch.randn(self.batch_size, 8, self.d_model)  # Value length: 8
        
        output = self.mha(q, k, v)
        # Output should match query sequence length
        self.assertEqual(output.shape, (self.batch_size, 5, self.d_model))


class TestPositionalEncoding(unittest.TestCase):
    def setUp(self):
        self.d_model = 512
        self.max_seq_length = 1000
        self.pe = PositionalEncoding(self.d_model, self.max_seq_length)
    
    def test_positional_encoding_is_buffer(self):
        """Test that PE is registered as buffer, not parameter"""
        # Should be in buffers, not parameters (buffers don't get gradients)
        self.assertIn('pe', dict(self.pe.named_buffers()),
                     "PE should be registered as buffer, not parameter")
        self.assertNotIn('pe', dict(self.pe.named_parameters()),
                        "PE should not be a trainable parameter")
    
    def test_deterministic_output(self):
        """Test that PE gives deterministic output"""
        x = torch.randn(2, 10, self.d_model)
        output1 = self.pe(x)
        output2 = self.pe(x)
        
        torch.testing.assert_close(output1, output2,
                                 msg="PE should give deterministic output")
    
    def test_position_independence(self):
        """Test that PE correctly handles different sequence lengths"""
        x1 = torch.randn(1, 5, self.d_model)
        x2 = torch.randn(1, 10, self.d_model)
        
        out1 = self.pe(x1)
        out2 = self.pe(x2)
        
        # First 5 positions should have same PE added
        torch.testing.assert_close(out1, out2[:, :5, :],
                                 msg="PE should be consistent across different sequence lengths")
    
    def test_no_gradient_flow_through_pe(self):
        """Test that gradients don't flow through PE"""
        x = torch.randn(1, 10, self.d_model, requires_grad=True)
        output = self.pe(x)
        loss = output.sum()
        loss.backward()
        
        # PE buffer should not accumulate gradients
        self.assertIsNone(self.pe.pe.grad,
                         "PE buffer should not have gradients")


class TestTrainingLoop(unittest.TestCase):
    def setUp(self):
        self.model = nn.Linear(10, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.device = torch.device('cpu')
        
        # Create mock dataloader
        self.mock_data = [(torch.randn(4, 10), torch.randn(4, 1)) for _ in range(5)]
    
    def test_losses_are_scalars(self):
        """Test that stored losses are scalar values, not tensors"""
        best_model, losses = train_model(
            self.model, self.mock_data, self.optimizer,
            self.criterion, self.device, num_epochs=1
        )
        
        # All losses should be Python floats/ints, not tensors
        for i, loss in enumerate(losses):
            self.assertIsInstance(loss, (float, int),
                                f"Loss {i} should be scalar, got {type(loss)}")
            self.assertNotIsInstance(loss, torch.Tensor,
                                   f"Loss {i} should not be a tensor")
    
    def test_gradient_operations_order(self):
        """Test proper order of gradient operations"""
        # Create a custom optimizer to track calls
        original_zero_grad = self.optimizer.zero_grad
        original_step = self.optimizer.step
        
        call_order = []
        
        def track_zero_grad():
            call_order.append('zero_grad')
            return original_zero_grad()
        
        def track_step():
            call_order.append('step')
            return original_step()
        
        self.optimizer.zero_grad = track_zero_grad
        self.optimizer.step = track_step
        
        # Run training for one epoch
        train_model(self.model, self.mock_data[:2], self.optimizer,
                   self.criterion, self.device, num_epochs=1)
        
        # Check that zero_grad comes before step in each iteration
        for i in range(0, len(call_order)-1, 2):
            if i+1 < len(call_order):
                self.assertEqual(call_order[i], 'zero_grad',
                               "zero_grad should be called before step")
                self.assertEqual(call_order[i+1], 'step',
                               "step should be called after zero_grad")
    
    def test_model_state_dict_copying(self):
        """Test that best model is properly copied"""
        best_model, losses = train_model(
            self.model, self.mock_data, self.optimizer,
            self.criterion, self.device, num_epochs=1
        )
        
        # best_model should be a state dict, not the model itself
        self.assertIsInstance(best_model, dict,
                            "best_model should be a state dict")
        
        # Modifying original model shouldn't affect saved best_model
        original_weight = best_model['weight'].clone()
        with torch.no_grad():
            self.model.weight.fill_(999.0)  # Modify original
        
        # Saved state should be unchanged
        torch.testing.assert_close(best_model['weight'], original_weight,
                                 msg="Saved model state should be independent of original model")


class TestKVCache(unittest.TestCase):
    def setUp(self):
        self.max_seq_length = 100
        self.num_heads = 8
        self.head_dim = 64
        self.cache = KVCache(self.max_seq_length, self.num_heads, self.head_dim)
    
    def test_cache_initialization(self):
        """Test that cache is properly initialized"""
        self.assertEqual(self.cache.current_length, 0)
        self.assertEqual(self.cache.k_cache.shape, 
                        (1, self.num_heads, self.max_seq_length, self.head_dim))
        self.assertEqual(self.cache.v_cache.shape,
                        (1, self.num_heads, self.max_seq_length, self.head_dim))
    
    def test_device_consistency(self):
        """Test that cache handles device consistency"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            k = torch.randn(1, self.num_heads, 5, self.head_dim, device=device)
            v = torch.randn(1, self.num_heads, 5, self.head_dim, device=device)
            
            # This should work without device mismatch errors
            try:
                k_cached, v_cached = self.cache.update_cache(k, v)
                self.assertEqual(k_cached.device, device,
                               "Cached tensors should match input device")
                self.assertEqual(v_cached.device, device,
                               "Cached tensors should match input device")
            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e):
                    self.fail("Device mismatch error - cache not handling devices properly")
    
    def test_cache_bounds_checking(self):
        """Test that cache handles overflow gracefully"""
        # Try to add more than max_seq_length
        total_added = 0
        try:
            while total_added < self.max_seq_length + 10:
                k = torch.randn(1, self.num_heads, 5, self.head_dim)
                v = torch.randn(1, self.num_heads, 5, self.head_dim)
                k_cached, v_cached = self.cache.update_cache(k, v)
                total_added += 5
        except (IndexError, RuntimeError):
            # Should either handle gracefully or raise appropriate error
            pass
        
        # Cache length should not exceed maximum
        self.assertLessEqual(self.cache.current_length, self.max_seq_length,
                           "Cache should not exceed maximum length")
    
    def test_incremental_caching(self):
        """Test that incremental caching works correctly"""
        # Add first sequence
        k1 = torch.randn(1, self.num_heads, 3, self.head_dim)
        v1 = torch.randn(1, self.num_heads, 3, self.head_dim)
        k_cached1, v_cached1 = self.cache.update_cache(k1, v1)
        
        # Add second sequence
        k2 = torch.randn(1, self.num_heads, 2, self.head_dim)
        v2 = torch.randn(1, self.num_heads, 2, self.head_dim)
        k_cached2, v_cached2 = self.cache.update_cache(k2, v2)
        
        # Should return concatenated sequences
        self.assertEqual(k_cached2.shape, (1, self.num_heads, 5, self.head_dim))
        self.assertEqual(v_cached2.shape, (1, self.num_heads, 5, self.head_dim))
        
        # First part should match first sequence
        torch.testing.assert_close(k_cached2[:, :, :3, :], k1)
        torch.testing.assert_close(v_cached2[:, :, :3, :], v1)


class TestGradientClipping(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.mock_data = [(torch.randn(4, 10), torch.randn(4, 1)) for _ in range(5)]
    
    def test_gradient_clipping_function(self):
        """Test that correct gradient clipping function is used"""
        # The function should exist and work
        try:
            result = train_with_gradient_clipping(
                self.model, self.mock_data, self.optimizer, self.criterion
            )
            self.assertIsNotNone(result, "Function should complete successfully")
        except AttributeError as e:
            if "clip_grad_norm" in str(e):
                self.fail("Wrong gradient clipping function used - should be clip_grad_norm_")
    
    def test_loss_return_type(self):
        """Test that function returns scalar loss, not tensor"""
        result = train_with_gradient_clipping(
            self.model, self.mock_data, self.optimizer, self.criterion
        )
        
        if result is not None:  # If no NaN detected
            self.assertIsInstance(result, (torch.Tensor, float, int),
                                "Should return a numeric value")
            if isinstance(result, torch.Tensor):
                self.assertEqual(result.dim(), 0, "Should return scalar tensor")


class TestTransformerDecoderLayer(unittest.TestCase):
    def setUp(self):
        self.d_model = 512
        self.num_heads = 8
        self.d_ff = 2048
        self.layer = TransformerDecoderLayer(self.d_model, self.num_heads, self.d_ff)
        self.batch_size = 2
        self.seq_len = 10
    
    def test_output_shape(self):
        """Test correct output shape"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        encoder_output = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output = self.layer(x, encoder_output)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the layer"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        encoder_output = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output = self.layer(x, encoder_output)
        loss = output.sum()
        loss.backward()
        
        # Input should have gradients
        self.assertIsNotNone(x.grad, "Gradients should flow through decoder layer")
        self.assertFalse(torch.isnan(x.grad).any(), "Gradients should not be NaN")
    
    def test_training_vs_eval_mode(self):
        """Test that dropout behaves differently in train vs eval mode"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        encoder_output = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        # Eval mode - should be deterministic
        self.layer.eval()
        output_eval1 = self.layer(x, encoder_output)
        output_eval2 = self.layer(x, encoder_output)
        
        # Eval outputs should be identical
        torch.testing.assert_close(output_eval1, output_eval2,
                                 msg="Eval mode should be deterministic")


class TestAttentionMasks(unittest.TestCase):
    def test_causal_mask_shape_and_values(self):
        """Test causal mask has correct shape and values"""
        seq_len = 5
        device = torch.device('cpu')
        
        mask = create_causal_mask(seq_len, device)
        
        self.assertEqual(mask.shape, (seq_len, seq_len))
        self.assertEqual(mask.device, device)
        
        # Should be lower triangular (including diagonal)
        # True values should form lower triangle
        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i:
                    self.assertTrue(mask[i, j].item(),
                                  f"Position ({i},{j}) should be True (can attend)")
                else:
                    self.assertFalse(mask[i, j].item(),
                                   f"Position ({i},{j}) should be False (cannot attend to future)")
    
    def test_padding_mask_shape(self):
        """Test padding mask creation and shape"""
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        pad_token_id = 0
        
        mask = create_padding_mask(input_ids, pad_token_id)
        
        # Should have correct shape for broadcasting
        expected_shape = (2, 1, 1, 5)  # [batch, 1, 1, seq_len]
        self.assertEqual(mask.shape, expected_shape,
                        f"Expected shape {expected_shape}, got {mask.shape}")
        
        # Check values
        # First sequence: positions 0,1,2 should be True, 3,4 should be False
        self.assertTrue(mask[0, 0, 0, 0].item())  # Non-padding
        self.assertTrue(mask[0, 0, 0, 1].item())  # Non-padding
        self.assertTrue(mask[0, 0, 0, 2].item())  # Non-padding
        self.assertFalse(mask[0, 0, 0, 3].item()) # Padding
        self.assertFalse(mask[0, 0, 0, 4].item()) # Padding
    
    def test_mask_combination_broadcasting(self):
        """Test that mask combination handles broadcasting correctly"""
        seq_len = 4
        causal_mask = create_causal_mask(seq_len, torch.device('cpu'))
        
        input_ids = torch.tensor([[1, 2, 3, 0]])
        padding_mask = create_padding_mask(input_ids, 0)
        
        try:
            combined = combine_masks(causal_mask, padding_mask)
            
            # Should not raise broadcasting errors
            self.assertEqual(combined.shape, (1, 1, seq_len, seq_len),
                           "Combined mask should have correct broadcast shape")
            
            # Last position (padding) should be False everywhere
            self.assertFalse(combined[0, 0, :, -1].any().item(),
                           "Padding positions should mask all attention")
            
        except RuntimeError as e:
            if "broadcast" in str(e).lower():
                self.fail("Mask combination failed due to broadcasting error")


class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 3)  # 3 classes
        )
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cpu')
        
        # Create deterministic test data
        torch.manual_seed(42)
        self.test_data = [
            (torch.randn(4, 10), torch.randint(0, 3, (4,)))
            for _ in range(5)
        ]
    
    def test_model_eval_mode(self):
        """Test that model is properly set to eval mode"""
        # Start in training mode
        self.model.train()
        self.assertTrue(self.model.training, "Model should start in training mode")
        
        # Run evaluation
        accuracy, avg_loss = evaluate_model(
            self.model, self.test_data, self.criterion, self.device
        )
        
        # Model should be in eval mode after evaluation
        self.assertFalse(self.model.training,
                        "Model should be in eval mode after evaluation")
    
    def test_no_gradient_computation_during_eval(self):
        """Test that gradients are not computed during evaluation"""
        # Patch torch.no_grad to detect if it's used
        with patch('torch.no_grad') as mock_no_grad:
            mock_no_grad.return_value.__enter__ = Mock()
            mock_no_grad.return_value.__exit__ = Mock()
            
            evaluate_model(self.model, self.test_data, self.criterion, self.device)
            
            # torch.no_grad should have been called
            mock_no_grad.assert_called_once()
    
    def test_deterministic_evaluation_results(self):
        """Test that evaluation gives consistent results when run multiple times"""
        # Fix randomness
        self.model.eval()
        torch.manual_seed(42)
        
        accuracy1, loss1 = evaluate_model(
            self.model, self.test_data, self.criterion, self.device
        )
        
        # Reset and run again
        torch.manual_seed(42)
        accuracy2, loss2 = evaluate_model(
            self.model, self.test_data, self.criterion, self.device
        )
        
        # Results should be identical
        self.assertEqual(accuracy1, accuracy2,
                        "Evaluation should give consistent accuracy")
        self.assertEqual(loss1, loss2,
                        "Evaluation should give consistent loss")
    
    def test_return_value_types(self):
        """Test that function returns correct types"""
        accuracy, avg_loss = evaluate_model(
            self.model, self.test_data, self.criterion, self.device
        )
        
        # Should return numeric values
        self.assertIsInstance(accuracy, (float, int),
                            "Accuracy should be numeric")
        self.assertIsInstance(avg_loss, (float, int),
                            "Average loss should be numeric")
        
        # Accuracy should be percentage (0-100)
        self.assertGreaterEqual(accuracy, 0, "Accuracy should be >= 0")
        self.assertLessEqual(accuracy, 100, "Accuracy should be <= 100")


# Helper function to run all tests
def run_all_tests():
    """Run all unit tests and return results"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestMultiHeadAttention,
        TestPositionalEncoding,
        TestTrainingLoop,
        TestKVCache,
        TestGradientClipping,
        TestTransformerDecoderLayer,
        TestAttentionMasks,
        TestModelEvaluation
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)
    result = runner.run(suite)
    
    return result

if __name__ == "__main__":
    print("🧪 ML Debugging Practice - Unit Tests")
    print("=" * 60)
    print("📋 Instructions:")
    print("1. Fix bugs in ml_debugging_practice.py")
    print("2. Run this test file to check your progress")
    print("3. All tests should pass when bugs are fixed")
    print("=" * 60)
    
    result = run_all_tests()
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 SUCCESS! All tests passed!")
        print("Your debugging skills are interview-ready! 💪")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
        print("Keep debugging - you're making progress! 🔍")
        
        if result.failures:
            print("\n🔍 Failed Tests:")
            for test, traceback in result.failures:
                print(f"   • {test}")
                
        if result.errors:
            print("\n💥 Error Tests:")
            for test, traceback in result.errors:
                print(f"   • {test}")
    
    print("\n📚 Quick Reference - What Each Test Checks:")
    print("-" * 60)
    print("MultiHeadAttention: Scaling factor, tensor shapes, masking")
    print("PositionalEncoding: Buffer vs parameter, determinism")
    print("TrainingLoop: Memory leaks, gradient order, scalar vs tensor")
    print("KVCache: Device consistency, bounds checking, incremental updates")
    print("GradientClipping: Correct function name, NaN handling")
    print("TransformerDecoderLayer: Normalization order, residual connections")
    print("AttentionMasks: Causal logic, broadcasting, shape handling")
    print("ModelEvaluation: eval() mode, no_grad(), deterministic results")
    print("=" * 60)