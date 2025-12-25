"""
Tests for Morpher wrapper classes.

Run with:
    poetry run pytest tests/test_morpher_wrapper.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "HRM"))

from morpher_wrapper import (
    MorpherACT,
    MorpherFixedSteps,
    MorpherInner,
    MorpherWrapperConfig,
    MorpherACTCarry,
    MorpherStackedCarry,
    MorpherInnerCarry,
)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def minimal_config():
    """Minimal config for fast CPU tests."""
    return dict(
        batch_size=2,
        seq_len=4,
        vocab_size=5,
        num_puzzle_identifiers=3,
        d=8,
        time_scales=[1],
        enc_dec_rank=4,
        io_dim=16,
        puzzle_emb_ndim=0,
        halt_max_steps=2,
        cycles_per_level=[1],
        forward_dtype='float32',
    )


@pytest.fixture
def config_with_puzzle_emb():
    """Config with puzzle embedding enabled."""
    return dict(
        batch_size=2,
        seq_len=4,
        vocab_size=5,
        num_puzzle_identifiers=3,
        d=8,
        time_scales=[1, 2],
        enc_dec_rank=4,
        io_dim=16,
        puzzle_emb_ndim=16,
        halt_max_steps=3,
        cycles_per_level=[1],
        forward_dtype='float32',
    )


@pytest.fixture
def sample_batch():
    """Sample batch for testing."""
    return {
        'inputs': torch.randint(0, 5, (2, 4)),
        'labels': torch.randint(0, 5, (2, 4)),
        'puzzle_identifiers': torch.zeros(2, dtype=torch.long),
    }


# =============================================================================
# MorpherWrapperConfig tests
# =============================================================================
class TestMorpherWrapperConfig:
    def test_minimal_config(self, minimal_config):
        config = MorpherWrapperConfig(**minimal_config)
        assert config.batch_size == 2
        assert config.seq_len == 4
        assert config.d == 8
        assert config.time_scales == [1]

    def test_defaults(self):
        config = MorpherWrapperConfig(
            batch_size=1,
            seq_len=4,
            vocab_size=5,
            num_puzzle_identifiers=1,
        )
        assert config.d == 64
        assert config.time_scales == [1, 2, 4]
        assert config.halt_max_steps == 16


# =============================================================================
# MorpherInner tests
# =============================================================================
class TestMorpherInner:
    def test_creation(self, minimal_config):
        config = MorpherWrapperConfig(**minimal_config)
        model = MorpherInner(config)
        assert model is not None
        assert len(model.morphers) == 1

    def test_empty_carry(self, minimal_config):
        config = MorpherWrapperConfig(**minimal_config)
        model = MorpherInner(config)
        carry = model.empty_carry(batch_size=2)
        
        assert isinstance(carry, MorpherStackedCarry)
        assert len(carry.level_states) == 1
        assert carry.level_states[0].z_btnd.shape[0] == 2

    def test_forward(self, minimal_config, sample_batch):
        config = MorpherWrapperConfig(**minimal_config)
        model = MorpherInner(config)
        
        # Get input embeddings for reset
        input_emb = model._input_embeddings(
            sample_batch['inputs'],
            sample_batch['puzzle_identifiers']
        )
        
        # Create and reset carry
        carry = model.empty_carry(batch_size=2)
        reset_flags = torch.ones(2, dtype=torch.bool)
        carry = model.reset_carry(reset_flags, carry, input_emb)
        
        # Forward
        new_carry, logits, (q_halt, q_continue) = model(carry, sample_batch)
        
        assert logits.shape == (2, 4, 5)  # [B, seq_len, vocab_size]
        assert q_halt.shape == (2,)
        assert q_continue.shape == (2,)


# =============================================================================
# MorpherACT tests
# =============================================================================
class TestMorpherACT:
    def test_creation(self, minimal_config):
        model = MorpherACT(minimal_config)
        assert model is not None

    def test_initial_carry(self, minimal_config, sample_batch):
        model = MorpherACT(minimal_config)
        carry = model.initial_carry(sample_batch)
        
        assert isinstance(carry, MorpherACTCarry)
        assert carry.halted.all()  # Should start halted
        assert (carry.steps == 0).all()

    def test_forward(self, minimal_config, sample_batch):
        model = MorpherACT(minimal_config)
        carry = model.initial_carry(sample_batch)
        
        new_carry, outputs = model(carry, sample_batch)
        
        assert 'logits' in outputs
        assert 'q_halt_logits' in outputs
        assert 'q_continue_logits' in outputs
        assert outputs['logits'].shape == (2, 4, 5)

    def test_halting_behavior(self, minimal_config, sample_batch):
        """Test that halting works correctly over multiple steps."""
        model = MorpherACT(minimal_config)
        model.eval()  # Disable training mode
        
        carry = model.initial_carry(sample_batch)
        steps_taken = 0
        
        for _ in range(10):  # Max iterations
            carry, outputs = model(carry, sample_batch)
            steps_taken += 1
            
            if carry.halted.all():
                break
        
        # Should halt within halt_max_steps (2 in minimal_config)
        assert steps_taken <= minimal_config['halt_max_steps']

    def test_with_puzzle_embedding(self, config_with_puzzle_emb, sample_batch):
        """Test with puzzle embeddings enabled."""
        model = MorpherACT(config_with_puzzle_emb)
        carry = model.initial_carry(sample_batch)
        
        new_carry, outputs = model(carry, sample_batch)
        
        assert outputs['logits'].shape == (2, 4, 5)


# =============================================================================
# MorpherFixedSteps tests
# =============================================================================
class TestMorpherFixedSteps:
    def test_creation(self, minimal_config):
        model = MorpherFixedSteps(minimal_config)
        assert model is not None

    def test_forward(self, minimal_config, sample_batch):
        model = MorpherFixedSteps(minimal_config)
        carry = model.initial_carry(sample_batch)
        
        new_carry, outputs = model(carry, sample_batch)
        
        assert 'logits' in outputs
        assert outputs['logits'].shape == (2, 4, 5)

    def test_fixed_halting(self, minimal_config, sample_batch):
        """Test that fixed steps always halt after halt_max_steps."""
        minimal_config['halt_max_steps'] = 3
        model = MorpherFixedSteps(minimal_config)
        
        carry = model.initial_carry(sample_batch)
        steps_taken = 0
        
        for _ in range(5):
            carry, outputs = model(carry, sample_batch)
            steps_taken += 1
            
            if carry.halted.all():
                break
        
        assert steps_taken == 3  # Should be exactly halt_max_steps


# =============================================================================
# ACTLossHead integration tests
# =============================================================================
class TestACTLossHeadIntegration:
    def test_with_morpher_act(self, minimal_config, sample_batch):
        """Test MorpherACT with ACTLossHead."""
        from models.losses import ACTLossHead
        
        model = MorpherACT(minimal_config)
        loss_model = ACTLossHead(model, loss_type='stablemax_cross_entropy')
        
        carry = loss_model.initial_carry(sample_batch)
        new_carry, loss, metrics, outputs, all_finish = loss_model(
            return_keys=[], carry=carry, batch=sample_batch
        )
        
        assert loss.requires_grad
        assert 'count' in metrics
        assert 'accuracy' in metrics
        assert 'lm_loss' in metrics

    def test_with_morpher_fixed(self, minimal_config, sample_batch):
        """Test MorpherFixedSteps with ACTLossHead."""
        from models.losses import ACTLossHead
        
        model = MorpherFixedSteps(minimal_config)
        loss_model = ACTLossHead(model, loss_type='stablemax_cross_entropy')
        
        carry = loss_model.initial_carry(sample_batch)
        new_carry, loss, metrics, outputs, all_finish = loss_model(
            return_keys=[], carry=carry, batch=sample_batch
        )
        
        assert loss.requires_grad

    def test_training_loop(self, minimal_config, sample_batch):
        """Test a mini training loop."""
        from models.losses import ACTLossHead
        
        model = MorpherACT(minimal_config)
        loss_model = ACTLossHead(model, loss_type='stablemax_cross_entropy')
        
        carry = loss_model.initial_carry(sample_batch)
        
        # Run until all finish
        for step in range(10):
            carry, loss, metrics, _, all_finish = loss_model(
                return_keys=[], carry=carry, batch=sample_batch
            )
            if all_finish:
                break
        
        assert all_finish


# =============================================================================
# Multi-level tests
# =============================================================================
class TestMultiLevel:
    def test_two_levels(self, minimal_config, sample_batch):
        """Test with two stacked Morpher levels."""
        minimal_config['num_levels'] = 2
        minimal_config['cycles_per_level'] = [1, 1]
        
        model = MorpherACT(minimal_config)
        assert len(model.inner.morphers) == 2
        
        carry = model.initial_carry(sample_batch)
        new_carry, outputs = model(carry, sample_batch)
        
        assert outputs['logits'].shape == (2, 4, 5)
        assert len(new_carry.inner_carry.level_states) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
