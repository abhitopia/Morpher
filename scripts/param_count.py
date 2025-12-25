"""
Calculate parameter counts for HRM vs Morpher comparison.

Run with:
    poetry run python scripts/param_count.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock flash_attn for CPU/macOS (only needed for parameter counting, not actual inference)
sys.modules['flash_attn'] = MagicMock()
sys.modules['flash_attn_interface'] = MagicMock()

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    print("=" * 60)
    print("Parameter Count Comparison: HRM vs Morpher")
    print("=" * 60)
    
    # HRM
    print("\n--- HRM ---")
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
    
    hrm_config = dict(
        batch_size=1, seq_len=64, vocab_size=11, num_puzzle_identifiers=100,
        puzzle_emb_ndim=512, H_cycles=2, L_cycles=2, H_layers=4, L_layers=4,
        hidden_size=512, num_heads=8, expansion=4, pos_encodings='rope',
        halt_max_steps=16, halt_exploration_prob=0.1,
    )
    hrm = HierarchicalReasoningModel_ACTV1(hrm_config)
    hrm_params = count_params(hrm)
    print(f"HRM (hidden=512, H/L=4+4 layers): {hrm_params:,} params")
    
    # Morpher
    print("\n--- Single-Level Morpher ---")
    from morpher_wrapper import MorpherACT
    
    base_config = dict(
        batch_size=1, seq_len=64, vocab_size=11, num_puzzle_identifiers=100,
        puzzle_emb_ndim=512, halt_max_steps=16, forward_dtype='float32',
    )
    
    configs_1level = [
        {'d': 64, 'time_scales': [1, 2, 4], 'enc_dec_rank': 32, 'io_dim': 512, 'num_levels': 1},
        {'d': 96, 'time_scales': [1, 2, 4], 'enc_dec_rank': 48, 'io_dim': 512, 'num_levels': 1},
        {'d': 128, 'time_scales': [1, 2, 4], 'enc_dec_rank': 64, 'io_dim': 512, 'num_levels': 1},
    ]
    
    for cfg in configs_1level:
        full_cfg = {**base_config, **cfg, 'cycles_per_level': [2]}
        model = MorpherACT(full_cfg)
        params = count_params(model)
        print(f"  d={cfg['d']}, scales={cfg['time_scales']}, rank={cfg['enc_dec_rank']}: {params:,} params")
    
    print("\n--- Two-Level Morpher ---")
    configs_2level = [
        {'d': 64, 'time_scales': [1, 2, 4], 'enc_dec_rank': 32, 'io_dim': 512, 'num_levels': 2},
        {'d': 96, 'time_scales': [1, 2, 4], 'enc_dec_rank': 48, 'io_dim': 512, 'num_levels': 2},
        {'d': 128, 'time_scales': [1, 2, 4], 'enc_dec_rank': 64, 'io_dim': 512, 'num_levels': 2},
    ]
    
    for cfg in configs_2level:
        full_cfg = {**base_config, **cfg, 'cycles_per_level': [2, 2]}
        model = MorpherACT(full_cfg)
        params = count_params(model)
        print(f"  d={cfg['d']}, scales={cfg['time_scales']}, rank={cfg['enc_dec_rank']}: {params:,} params")
    
    print("\n" + "=" * 60)
    print(f"Target: {hrm_params:,} params (HRM)")
    print("=" * 60)


if __name__ == "__main__":
    main()
