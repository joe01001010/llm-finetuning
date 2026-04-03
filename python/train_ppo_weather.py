#!/usr/bin/env python
"""
Compatibility entrypoint for PPO weather post-training.

The implementation now lives in `fine_tune_ppo.py` so the PPO stage stays
separate from SFT and evaluation while preserving the original CLI command.
"""

from fine_tune_ppo import main


if __name__ == "__main__":
    main()
