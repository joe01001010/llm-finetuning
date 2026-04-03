#!/usr/bin/env python
"""
Compatibility entrypoint for supervised weather fine-tuning.

The implementation now lives in `fine_tune_sft.py` so the SFT, PPO, and
evaluation stages each have their own dedicated module.
"""

from fine_tune_sft import main


if __name__ == "__main__":
    main()
