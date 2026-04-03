#!/usr/bin/env python
"""
Compatibility entrypoint for unified weather model evaluation.

The implementation now lives in `evaluate_weather.py` so evaluation has its
own dedicated module while the original CLI command keeps working.
"""

from evaluate_weather import main


if __name__ == "__main__":
    main()
