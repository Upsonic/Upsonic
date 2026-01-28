"""
Overhead Analysis Benchmark

Compares the performance overhead between Direct LLM Call (minimal overhead)
and Agent (full-featured) approaches in the Upsonic framework.

This benchmark measures:
- Memory footprint (object sizes, peak memory usage)
- Execution speed (initialization time, task completion time)
- Performance ratios and comparisons
"""

__version__ = "0.1.0"

from .test_cases import TestCases

__all__ = ["TestCases"]

