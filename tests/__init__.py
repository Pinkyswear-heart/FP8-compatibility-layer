"""Test, benchmark and verification scripts, kept separate from the core packages.

Run from the repo root, e.g. ``python -I tests/test_tfp8.py``; each script bootstraps the
repo root onto ``sys.path`` so it imports the ``tfp8`` package from any cwd. Benchmark
outputs still go to ``.verify/bench/`` (paths are resolved from the repo root).
"""
