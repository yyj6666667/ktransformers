"""Low-overhead, optimizer-step profiling for KT SFT.

The package is a standalone harness.  Importing it does not patch Trainer or
enable any counters; callers must explicitly invoke :func:`install`.
"""

from .trainer_integration import ProfileMode, install

__all__ = ["ProfileMode", "install"]
