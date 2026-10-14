"""Lightweight top-level package: pip install ktransformers -> installs kt-kernel.

Extras:
  - ktransformers[sft] installs the matching LLaMA-Factory SFT stack
  - ktransformers[sglang] installs sglang-kt
"""
from pathlib import Path
from setuptools import setup

_version_file = Path(__file__).resolve().parent / "version.py"
_ns = {}
exec(_version_file.read_text(), _ns)
_v = _ns["__version__"]

setup(
    version=_v,
    install_requires=[
        f"kt-kernel=={_v}",
    ],
    extras_require={
        "sft": [
            "transformers-kt==5.6.0.post6",
            "accelerate-kt==1.14.0.post4",
            "llamafactory-kt==0.9.6.dev0",
        ],
        "sglang": [
            f"sglang-kt=={_v}",
        ],
    },
)
