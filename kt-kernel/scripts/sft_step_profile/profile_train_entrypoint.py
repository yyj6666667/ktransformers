#!/usr/bin/env python3
"""Install the external step profiler and enter LLaMA-Factory training.

All LLaMA-Factory arguments are left in ``sys.argv`` and parsed by its regular
``run_exp`` entry point.  Profiling is configured only through ``KT_STEP_PROFILE_*``
environment variables, so this wrapper does not add or reinterpret YAML fields.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_package() -> None:
    scripts_dir = Path(__file__).resolve().parents[1]
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def main() -> None:
    _bootstrap_package()
    from sft_step_profile.trainer_integration import install

    mode = os.environ.get("KT_STEP_PROFILE_MODE", "phase").strip().lower()
    manager = install(mode=mode)
    try:
        from llamafactory.train.tuner import run_exp

        run_exp()
    except BaseException:
        try:
            manager.close(partial=True)
        except Exception as close_error:
            print(
                f"[kt-step-profile] cleanup failed: {close_error!r}",
                file=sys.stderr,
                flush=True,
            )
        raise
    else:
        manager.close(partial=False)


if __name__ == "__main__":
    main()
