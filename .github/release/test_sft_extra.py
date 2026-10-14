"""Keep the SFT extra aligned with the public KT training stack."""

import runpy
from pathlib import Path
from unittest.mock import patch

from packaging.requirements import Requirement


def test_sft_extra_pins_matching_llamafactory_stack():
    setup_path = Path(__file__).resolve().parents[2] / "setup.py"
    with patch("setuptools.setup") as setup:
        runpy.run_path(str(setup_path))

    extras = setup.call_args.kwargs["extras_require"]
    requirements = {req.name: str(req.specifier) for req in map(Requirement, extras["sft"])}
    assert requirements == {
        "transformers-kt": "==5.6.0.post6",
        "accelerate-kt": "==1.14.0.post4",
        "llamafactory-kt": "==0.9.6.dev0",
    }
    assert all("llamafactory" not in dep for dep in extras["sglang"])
