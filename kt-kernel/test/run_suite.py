import argparse
import glob
import os
import subprocess
import sys
from typing import List

from ci.ci_register import HWBackend, CIRegistry, collect_tests
from ci.ci_utils import TestFile, run_unittest_files

HW_MAPPING = {
    "cpu": HWBackend.CPU,
    "cuda": HWBackend.CUDA,
    "amd": HWBackend.AMD,
}

LABEL_MAPPING = {
    HWBackend.CPU: ["default"],
    HWBackend.AMD: ["stage-a-test-1"],
    HWBackend.CUDA: ["stage-a-test-1"],
}


def _filter_tests(
    ci_tests: List[CIRegistry], hw: HWBackend, suite: str
) -> List[CIRegistry]:
    ci_tests = [t for t in ci_tests if t.backend == hw]
    ret = []
    for t in ci_tests:
        assert t.suite in LABEL_MAPPING[hw], f"Unknown stage {t.suite} for backend {hw}"
        if t.suite == suite:
            ret.append(t)
    return ret


def _run_pytest_sft_files(hw: HWBackend, suite: str) -> int:
    if hw != HWBackend.CPU or suite != "default":
        return 0
    files = sorted(glob.glob("per_commit/**/test_sft_*.py", recursive=True))
    files.extend(sorted(glob.glob("per_commit/**/test_cpu_detect_metadata.py", recursive=True)))
    if not files:
        return 0
    command = [sys.executable, "-m", "pytest", "-q", *files]
    print(f"Running pytest suite: {' '.join(command)}", flush=True)
    environment = os.environ.copy()
    environment["KT_REQUIRE_SFT_EXTENSION"] = "1"
    return subprocess.run(command, check=False, env=environment).returncode


def run_per_commit(hw: HWBackend, suite: str):
    files = glob.glob("per_commit/**/*.py", recursive=True)
    # Exclude __init__.py files as they don't contain test registrations
    files = [f for f in files if not f.endswith("__init__.py")]
    pytest_files = set(glob.glob("per_commit/**/test_sft_*.py", recursive=True))
    pytest_files.update(glob.glob("per_commit/**/test_cpu_detect_metadata.py", recursive=True))
    files = [file for file in files if file not in pytest_files]
    ci_tests = _filter_tests(collect_tests(files), hw, suite)
    test_files = [TestFile(t.filename, t.est_time) for t in ci_tests]

    registry_result = run_unittest_files(
        test_files,
        timeout_per_file=1200,
        continue_on_error=False,
    )
    if registry_result != 0:
        return registry_result
    return 0 if _run_pytest_sft_files(hw, suite) == 0 else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hw",
        type=str,
        choices=["cpu", "cuda", "amd"],
        required=True,
        help="Hardware backend to run tests on.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        help="Test suite to run.",
    )
    args = parser.parse_args()
    hw = HW_MAPPING[args.hw]
    exit_code = run_per_commit(hw, args.suite)
    # run_unittest_files returns 0 for success, -1 for failure
    # Convert to standard exit codes: 0 for success, 1 for failure
    sys.exit(0 if exit_code == 0 else 1)


if __name__ == "__main__":
    main()
