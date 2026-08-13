from __future__ import annotations

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ROOT = Path(__file__).parent.resolve()


setup(
    name="goldrush2-policy-runtime",
    version="0.0.0",
    packages=["policy_runtime"],
    ext_modules=[
        Pybind11Extension(
            "policy_runtime._runtime",
            [
                str(ROOT / "policy_runtime" / "src" / "fast_option.cpp"),
                str(ROOT / "policy_runtime" / "src" / "feature_extractor.cpp"),
                str(ROOT / "policy_runtime" / "python" / "policy_runtime_py.cpp"),
            ],
            include_dirs=[
                str(ROOT),
                str(ROOT / "policy_runtime" / "include"),
            ],
            define_macros=[
                ("POLICY_RUNTIME_FAST_DEBUG", "1"),
            ],
            cxx_std=17,
        )
    ],
    cmdclass={"build_ext": build_ext},
)
