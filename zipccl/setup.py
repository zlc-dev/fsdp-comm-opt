"""Build script for hopper_fastzip2 – SM90-optimized ZipCCL kernels (multi-work
compress optimized: per-block work lookup is O(1) and the deterministic stream
is written compacted directly, so 8/16-work throughput stays close to 1-work)."""
import torch
import os
import sys
import platform as pf

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root_path = os.path.dirname(sys.argv[0])
root_path = root_path if root_path else "."
root_path = os.path.abspath(root_path)
os.chdir(root_path)


def install():
    ext_libs = []
    ext_args = (
        [
            "-Wno-sign-compare",
            "-Wno-unused-but-set-variable",
            "-Wno-terminate",
            "-Wno-unused-function",
            "-Wno-strict-aliasing",
        ]
        if pf.system() == "Linux"
        else []
    )

    ext_libs += ["cuda", "nvrtc"]
    ext_args += ["-DUSE_GPU"]

    nvcc_flags = [
        "-arch=sm_90a",
        "-gencode", "arch=compute_90,code=compute_90", # support RTX 5090
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-O3",
        "--use_fast_math",
    ]

    setup(
        name="hopper_fastzip2",
        packages=find_packages(),
        python_requires=">=3.6, <4",
        install_requires=[],
        zip_safe=False,
        ext_modules=[
            CUDAExtension(
                "hopper_fastzip2",
                sources=[
                    "hopper_fastzip.cpp",
                    "hopper_zip.cu",
                ],
                library_dirs=["/usr/local/cuda/lib64/stubs"],
                libraries=ext_libs,
                extra_compile_args={
                    "cxx": ext_args,
                    "nvcc": nvcc_flags,
                },
            )
        ],
        cmdclass={
            "build_ext": BuildExtension,
        },
    )


install()
