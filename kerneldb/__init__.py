# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

"""
kernelDB - Python bindings for querying CDNA kernel data

This module provides Python access to the kernelDB C++ library for analyzing
HIP/ROCm kernel binaries and extracting instruction-level information mapped
to source code lines.
"""

from .api import (
    KernelDB,
    Kernel,
    Instruction,
    BasicBlock,
    CDNAKernel,
    ArchDescriptor,
    HsaAgent,
    KernelArgument,
)

# Get version from package metadata
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("kerneldb")
except PackageNotFoundError:
    # Package is not installed, use fallback
    __version__ = "0.0.0+unknown"
__all__ = [
    "KernelDB",
    "Kernel",
    "Instruction",
    "BasicBlock",
    "CDNAKernel",
    "ArchDescriptor",
    "HsaAgent",
    "KernelArgument",
]

