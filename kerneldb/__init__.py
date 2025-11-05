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
)

__version__ = "0.1.0"
__all__ = [
    "KernelDB",
    "Kernel",
    "Instruction",
    "BasicBlock",
    "CDNAKernel",
    "ArchDescriptor",
    "HsaAgent",
]

