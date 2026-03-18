# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests that verify the public Python API of kernelDB can be imported correctly.

These tests require the kernelDB C++ extension to be compiled and installed.
They are automatically skipped when the extension is not available.
"""

import pytest

# Skip the entire module if the kernelDB C++ extension is not available
kerneldb = pytest.importorskip(
    "kerneldb",
    reason="kernelDB C++ extension not available – build with ROCm toolchain",
)

PUBLIC_CLASSES = [
    "KernelDB",
    "Kernel",
    "Instruction",
    "BasicBlock",
    "CDNAKernel",
    "ArchDescriptor",
    "HsaAgent",
    "KernelArgument",
]


def test_package_metadata():
    """kerneldb must expose a non-empty __version__ and a valid __all__."""
    assert isinstance(kerneldb.__version__, str) and kerneldb.__version__
    assert isinstance(kerneldb.__all__, list) and kerneldb.__all__
    assert all(isinstance(n, str) for n in kerneldb.__all__)


def test_public_api_accessible():
    """All expected public classes must be importable from the package."""
    for name in PUBLIC_CLASSES:
        obj = getattr(kerneldb, name, None)
        assert obj is not None, f"kerneldb.{name} not accessible"


def test_KernelDB_and_Kernel_are_classes():
    assert isinstance(kerneldb.KernelDB, type)
    assert isinstance(kerneldb.Kernel, type)
