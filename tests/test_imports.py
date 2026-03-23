# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Tests that the public Python API of kernelDB can be imported correctly."""

import pytest

kerneldb = pytest.importorskip("kerneldb", reason="kernelDB C++ extension not available")


def test_package_metadata():
    assert isinstance(kerneldb.__version__, str) and kerneldb.__version__
    assert isinstance(kerneldb.__all__, list) and kerneldb.__all__


def test_public_api_accessible():
    for name in ("KernelDB", "Kernel", "Instruction", "BasicBlock",
                 "CDNAKernel", "ArchDescriptor", "HsaAgent", "KernelArgument"):
        assert getattr(kerneldb, name, None) is not None, f"kerneldb.{name} not accessible"


def test_KernelDB_and_Kernel_are_classes():
    assert isinstance(kerneldb.KernelDB, type)
    assert isinstance(kerneldb.Kernel, type)
