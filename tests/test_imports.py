# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests that verify the public Python API of kernelDB can be imported correctly.

These tests require the kernelDB C++ extension to be compiled and installed.
They are automatically skipped when the extension is not available.
"""

import types
import pytest

# Skip the entire module if the kernelDB C++ extension is not available
kerneldb = pytest.importorskip(
    "kerneldb",
    reason="kernelDB C++ extension not available – build with ROCm toolchain",
    exc_type=ImportError,
)


# ---------------------------------------------------------------------------
# Module-level import tests
# ---------------------------------------------------------------------------


def test_import_kerneldb_top_level():
    """The top-level kerneldb package must be importable."""
    import kerneldb  # noqa: F401


def test_version_attribute_exists():
    """kerneldb.__version__ must be a non-empty string."""
    import kerneldb

    assert hasattr(kerneldb, "__version__")
    assert isinstance(kerneldb.__version__, str)
    assert len(kerneldb.__version__) > 0


def test_all_attribute_exists():
    """kerneldb.__all__ must be a non-empty list of strings."""
    import kerneldb

    assert hasattr(kerneldb, "__all__")
    assert isinstance(kerneldb.__all__, list)
    assert len(kerneldb.__all__) > 0
    for name in kerneldb.__all__:
        assert isinstance(name, str), f"__all__ entry {name!r} is not a string"


def test_all_entries_are_accessible():
    """Every name listed in __all__ must be accessible as an attribute."""
    import kerneldb

    for name in kerneldb.__all__:
        assert hasattr(kerneldb, name), f"kerneldb.{name} listed in __all__ but not accessible"


# ---------------------------------------------------------------------------
# Public class import tests
# ---------------------------------------------------------------------------


def test_import_KernelDB():
    from kerneldb import KernelDB  # noqa: F401


def test_import_Kernel():
    from kerneldb import Kernel  # noqa: F401


def test_import_Instruction():
    from kerneldb import Instruction  # noqa: F401


def test_import_BasicBlock():
    from kerneldb import BasicBlock  # noqa: F401


def test_import_CDNAKernel():
    from kerneldb import CDNAKernel  # noqa: F401


def test_import_ArchDescriptor():
    from kerneldb import ArchDescriptor  # noqa: F401


def test_import_HsaAgent():
    from kerneldb import HsaAgent  # noqa: F401


def test_import_KernelArgument():
    from kerneldb import KernelArgument  # noqa: F401


def test_import_all_public_classes_at_once():
    """All public classes can be imported in a single statement."""
    from kerneldb import (  # noqa: F401
        KernelDB,
        Kernel,
        Instruction,
        BasicBlock,
        CDNAKernel,
        ArchDescriptor,
        HsaAgent,
        KernelArgument,
    )


# ---------------------------------------------------------------------------
# Type / callable tests (no GPU required)
# ---------------------------------------------------------------------------


def test_KernelDB_is_class():
    from kerneldb import KernelDB

    assert isinstance(KernelDB, type)


def test_Kernel_is_class():
    from kerneldb import Kernel

    assert isinstance(Kernel, type)
