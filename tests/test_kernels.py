# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for KernelDB kernel-level operations:
  - Kernel discovery (get_kernels)
  - Source-line mapping (get_kernel_lines)
  - Instruction extraction and regex filtering (get_instructions_for_line)
  - Basic block extraction
  - High-level Kernel wrapper properties

All tests in this module require a ROCm environment with hipcc and a GPU.
"""

import re
import pytest
from conftest import requires_rocm

kerneldb = pytest.importorskip("kerneldb", reason="kernelDB C++ extension not available")
KernelDB = kerneldb.KernelDB


def _find_kernel(kernels, fragment):
    """Return the first kernel name that contains *fragment*, or skip."""
    matches = [k for k in kernels if fragment in k]
    if not matches:
        pytest.skip(f"No kernel containing {fragment!r} found in binary")
    return matches[0]


# ---------------------------------------------------------------------------
# Kernel discovery
# ---------------------------------------------------------------------------


@requires_rocm
def test_kernel_discovery(simple_binary):
    """Binary must expose two named kernels: vector_add and vector_multiply."""
    kdb = KernelDB(simple_binary)
    kernels = kdb.get_kernels()

    assert isinstance(kernels, list)
    assert len(kernels) == 2
    assert all(isinstance(k, str) and k for k in kernels)
    assert any("vector_add" in k for k in kernels)
    assert any("vector_multiply" in k for k in kernels)


# ---------------------------------------------------------------------------
# Source-line mapping
# ---------------------------------------------------------------------------


@requires_rocm
def test_kernel_lines(simple_binary):
    """Each kernel must map to a non-empty list of positive source lines."""
    kdb = KernelDB(simple_binary)
    kernels = kdb.get_kernels()

    for kernel_name in kernels:
        lines = kdb.get_kernel_lines(kernel_name)
        assert isinstance(lines, list) and lines, f"No lines for {kernel_name}"
        assert all(isinstance(ln, int) and ln > 0 for ln in lines)

    add_name = _find_kernel(kernels, "vector_add")
    mul_name = _find_kernel(kernels, "vector_multiply")
    assert set(kdb.get_kernel_lines(add_name)) != set(kdb.get_kernel_lines(mul_name))


# ---------------------------------------------------------------------------
# Instruction extraction
# ---------------------------------------------------------------------------


@requires_rocm
def test_instructions(simple_binary):
    """Instructions for each source line must have the expected attributes."""
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")

    for line in kdb.get_kernel_lines(kernel_name):
        instructions = kdb.get_instructions_for_line(kernel_name, line)
        assert isinstance(instructions, list)
        for inst in instructions:
            assert isinstance(inst.disassembly, str) and inst.disassembly
            assert isinstance(inst.line, int) and inst.line == line
            assert isinstance(inst.column, int)
            assert hasattr(inst, "file_name")


# ---------------------------------------------------------------------------
# Instruction filtering
# ---------------------------------------------------------------------------


@requires_rocm
def test_instruction_filtering(simple_binary):
    """Regex filtering must return a subset that each matches the pattern."""
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    pattern = ".*(load|store).*"

    all_insts, mem_ops = [], []
    for line in kdb.get_kernel_lines(kernel_name):
        all_insts.extend(kdb.get_instructions_for_line(kernel_name, line))
        mem_ops.extend(kdb.get_instructions_for_line(kernel_name, line, pattern))

    assert len(mem_ops) > 0, "vector_add must have at least one load/store"
    assert len(mem_ops) <= len(all_insts)
    assert all(re.search(pattern, inst.disassembly, re.IGNORECASE) for inst in mem_ops)

    # A pattern that matches nothing returns empty lists
    for line in kdb.get_kernel_lines(kernel_name):
        assert kdb.get_instructions_for_line(kernel_name, line, "NOMATCH__xyz") == []


# ---------------------------------------------------------------------------
# Basic blocks and Kernel wrapper
# ---------------------------------------------------------------------------


@requires_rocm
def test_basic_blocks_and_kernel_wrapper(simple_binary):
    """Kernel wrapper properties and basic-block extraction must work correctly."""
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    kernel = kdb.get_kernel(kernel_name)

    # basic blocks
    blocks = kernel.get_basic_blocks()
    assert isinstance(blocks, list) and blocks

    # wrapper properties
    assert kernel.name == kernel_name
    assert isinstance(kernel.signature, str) and kernel.signature
    assert isinstance(kernel.lines, list) and all(ln > 0 for ln in kernel.lines)
    assert isinstance(kernel.assembly, list) and all(isinstance(s, str) for s in kernel.assembly)
    assert isinstance(kernel.files, list) and all(isinstance(f, str) for f in kernel.files)

    # instructions via wrapper
    first_line = kernel.lines[0]
    insts = kernel.get_instructions_for_line(first_line)
    assert isinstance(insts, list) and insts
    filtered = kernel.get_instructions_for_line(first_line, ".*load.*")
    assert len(filtered) <= len(insts)

    # file name helper
    assert isinstance(kdb.get_file_name(kernel_name, 0), str)
