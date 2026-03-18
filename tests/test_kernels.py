# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for KernelDB kernel-level operations:
  - Kernel discovery (get_kernels)
  - Source-line mapping (get_kernel_lines)
  - Instruction extraction (get_instructions_for_line)
  - Instruction regex filtering
  - Basic block extraction
  - High-level Kernel wrapper properties

All tests in this module require a ROCm environment with hipcc and a GPU.
"""

import re
import pytest
from conftest import requires_rocm

kerneldb = pytest.importorskip("kerneldb", reason="kernelDB C++ extension not available", exc_type=ImportError)
KernelDB = kerneldb.KernelDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_kernel(kernels: list[str], fragment: str) -> str:
    """Return the first kernel name that contains *fragment*, or skip."""
    matches = [k for k in kernels if fragment in k]
    if not matches:
        pytest.skip(f"No kernel containing {fragment!r} found in binary")
    return matches[0]


# ---------------------------------------------------------------------------
# KernelDB initialisation
# ---------------------------------------------------------------------------


@requires_rocm
def test_kerneldb_init_with_binary(simple_binary):
    """KernelDB must initialise without raising given a valid binary path."""
    kdb = KernelDB(simple_binary)
    assert kdb is not None


@requires_rocm
def test_kerneldb_binary_path_stored(simple_binary):
    """The binary_path attribute must reflect the path that was given."""
    kdb = KernelDB(simple_binary)
    assert kdb.binary_path == simple_binary


# ---------------------------------------------------------------------------
# Kernel discovery
# ---------------------------------------------------------------------------


@requires_rocm
def test_get_kernels_returns_list(simple_binary):
    kdb = KernelDB(simple_binary)
    kernels = kdb.get_kernels()
    assert isinstance(kernels, list)


@requires_rocm
def test_get_kernels_non_empty(simple_binary):
    kdb = KernelDB(simple_binary)
    kernels = kdb.get_kernels()
    assert len(kernels) > 0, "Expected at least one kernel in the binary"


@requires_rocm
def test_get_kernels_names_are_strings(simple_binary):
    kdb = KernelDB(simple_binary)
    for name in kdb.get_kernels():
        assert isinstance(name, str)
        assert len(name) > 0


@requires_rocm
def test_get_kernels_contains_vector_add(simple_binary):
    kdb = KernelDB(simple_binary)
    kernels = kdb.get_kernels()
    assert any("vector_add" in k for k in kernels), (
        f"'vector_add' not found in kernel list: {kernels}"
    )


@requires_rocm
def test_get_kernels_contains_vector_multiply(simple_binary):
    kdb = KernelDB(simple_binary)
    kernels = kdb.get_kernels()
    assert any("vector_multiply" in k for k in kernels), (
        f"'vector_multiply' not found in kernel list: {kernels}"
    )


@requires_rocm
def test_get_kernels_two_kernels(simple_binary):
    """The simple binary has exactly two kernels."""
    kdb = KernelDB(simple_binary)
    kernels = kdb.get_kernels()
    assert len(kernels) == 2, f"Expected 2 kernels, got {len(kernels)}: {kernels}"


# ---------------------------------------------------------------------------
# Source-line mapping
# ---------------------------------------------------------------------------


@requires_rocm
def test_get_kernel_lines_returns_list(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    lines = kdb.get_kernel_lines(kernel_name)
    assert isinstance(lines, list)


@requires_rocm
def test_get_kernel_lines_non_empty(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    lines = kdb.get_kernel_lines(kernel_name)
    assert len(lines) > 0, "Expected at least one source line in kernel"


@requires_rocm
def test_get_kernel_lines_are_positive_integers(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    for line in kdb.get_kernel_lines(kernel_name):
        assert isinstance(line, int)
        assert line > 0, f"Line number must be positive, got {line}"


@requires_rocm
def test_different_kernels_have_different_lines(simple_binary):
    """vector_add and vector_multiply live on different source lines."""
    kdb = KernelDB(simple_binary)
    add_name = _find_kernel(kdb.get_kernels(), "vector_add")
    mul_name = _find_kernel(kdb.get_kernels(), "vector_multiply")

    add_lines = set(kdb.get_kernel_lines(add_name))
    mul_lines = set(kdb.get_kernel_lines(mul_name))

    # They may share some lines (e.g., the common expression line) but not all
    assert add_lines != mul_lines, (
        "vector_add and vector_multiply reported identical line sets"
    )


# ---------------------------------------------------------------------------
# Instruction extraction
# ---------------------------------------------------------------------------


@requires_rocm
def test_get_instructions_for_line_returns_list(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    first_line = kdb.get_kernel_lines(kernel_name)[0]
    instructions = kdb.get_instructions_for_line(kernel_name, first_line)
    assert isinstance(instructions, list)


@requires_rocm
def test_get_instructions_for_line_non_empty(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    first_line = kdb.get_kernel_lines(kernel_name)[0]
    instructions = kdb.get_instructions_for_line(kernel_name, first_line)
    assert len(instructions) > 0


@requires_rocm
def test_instructions_have_disassembly_attribute(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    first_line = kdb.get_kernel_lines(kernel_name)[0]
    instructions = kdb.get_instructions_for_line(kernel_name, first_line)
    for inst in instructions:
        assert hasattr(inst, "disassembly")
        assert isinstance(inst.disassembly, str)
        assert len(inst.disassembly) > 0


@requires_rocm
def test_instructions_have_line_attribute(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    first_line = kdb.get_kernel_lines(kernel_name)[0]
    instructions = kdb.get_instructions_for_line(kernel_name, first_line)
    for inst in instructions:
        assert hasattr(inst, "line")
        assert isinstance(inst.line, int)


@requires_rocm
def test_instructions_have_column_attribute(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    first_line = kdb.get_kernel_lines(kernel_name)[0]
    instructions = kdb.get_instructions_for_line(kernel_name, first_line)
    for inst in instructions:
        assert hasattr(inst, "column")
        assert isinstance(inst.column, int)


@requires_rocm
def test_instructions_have_file_name_attribute(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    first_line = kdb.get_kernel_lines(kernel_name)[0]
    instructions = kdb.get_instructions_for_line(kernel_name, first_line)
    for inst in instructions:
        assert hasattr(inst, "file_name")


@requires_rocm
def test_instruction_line_numbers_match_queried_line(simple_binary):
    """Instructions returned for a line should report that same line number."""
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    for line in kdb.get_kernel_lines(kernel_name):
        for inst in kdb.get_instructions_for_line(kernel_name, line):
            assert inst.line == line, (
                f"Expected inst.line == {line}, got {inst.line}"
            )


# ---------------------------------------------------------------------------
# Instruction filtering
# ---------------------------------------------------------------------------


@requires_rocm
def test_filter_pattern_returns_subset(simple_binary):
    """Filtering instructions reduces or keeps the same count."""
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    all_instructions = []
    filtered_instructions = []
    for line in kdb.get_kernel_lines(kernel_name):
        all_instructions.extend(kdb.get_instructions_for_line(kernel_name, line))
        filtered_instructions.extend(
            kdb.get_instructions_for_line(kernel_name, line, ".*load.*")
        )
    assert len(filtered_instructions) <= len(all_instructions)


@requires_rocm
def test_filter_pattern_matches_disassembly(simple_binary):
    """Each instruction returned by a filter must match the pattern."""
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    pattern = ".*load.*"
    for line in kdb.get_kernel_lines(kernel_name):
        for inst in kdb.get_instructions_for_line(kernel_name, line, pattern):
            assert re.search(pattern, inst.disassembly, re.IGNORECASE), (
                f"Instruction {inst.disassembly!r} does not match pattern {pattern!r}"
            )


@requires_rocm
def test_load_store_filter_finds_memory_ops(simple_binary):
    """The vector_add kernel must have at least one load/store instruction."""
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    mem_ops = []
    for line in kdb.get_kernel_lines(kernel_name):
        mem_ops.extend(
            kdb.get_instructions_for_line(kernel_name, line, ".*(load|store).*")
        )
    assert len(mem_ops) > 0, "Expected at least one memory operation in vector_add"


@requires_rocm
def test_no_match_filter_returns_empty(simple_binary):
    """A filter that matches nothing must return an empty list per line."""
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    for line in kdb.get_kernel_lines(kernel_name):
        result = kdb.get_instructions_for_line(
            kernel_name, line, "THISDOESNOTEXISTINANYASM__xyz"
        )
        assert result == [], (
            f"Expected empty list for impossible filter, got {result}"
        )


# ---------------------------------------------------------------------------
# Basic blocks
# ---------------------------------------------------------------------------


@requires_rocm
def test_get_basic_blocks_returns_list(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    kernel = kdb.get_kernel(kernel_name)
    blocks = kernel.get_basic_blocks()
    assert isinstance(blocks, list)


@requires_rocm
def test_get_basic_blocks_non_empty(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    kernel = kdb.get_kernel(kernel_name)
    blocks = kernel.get_basic_blocks()
    assert len(blocks) > 0, "Expected at least one basic block in vector_add"


# ---------------------------------------------------------------------------
# Kernel wrapper (high-level API)
# ---------------------------------------------------------------------------


@requires_rocm
def test_kernel_wrapper_name(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    kernel = kdb.get_kernel(kernel_name)
    assert kernel.name == kernel_name


@requires_rocm
def test_kernel_wrapper_lines_property(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    kernel = kdb.get_kernel(kernel_name)
    assert isinstance(kernel.lines, list)
    assert len(kernel.lines) > 0
    for line in kernel.lines:
        assert isinstance(line, int) and line > 0


@requires_rocm
def test_kernel_wrapper_assembly_property(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    kernel = kdb.get_kernel(kernel_name)
    asm = kernel.assembly
    assert isinstance(asm, list)
    assert len(asm) > 0
    for line in asm:
        assert isinstance(line, str) and len(line) > 0


@requires_rocm
def test_kernel_wrapper_files_property(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    kernel = kdb.get_kernel(kernel_name)
    files = kernel.files
    assert isinstance(files, list)
    # Source files should be strings
    for f in files:
        assert isinstance(f, str)


@requires_rocm
def test_kernel_wrapper_signature_property(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    kernel = kdb.get_kernel(kernel_name)
    # signature is an alias for name
    assert isinstance(kernel.signature, str)
    assert len(kernel.signature) > 0


@requires_rocm
def test_kernel_wrapper_get_instructions_for_line(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    kernel = kdb.get_kernel(kernel_name)
    first_line = kernel.lines[0]
    instructions = kernel.get_instructions_for_line(first_line)
    assert isinstance(instructions, list)
    assert len(instructions) > 0


@requires_rocm
def test_kernel_wrapper_get_instructions_with_filter(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    kernel = kdb.get_kernel(kernel_name)
    all_insts = kernel.get_instructions_for_line(kernel.lines[0])
    filtered = kernel.get_instructions_for_line(kernel.lines[0], ".*load.*")
    assert len(filtered) <= len(all_insts)


@requires_rocm
def test_kernel_wrapper_get_file_name(simple_binary):
    kdb = KernelDB(simple_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "vector_add")
    result = kdb.get_file_name(kernel_name, 0)
    assert isinstance(result, str)
