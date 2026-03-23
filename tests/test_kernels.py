# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for KernelDB kernel-level operations: discovery, source-line mapping,
instruction extraction, filtering, basic blocks, and the Kernel wrapper.

All tests require a ROCm environment with hipcc and a GPU.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from conftest import requires_rocm

kerneldb = pytest.importorskip("kerneldb", reason="kernelDB C++ extension not available")
KernelDB = kerneldb.KernelDB

# -- HIP source compiled once per module ------------------------------------

_HIP_SOURCE = r"""
#include <hip/hip_runtime.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_multiply(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

int main() { return 0; }
"""

_binary_path = None


def _get_binary():
    global _binary_path
    if _binary_path is not None:
        return _binary_path
    tmp = Path(tempfile.mkdtemp(prefix="kerneldb_test_"))
    src = tmp / "simple.cpp"
    exe = tmp / "simple"
    src.write_text(_HIP_SOURCE)
    r = subprocess.run(["hipcc", "-g", str(src), "-o", str(exe)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"hipcc compilation failed:\n{r.stderr}")
    _binary_path = str(exe)
    return _binary_path


def _find_kernel(kernels, fragment):
    matches = [k for k in kernels if fragment in k]
    if not matches:
        pytest.skip(f"No kernel containing {fragment!r} found")
    return matches[0]


# -- Tests ------------------------------------------------------------------


@requires_rocm
def test_kernel_discovery():
    kdb = KernelDB(_get_binary())
    kernels = kdb.get_kernels()

    assert isinstance(kernels, list)
    assert len(kernels) == 2
    assert all(isinstance(k, str) and k for k in kernels)
    assert any("vector_add" in k for k in kernels)
    assert any("vector_multiply" in k for k in kernels)


@requires_rocm
def test_kernel_lines():
    kdb = KernelDB(_get_binary())
    kernels = kdb.get_kernels()

    for name in kernels:
        lines = kdb.get_kernel_lines(name)
        assert isinstance(lines, list) and lines, f"No lines for {name}"
        assert all(isinstance(ln, int) and ln >= 0 for ln in lines)

    add = _find_kernel(kernels, "vector_add")
    mul = _find_kernel(kernels, "vector_multiply")
    assert set(kdb.get_kernel_lines(add)) != set(kdb.get_kernel_lines(mul))


@requires_rocm
def test_instructions():
    kdb = KernelDB(_get_binary())
    name = _find_kernel(kdb.get_kernels(), "vector_add")

    for line in kdb.get_kernel_lines(name):
        for inst in kdb.get_instructions_for_line(name, line):
            assert isinstance(inst.disassembly, str) and inst.disassembly
            assert isinstance(inst.line, int) and inst.line == line
            assert isinstance(inst.column, int)
            assert hasattr(inst, "file_name")


@requires_rocm
def test_instruction_filtering():
    kdb = KernelDB(_get_binary())
    name = _find_kernel(kdb.get_kernels(), "vector_add")
    pattern = ".*(load|store).*"

    all_insts, mem_ops = [], []
    for line in kdb.get_kernel_lines(name):
        all_insts.extend(kdb.get_instructions_for_line(name, line))
        mem_ops.extend(kdb.get_instructions_for_line(name, line, pattern))

    assert len(mem_ops) > 0, "vector_add must have at least one load/store"
    assert len(mem_ops) <= len(all_insts)
    assert all(re.search(pattern, i.disassembly, re.IGNORECASE) for i in mem_ops)

    for line in kdb.get_kernel_lines(name):
        assert kdb.get_instructions_for_line(name, line, "NOMATCH__xyz") == []


@requires_rocm
def test_basic_blocks_and_kernel_wrapper():
    kdb = KernelDB(_get_binary())
    name = _find_kernel(kdb.get_kernels(), "vector_add")
    kernel = kdb.get_kernel(name)

    assert isinstance(kernel.get_basic_blocks(), list) and kernel.get_basic_blocks()
    assert kernel.name == name
    assert isinstance(kernel.signature, str) and kernel.signature
    assert isinstance(kernel.lines, list) and all(ln >= 0 for ln in kernel.lines)
    assert isinstance(kernel.assembly, list) and all(isinstance(s, str) for s in kernel.assembly)
    assert isinstance(kernel.files, list) and all(isinstance(f, str) for f in kernel.files)

    first_line = kernel.lines[0]
    insts = kernel.get_instructions_for_line(first_line)
    assert isinstance(insts, list) and insts
    filtered = kernel.get_instructions_for_line(first_line, ".*load.*")
    assert len(filtered) <= len(insts)

    assert isinstance(kdb.get_file_name(name, 0), str)
