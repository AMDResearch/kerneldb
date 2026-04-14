# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for KernelDB lazy loading path: add_file(path, lazy=True) should
index kernel names without disassembly, then load on demand and produce
results identical to eager loading.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

from conftest import requires_rocm

kerneldb = pytest.importorskip("kerneldb", reason="kernelDB C++ extension not available")
KernelDB = kerneldb.KernelDB

_HIP_SOURCE = r"""
#include <hip/hip_runtime.h>

__global__ void lazy_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void lazy_mul(float* a, float* b, float* c, int n) {
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
    tmp = Path(tempfile.mkdtemp(prefix="kerneldb_lazy_test_"))
    src = tmp / "lazy.cpp"
    exe = tmp / "lazy"
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


@requires_rocm
def test_lazy_kernel_discovery():
    """Lazy add_file should discover the same kernels as eager loading."""
    path = _get_binary()

    eager = KernelDB(path)
    eager_kernels = sorted(eager.get_kernels())

    lazy = KernelDB(lazy=True)
    lazy.add_file(path, lazy=True)
    lazy_kernels = sorted(lazy.get_kernels())

    assert lazy_kernels == eager_kernels


@requires_rocm
def test_lazy_has_kernel():
    """has_kernel() should return True for lazy-indexed kernels before loading."""
    path = _get_binary()
    kdb = KernelDB(lazy=True)
    kdb.add_file(path, lazy=True)

    kernels = kdb.get_kernels()
    assert len(kernels) >= 2

    for name in kernels:
        assert kdb.has_kernel(name), f"has_kernel({name!r}) returned False"


@requires_rocm
def test_lazy_get_kernel_matches_eager():
    """Lazy-loaded kernel assembly should match eager-loaded assembly."""
    path = _get_binary()

    eager = KernelDB(path)
    eager_kernel = eager.get_kernel(_find_kernel(eager.get_kernels(), "lazy_add"))

    lazy = KernelDB(lazy=True)
    lazy.add_file(path, lazy=True)
    lazy_kernel = lazy.get_kernel(_find_kernel(lazy.get_kernels(), "lazy_add"))

    assert lazy_kernel.assembly == eager_kernel.assembly
    assert lazy_kernel.lines == eager_kernel.lines


@requires_rocm
def test_lazy_instructions_for_line():
    """Instructions retrieved via lazy loading should match eager loading."""
    path = _get_binary()

    eager = KernelDB(path)
    eager_name = _find_kernel(eager.get_kernels(), "lazy_add")
    eager_lines = eager.get_kernel_lines(eager_name)

    lazy = KernelDB(lazy=True)
    lazy.add_file(path, lazy=True)
    lazy_name = _find_kernel(lazy.get_kernels(), "lazy_add")
    lazy_lines = lazy.get_kernel_lines(lazy_name)

    assert lazy_lines == eager_lines

    for line in eager_lines:
        eager_insts = [i.disassembly for i in eager.get_instructions_for_line(eager_name, line)]
        lazy_insts = [i.disassembly for i in lazy.get_instructions_for_line(lazy_name, line)]
        assert lazy_insts == eager_insts, f"Mismatch at line {line}"
