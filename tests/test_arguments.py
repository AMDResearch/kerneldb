# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for KernelDB kernel argument extraction: basic metadata, nested structs,
typedef resolution, and template instantiations.

All tests require a ROCm environment with hipcc and a GPU.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

from conftest import requires_rocm

kerneldb = pytest.importorskip("kerneldb", reason="kernelDB C++ extension not available")
KernelDB = kerneldb.KernelDB


def _compile(source, name):
    tmp = Path(tempfile.mkdtemp(prefix=f"kerneldb_{name}_"))
    src = tmp / f"{name}.cpp"
    exe = tmp / name
    src.write_text(source)
    r = subprocess.run(["hipcc", "-g", str(src), "-o", str(exe)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"hipcc compilation failed:\n{r.stderr}")
    return str(exe)


def _find_kernel(kernels, fragment):
    matches = [k for k in kernels if fragment in k]
    if not matches:
        pytest.skip(f"No kernel containing {fragment!r} found")
    return matches[0]


def _arg_by_name(arguments, name):
    return next((a for a in arguments if a.name == name), None)


_ARGS_SOURCE = r"""
#include <hip/hip_runtime.h>

__global__ void kernel_with_args(double* a, double* b, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() { return 0; }
"""

_NESTED_SOURCE = r"""
#include <hip/hip_runtime.h>

struct Point3D {
    float x;
    float y;
    float z;
};

struct BoundingBox {
    Point3D min_pt;
    Point3D max_pt;
};

__global__ void update_bounds(BoundingBox* boxes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        boxes[idx].min_pt.x = 0.0f;
    }
}

int main() { return 0; }
"""

_TYPEDEF_SOURCE = r"""
#include <hip/hip_runtime.h>

using MyInt = int;
using MyFloat = float;
typedef double MyDouble;

__global__ void typedef_kernel(MyInt a, MyFloat b, MyDouble c, int* d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        d[0] = a + static_cast<int>(b) + static_cast<int>(c);
    }
}

int main() { return 0; }
"""

_TEMPLATE_SOURCE = r"""
#include <hip/hip_runtime.h>

template<typename T>
__global__ void scale_values(T* input, T* output, T factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * factor;
    }
}

template __global__ void scale_values<float>(float*, float*, float, int);
template __global__ void scale_values<double>(double*, double*, double, int);

int main() { return 0; }
"""

_cache = {}


def _get_binary(source, name):
    if name not in _cache:
        _cache[name] = _compile(source, name)
    return _cache[name]


@requires_rocm
def test_kernel_arguments():
    kdb = KernelDB(_get_binary(_ARGS_SOURCE, "args"))
    name = _find_kernel(kdb.get_kernels(), "kernel_with_args")
    args = kdb.get_kernel_arguments(name)

    assert isinstance(args, list) and len(args) == 4
    for arg in args:
        assert isinstance(arg.name, str) and arg.name
        assert isinstance(arg.type_name, str) and arg.type_name
        assert isinstance(arg.size, int) and arg.size > 0
        assert isinstance(arg.alignment, int) and arg.alignment > 0

    for expected in ("a", "b", "c", "n"):
        assert expected in [a.name for a in args]

    assert _arg_by_name(args, "a").size == 8  # pointer = 8 bytes
    n_arg = _arg_by_name(args, "n")
    assert n_arg.size == 4
    assert "int" in n_arg.type_name.lower()


@requires_rocm
def test_kernel_wrapper_arguments():
    kdb = KernelDB(_get_binary(_ARGS_SOURCE, "args"))
    name = _find_kernel(kdb.get_kernels(), "kernel_with_args")
    kernel = kdb.get_kernel(name)

    assert kernel.has_arguments() is True
    assert isinstance(kernel.arguments, list) and kernel.arguments


@requires_rocm
def test_nested_struct_arguments():
    kdb = KernelDB(_get_binary(_NESTED_SOURCE, "nested"))
    name = _find_kernel(kdb.get_kernels(), "update_bounds")
    args = kdb.get_kernel_arguments(name)

    for arg in args:
        assert hasattr(arg, "members")

    def find_type(arg_list, fragment):
        for arg in arg_list:
            if fragment in arg.type_name:
                return arg
            if arg.members:
                found = find_type(arg.members, fragment)
                if found:
                    return found
        return None

    point = find_type(args, "Point3D")
    assert point is not None, "Point3D not found in argument tree"
    assert len(point.members) == 3


@requires_rocm
def test_typedef_resolution():
    kdb = KernelDB(_get_binary(_TYPEDEF_SOURCE, "typedef"))
    name = _find_kernel(kdb.get_kernels(), "typedef_kernel")

    args_raw = kdb.get_kernel_arguments(name, resolve_typedefs=False)
    assert len(args_raw) == 4
    raw_types = [a.type_name for a in args_raw]
    assert any(n in ("MyInt", "MyFloat", "MyDouble") for n in raw_types)

    args_resolved = kdb.get_kernel_arguments(name, resolve_typedefs=True)
    resolved_types = [a.type_name for a in args_resolved]
    for alias in ("MyInt", "MyFloat", "MyDouble"):
        assert alias not in resolved_types

    primitives = {"int", "float", "double", "unsigned int"}
    for t in resolved_types:
        base = t.replace("*", "").strip()
        assert any(p in base for p in primitives), f"{t!r} doesn't look primitive"


@requires_rocm
def test_template_kernel_arguments():
    kdb = KernelDB(_get_binary(_TEMPLATE_SOURCE, "template"))
    kernels = kdb.get_kernels()

    scale_kernels = [k for k in kernels if "scale_values" in k]
    assert len(scale_kernels) >= 2

    for type_frag in ("float", "double"):
        matching = [k for k in scale_kernels if type_frag in k]
        if not matching:
            pytest.skip(f"scale_values<{type_frag}> not found")
        args = kdb.get_kernel_arguments(matching[0])
        assert len(args) == 4
        non_int = [a for a in args if a.name != "n"]
        assert all(type_frag in a.type_name.lower() for a in non_int)
