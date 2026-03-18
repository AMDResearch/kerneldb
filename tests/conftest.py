# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

"""
Shared pytest fixtures and helpers for kernelDB tests.

Tests in this suite require a ROCm environment with:
  - hipcc compiler (to compile HIP test kernels)
  - An HSA-compatible GPU (AMD CDNA architecture)

Tests are automatically skipped if these requirements are not met.
"""

import shutil
import subprocess
import sys
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Environment detection helpers
# ---------------------------------------------------------------------------


def _hipcc_available() -> bool:
    """Return True if the hipcc compiler is on PATH."""
    return shutil.which("hipcc") is not None


def _gpu_available() -> bool:
    """Return True if kernelDB can initialise HSA and find a GPU agent."""
    try:
        from kerneldb import _kerneldb  # type: ignore[import]

        status = _kerneldb.hsa_init()
        if status != 0:
            return False
        agent = _kerneldb.get_first_gpu_agent()
        return agent.handle != 0
    except (ImportError, AttributeError):
        return False
    except RuntimeError:
        return False


def _compile_hip(src_path: Path, out_path: Path, extra_flags: list[str] | None = None) -> None:
    """Compile *src_path* with hipcc and write the executable to *out_path*.

    Raises ``pytest.skip`` on compilation failure so individual tests can
    choose how to handle the case.
    """
    cmd = ["hipcc", "-g", str(src_path), "-o", str(out_path)]
    if extra_flags:
        cmd.extend(extra_flags)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"HIP compilation failed:\n{result.stderr}")


# ---------------------------------------------------------------------------
# pytest marks / skip decorators
# ---------------------------------------------------------------------------

requires_hipcc = pytest.mark.skipif(
    not _hipcc_available(),
    reason="hipcc not available – ROCm toolchain required",
)

requires_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason="No HSA GPU agent found – AMD GPU with ROCm required",
)

requires_rocm = pytest.mark.skipif(
    not (_hipcc_available() and _gpu_available()),
    reason="ROCm environment (hipcc + GPU) required",
)


# ---------------------------------------------------------------------------
# HIP source snippets used across multiple test modules
# ---------------------------------------------------------------------------

SIMPLE_KERNELS_SOURCE = r"""
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

ARGUMENTS_SOURCE = r"""
#include <hip/hip_runtime.h>

__global__ void kernel_with_args(double* a, double* b, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() { return 0; }
"""

NESTED_STRUCTS_SOURCE = r"""
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

TYPEDEF_SOURCE = r"""
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

TEMPLATE_SOURCE = r"""
#include <hip/hip_runtime.h>

template<typename T>
__global__ void scale_values(T* input, T* output, T factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * factor;
    }
}

// Explicit instantiations so the kernels appear in the binary
template __global__ void scale_values<float>(float*, float*, float, int);
template __global__ void scale_values<double>(double*, double*, double, int);

int main() { return 0; }
"""


# ---------------------------------------------------------------------------
# Session-scoped compiled binary fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def simple_binary(tmp_path_factory):
    """Path to a compiled HIP binary containing vector_add and vector_multiply."""
    if not _hipcc_available():
        pytest.skip("hipcc not available")
    if not _gpu_available():
        pytest.skip("No GPU agent found")

    src_dir = tmp_path_factory.mktemp("hip_simple")
    cpp_file = src_dir / "simple_kernels.cpp"
    exe_file = src_dir / "simple_kernels"
    cpp_file.write_text(SIMPLE_KERNELS_SOURCE)
    _compile_hip(cpp_file, exe_file)
    return str(exe_file)


@pytest.fixture(scope="session")
def arguments_binary(tmp_path_factory):
    """Path to a compiled HIP binary with typed kernel arguments."""
    if not _hipcc_available():
        pytest.skip("hipcc not available")
    if not _gpu_available():
        pytest.skip("No GPU agent found")

    src_dir = tmp_path_factory.mktemp("hip_args")
    cpp_file = src_dir / "args_kernel.cpp"
    exe_file = src_dir / "args_kernel"
    cpp_file.write_text(ARGUMENTS_SOURCE)
    _compile_hip(cpp_file, exe_file)
    return str(exe_file)


@pytest.fixture(scope="session")
def nested_structs_binary(tmp_path_factory):
    """Path to a compiled HIP binary with nested struct kernel arguments."""
    if not _hipcc_available():
        pytest.skip("hipcc not available")
    if not _gpu_available():
        pytest.skip("No GPU agent found")

    src_dir = tmp_path_factory.mktemp("hip_nested")
    cpp_file = src_dir / "nested_structs.cpp"
    exe_file = src_dir / "nested_structs"
    cpp_file.write_text(NESTED_STRUCTS_SOURCE)
    _compile_hip(cpp_file, exe_file)
    return str(exe_file)


@pytest.fixture(scope="session")
def typedef_binary(tmp_path_factory):
    """Path to a compiled HIP binary with typedef/using-alias kernel arguments."""
    if not _hipcc_available():
        pytest.skip("hipcc not available")
    if not _gpu_available():
        pytest.skip("No GPU agent found")

    src_dir = tmp_path_factory.mktemp("hip_typedef")
    cpp_file = src_dir / "typedef_kernel.cpp"
    exe_file = src_dir / "typedef_kernel"
    cpp_file.write_text(TYPEDEF_SOURCE)
    _compile_hip(cpp_file, exe_file)
    return str(exe_file)


@pytest.fixture(scope="session")
def template_binary(tmp_path_factory):
    """Path to a compiled HIP binary with template kernel instantiations."""
    if not _hipcc_available():
        pytest.skip("hipcc not available")
    if not _gpu_available():
        pytest.skip("No GPU agent found")

    src_dir = tmp_path_factory.mktemp("hip_template")
    cpp_file = src_dir / "template_kernel.cpp"
    exe_file = src_dir / "template_kernel"
    cpp_file.write_text(TEMPLATE_SOURCE)
    _compile_hip(cpp_file, exe_file)
    return str(exe_file)
