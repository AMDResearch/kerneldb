#!/usr/bin/env python3
"""Kernel Arguments Example - Display kernel parameter information"""

import subprocess
import sys
import tempfile
from pathlib import Path
from kerneldb import KernelDB


HIP_CODE = """#include <hip/hip_runtime.h>
#include <iostream>

__global__ void vector_add(double* a, double* b, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int n = 1024;
    double *d_a, *d_b, *d_c;
    hipMalloc(&d_a, n * sizeof(double));
    hipMalloc(&d_b, n * sizeof(double));
    hipMalloc(&d_c, n * sizeof(double));

    hipLaunchKernelGGL(vector_add, dim3(4), dim3(256), 0, 0, d_a, d_b, d_c, n);
    hipDeviceSynchronize();

    std::cout << "Kernel executed successfully" << std::endl;
    return 0;
}
"""


def main():
    print("Kernel Arguments Example")
    print("="*80)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(HIP_CODE)
        cpp_file = Path(f.name)
    exe_file = cpp_file.with_suffix('')

    try:
        print("\n[1/2] Compiling HIP code with debug symbols...")
        result = subprocess.run(
            ["hipcc", "-g", str(cpp_file), "-o", str(exe_file)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return 1
        print("Compilation successful")

        print("\n[2/2] Analyzing kernel arguments...")
        kdb = KernelDB(str(exe_file))

        for kernel_name in kdb.get_kernels():
            print(f"\nKernel: {kernel_name}")
            arguments = kdb.get_kernel_arguments(kernel_name)

            if not arguments:
                print("  No argument information available")
                continue

            print(f"  Arguments ({len(arguments)}):")
            for arg in arguments:
                print(f"    {arg.name}: {arg.type_name} (size={arg.size}, align={arg.alignment})")

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    finally:
        cpp_file.unlink(missing_ok=True)
        exe_file.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())

