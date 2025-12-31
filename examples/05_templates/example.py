#!/usr/bin/env python3
"""Template Kernels Example - Discover template instantiations"""

import subprocess
import sys
import tempfile
from pathlib import Path
from kerneldb import KernelDB


HIP_CODE = """#include <hip/hip_runtime.h>
#include <iostream>

// Template kernel
template<typename T>
__global__ void scale_values(T* input, T* output, T factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * factor;
    }
}

int main() {
    const int n = 256;

    // Instantiate with float
    float *d_float_in, *d_float_out;
    hipMalloc(&d_float_in, n * sizeof(float));
    hipMalloc(&d_float_out, n * sizeof(float));
    hipLaunchKernelGGL((scale_values<float>), dim3(1), dim3(256), 0, 0,
                       d_float_in, d_float_out, 2.0f, n);

    // Instantiate with double
    double *d_double_in, *d_double_out;
    hipMalloc(&d_double_in, n * sizeof(double));
    hipMalloc(&d_double_out, n * sizeof(double));
    hipLaunchKernelGGL((scale_values<double>), dim3(1), dim3(256), 0, 0,
                       d_double_in, d_double_out, 2.0, n);

    // Instantiate with int
    int *d_int_in, *d_int_out;
    hipMalloc(&d_int_in, n * sizeof(int));
    hipMalloc(&d_int_out, n * sizeof(int));
    hipLaunchKernelGGL((scale_values<int>), dim3(1), dim3(256), 0, 0,
                       d_int_in, d_int_out, 2, n);

    hipDeviceSynchronize();
    std::cout << "Template kernels executed successfully" << std::endl;
    return 0;
}
"""


def main():
    print("Template Kernels Example")
    print("="*80)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(HIP_CODE)
        cpp_file = Path(f.name)
    exe_file = cpp_file.with_suffix('')

    try:
        print("\n[1/2] Compiling HIP code...")
        result = subprocess.run(
            ["hipcc", "-g", str(cpp_file), "-o", str(exe_file)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return 1
        print("Compilation successful")

        print("\n[2/2] Analyzing template instantiations...")
        kdb = KernelDB(str(exe_file))

        kernels = kdb.get_kernels()
        template_kernels = [k for k in kernels if "scale_values" in k]

        print(f"Found {len(template_kernels)} template instantiation(s)\n")

        for kernel_name in sorted(template_kernels):
            print(f"Kernel: {kernel_name}")

            arguments = kdb.get_kernel_arguments(kernel_name)
            if arguments:
                print(f"  Arguments ({len(arguments)}):")
                for arg in arguments:
                    print(f"    {arg.name}: {arg.type_name} (size={arg.size}, align={arg.alignment})")
            else:
                print(f"  No argument information available")
            print()

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    finally:
        cpp_file.unlink(missing_ok=True)
        exe_file.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())

