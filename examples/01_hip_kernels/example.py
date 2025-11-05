#!/usr/bin/env python3
"""HIP Kernel Analysis Example - Analyze vector add and multiply kernels"""

import subprocess
import sys
import tempfile
from pathlib import Path
from kerneldb import KernelDB


HIP_CODE = """#include <hip/hip_runtime.h>
#include <iostream>

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

int main() {
    const int n = 1024;
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, n * sizeof(float));
    hipMalloc(&d_b, n * sizeof(float));
    hipMalloc(&d_c, n * sizeof(float));

    hipLaunchKernelGGL(vector_add, dim3(4), dim3(256), 0, 0, d_a, d_b, d_c, n);
    hipDeviceSynchronize();

    hipLaunchKernelGGL(vector_multiply, dim3(4), dim3(256), 0, 0, d_a, d_b, d_c, n);
    hipDeviceSynchronize();

    std::cout << "Kernels executed successfully" << std::endl;
    return 0;
}
"""


def main():
    print("HIP Kernel Analysis Example")
    print("="*80)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(HIP_CODE)
        cpp_file = Path(f.name)
    exe_file = cpp_file.with_suffix('')

    try:
        print("\n[1/2] Compiling HIP code with debug symbols...")
        result = subprocess.run(["hipcc", "-gline-tables-only", str(cpp_file), "-o", str(exe_file)],
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return 1
        print("Compilation successful")

        print("\n[2/2] Analyzing binary with kernelDB...")
        kdb = KernelDB(str(exe_file))
        kdb.analyze()

        kernels = kdb.get_kernels()
        print(f"Found {len(kernels)} kernel(s)")

        for kernel_name in kernels:
            print(f"\n{'='*80}")
            print(f"Kernel: {kernel_name}")
            print('='*80)

            kernel = kdb.get_kernel(kernel_name)
            lines = kdb.get_kernel_lines(kernel_name)

            print(f"\nSource lines: {len(lines)} (range: {min(lines)}-{max(lines)})")
            print(f"Basic blocks: {len(kernel.get_basic_blocks())}")

            print(f"\nInstructions by source line:")
            for line in lines:
                instructions = kdb.get_instructions_for_line(kernel_name, line)
                if instructions:
                    print(f"\n  Line {line}: {len(instructions)} instruction(s)")
                    for inst in instructions:
                        print(f"    [{inst.line}:{inst.column}] {inst.disassembly}")

            print(f"\nMemory operations (load/store):")
            mem_count = 0
            for line in lines:
                mem_ops = kdb.get_instructions_for_line(kernel_name, line, ".*(load|store).*")
                mem_count += len(mem_ops)
                for inst in mem_ops:
                    print(f"  {inst.disassembly}")
            print(f"Total: {mem_count} memory operations")

        print(f"\n{'='*80}")
        print("Analysis complete!")
        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    finally:
        cpp_file.unlink(missing_ok=True)
        exe_file.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
