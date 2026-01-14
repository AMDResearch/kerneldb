#!/usr/bin/env python3
"""Typedef Resolution Example - Show typedef/using alias resolution"""

import subprocess
import sys
import tempfile
from pathlib import Path
from kerneldb import KernelDB


HIP_CODE = """#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>  // for bfloat16
#include <iostream>

// Simple type aliases
using MyInt = int;
using MyFloat = float;
using MyDouble = double;

// Nested type aliases (demonstrates chain resolution)
using Level1 = int;
using Level2 = Level1;
using Level3 = Level2;
using Level4 = Level3;

// Pointer aliases
using IntPtr = int*;
using FloatPtr = float*;

// Traditional typedef
typedef double MyTypedefDouble;

// HIP-specific type alias
using bfloat16 = __hip_bfloat16;

__global__ void typedef_kernel(
    MyInt a,           // Simple alias: MyInt -> int
    MyFloat b,         // Simple alias: MyFloat -> float
    Level4 c,          // Nested alias: Level4 -> Level3 -> Level2 -> Level1 -> int
    IntPtr d,          // Pointer alias: IntPtr -> int*
    MyTypedefDouble e, // Traditional typedef: MyTypedefDouble -> double
    bfloat16 g,        // HIP type alias: bfloat16 -> __hip_bfloat16
    unsigned int f     // Not typedefed - shows unchanged
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Dummy computation
    if (idx == 0) {
        d[0] = a + static_cast<int>(b) + c + static_cast<int>(e) + f;
    }
}

int main() {
    MyInt h_a = 10;
    MyFloat h_b = 3.14f;
    Level4 h_c = 42;
    int temp = 0;
    IntPtr h_d = &temp;
    MyTypedefDouble h_e = 2.71;
    bfloat16 h_g = bfloat16(1.5f);
    unsigned int h_f = 100;

    hipLaunchKernelGGL(typedef_kernel, dim3(1), dim3(256), 0, 0,
                       h_a, h_b, h_c, h_d, h_e, h_g, h_f);
    hipDeviceSynchronize();

    std::cout << "Kernel executed successfully" << std::endl;
    return 0;
}
"""


def main():
    print("Typedef Resolution Example")
    print("="*80)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(HIP_CODE)
        cpp_file = Path(f.name)
    exe_file = cpp_file.with_suffix('')

    try:
        print("\n[1/3] Compiling HIP code with debug symbols...")
        result = subprocess.run(
            ["hipcc", "-g", str(cpp_file), "-o", str(exe_file)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return 1
        print("✓ Compilation successful")

        print("\n[2/3] Analyzing with resolve_typedefs=False (default)...")
        print("     (Shows typedef/using names as written in source code)")
        kdb = KernelDB(str(exe_file))

        for kernel_name in kdb.get_kernels():
            print(f"\nKernel: {kernel_name}")

            # Get arguments with typedef names preserved
            arguments = kdb.get_kernel_arguments(kernel_name, resolve_typedefs=False)

            if not arguments:
                print("  No argument information available")
                continue

            print(f"\n  Arguments ({len(arguments)}):")
            for arg in arguments:
                print(f"    {arg.name}: {arg.type_name} (size={arg.size}, align={arg.alignment})")

        print("\n[3/3] Analyzing with resolve_typedefs=True...")

        for kernel_name in kdb.get_kernels():
            # Get arguments with typedefs resolved
            arguments_resolved = kdb.get_kernel_arguments(kernel_name, resolve_typedefs=True)

            print(f"\n  Arguments ({len(arguments_resolved)}):")
            for arg in arguments_resolved:
                print(f"    {arg.name}: {arg.type_name} (size={arg.size}, align={arg.alignment})")

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cpp_file.unlink(missing_ok=True)
        exe_file.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
