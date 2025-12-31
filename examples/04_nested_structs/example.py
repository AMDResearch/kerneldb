#!/usr/bin/env python3
"""Nested Structs Example - Display nested struct kernel arguments"""

import subprocess
import sys
import tempfile
from pathlib import Path
from kerneldb import KernelDB


HIP_CODE = """#include <hip/hip_runtime.h>
#include <iostream>

// Nested struct definitions
struct Point3D {
    float x;
    float y;
    float z;
};

struct BoundingBox {
    Point3D min;
    Point3D max;
};

struct Particle {
    Point3D position;
    Point3D velocity;
    float mass;
    int id;
};

// Kernel with nested struct arguments
__global__ void update_particles(Particle* particles, BoundingBox bounds, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // Clamp particle position within bounds
        particles[idx].position.x = fmaxf(bounds.min.x, fminf(particles[idx].position.x, bounds.max.x));
        particles[idx].position.y = fmaxf(bounds.min.y, fminf(particles[idx].position.y, bounds.max.y));
        particles[idx].position.z = fmaxf(bounds.min.z, fminf(particles[idx].position.z, bounds.max.z));
    }
}

int main() {
    const int n = 256;

    Particle *d_particles;
    hipMalloc(&d_particles, n * sizeof(Particle));
    BoundingBox bounds;
    hipLaunchKernelGGL(update_particles, dim3(1), dim3(256), 0, 0, d_particles, bounds, n);

    hipDeviceSynchronize();
    std::cout << "Kernel executed successfully" << std::endl;
    return 0;
}
"""


def print_struct_members(members, indent=4):
    """Recursively print struct members"""
    if not members:
        return

    indent_str = " " * indent
    for member in members:
        print(f"{indent_str}{member.name}: {member.type_name} (size={member.size}, offset={member.offset})")
        if member.members:
            print(f"{indent_str}  Members:")
            print_struct_members(member.members, indent + 4)


def main():
    print("Nested Structs Example")
    print("="*80)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(HIP_CODE)
        cpp_file = Path(f.name)
    exe_file = cpp_file.with_suffix('')

    try:
        print("\n[1/2] Compiling HIP code with debug symbols...")
        # Use -O0 to disable optimizations and get better debug info for templates
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

        kernels = kdb.get_kernels()
        print(f"Found {len(kernels)} kernel(s)\n")

        for kernel_name in kernels:
            print(f"\n{'='*80}")
            print(f"Kernel: {kernel_name}")
            print('='*80)

            arguments = kdb.get_kernel_arguments(kernel_name)

            if not arguments:
                print("  No argument information available")
                continue

            print(f"\nArguments ({len(arguments)}):")
            for i, arg in enumerate(arguments, 1):
                print(f"\n  [{i}] {arg.name}: {arg.type_name}")
                print(f"      Size: {arg.size} bytes, Alignment: {arg.alignment} bytes")

                if arg.members:
                    print(f"      Struct members:")
                    print_struct_members(arg.members, indent=8)

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

