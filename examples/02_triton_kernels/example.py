#!/usr/bin/env python3
"""Triton Kernel Analysis Example - Analyze vector add and multiply Triton kernels"""

import sys
import tempfile
import subprocess
import time
from pathlib import Path
from kerneldb import KernelDB


TRITON_SCRIPT = """import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)

@triton.jit
def mul_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(out_ptr + offsets, output, mask=mask)

if __name__ == "__main__":
    size = 1024
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')

    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    add_result = torch.empty_like(x)
    add_kernel[grid](x, y, add_result, size, BLOCK_SIZE=1024)

    mul_result = torch.empty_like(x)
    mul_kernel[grid](x, y, mul_result, size, BLOCK_SIZE=1024)

    print("Triton kernels executed")
"""


def main():
    print("Triton Kernel Analysis Example")
    print("="*80)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(TRITON_SCRIPT)
        triton_file = Path(f.name)

    try:
        print("\n[1/3] Running Triton script to compile kernels...")
        result = subprocess.run(["python3", str(triton_file)],
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return 1
        print(f"{result.stdout.strip()}")

        print("\n[2/3] Searching Triton cache for compiled kernels...")
        triton_cache = Path.home() / ".triton" / "cache"

        code_objects = []
        if triton_cache.exists():
            for pattern in ["**/*.hsaco", "**/*.so"]:
                code_objects.extend(triton_cache.glob(pattern))

        code_objects.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        recent_objects = code_objects[:3]

        if not recent_objects:
            print("No code objects found in Triton cache")
            return 1

        print(f"Found {len(code_objects)} objects, analyzing {len(recent_objects)} most recent")

        print("\n[3/3] Analyzing kernels with kernelDB...")
        all_kernels = {}

        for code_obj in recent_objects:
            try:
                kdb = KernelDB(str(code_obj))
                kdb.analyze()
                kernel_names = kdb.get_kernels()

                for name in kernel_names:
                    if "_amd_" not in name.lower():
                        all_kernels[name] = (kdb, code_obj)
            except:
                continue

        if not all_kernels:
            print("No analyzable kernels found")
            return 1

        print(f"Found {len(all_kernels)} kernel(s)\n")

        for idx, (kernel_name, (kdb, code_obj)) in enumerate(all_kernels.items(), 1):
            print(f"{'='*80}")
            print(f"Kernel #{idx}: {kernel_name}")
            print(f"Source: {code_obj.name}")
            print('='*80)

            kernel = kdb.get_kernel(kernel_name)
            lines = kdb.get_kernel_lines(kernel_name)

            if lines:
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

            if kernel.files:
                print(f"\nSource files:")
                for file in kernel.files:
                    print(f"  - {file}")

        print(f"\n{'='*80}")
        print("Analysis complete!")
        return 0

    except ImportError as e:
        print(f"ERROR: {e}")
        print("Install: pip install torch triton")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    finally:
        triton_file.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
