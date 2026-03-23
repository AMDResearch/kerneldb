#!/usr/bin/env python3
"""Lazy-load example: analyze torch's HIP library with KernelDB (lazy=True)."""

import sys
import time
from pathlib import Path

import torch
from kerneldb import KernelDB


def get_torch_hip_lib():
    lib_dir = Path(torch.__file__).resolve().parent / "lib"
    for name in ("libtorch_hip.so", "libtorch.so"):
        p = lib_dir / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No torch HIP lib in {lib_dir}")


def main():
    print("Lazy-load KernelDB Example (torch lib)")
    print("=" * 80)

    torch_lib = get_torch_hip_lib()
    print(f"Torch lib: {torch_lib}")

    print("\nAnalyzing with KernelDB (lazy=True)...")
    kdb = KernelDB(lazy=True)

    t0 = time.perf_counter()
    ok = kdb.add_file(torch_lib)
    t_add = time.perf_counter() - t0
    if not ok:
        print("add_file failed")
        return 1
    print(f"  add_file (index only): {t_add:.2f} s")

    kernels = kdb.get_kernels()
    print(f"  get_kernels(): {len(kernels)} names")
    print(f"Found {len(kernels)} kernel(s) (showing first 3)")

    shown = 0
    for kernel_name in kernels:
        if shown >= 3:
            break
        try:
            kernel = kdb.get_kernel(kernel_name)
        except RuntimeError:
            continue  # ELF name may not match disassembly name
        shown += 1
        print(f"\n{'='*80}")
        print(f"Kernel: {kernel_name}")
        print("=" * 80)

        asm_lines = []
        for block in kernel.get_basic_blocks():
            for inst in block.get_instructions():
                if inst.disassembly:
                    asm_lines.append(inst.disassembly)
        max_asm = 25
        print(f"\nDisassembly (first {min(max_asm, len(asm_lines))} of {len(asm_lines)} lines):")
        for ln in asm_lines[:max_asm]:
            print(f"  {ln}")
        if len(asm_lines) > max_asm:
            print(f"  ... ({len(asm_lines) - max_asm} more lines)")

        lines = kdb.get_kernel_lines(kernel_name)
        if not lines:
            print("\n(no source line mapping)")
            continue

        print(f"\nSource lines: {len(lines)} (range: {min(lines)}-{max(lines)})")
        print(f"Basic blocks: {len(kernel.get_basic_blocks())}")

        print("\nInstructions by source line:")
        for line in lines:
            instructions = kdb.get_instructions_for_line(kernel_name, line)
            if instructions:
                print(f"\n  Line {line}: {len(instructions)} instruction(s)")
                for inst in instructions:
                    print(f"    [{inst.line}:{inst.column}] {inst.disassembly}")

        print("\nMemory operations (load/store):")
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


if __name__ == "__main__":
    sys.exit(main())
