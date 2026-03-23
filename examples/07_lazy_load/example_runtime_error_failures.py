#!/usr/bin/env python3
"""Show concrete examples of RuntimeError when get_kernel(elf_name) fails (ELF vs disassembly name mismatch)."""

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
    kdb = KernelDB(lazy=True)
    kdb.add_file(get_torch_hip_lib())
    kernels = kdb.get_kernels()

    max_check = 50
    max_show = 5
    print(f"Checking first {max_check} kernel names for RuntimeError...\n")
    failures = []
    for name in list(kernels)[:max_check]:
        try:
            kdb.get_kernel(name)
        except RuntimeError as e:
            failures.append((name, str(e)))
    print(f"Failures: {len(failures)} / {max_check}\n")
    for i, (name, err) in enumerate(failures[:max_show]):
        print(f"--- Failure {i + 1} ---")
        print(f"ELF name: {name}")
        print(f"Error: {err}\n")


if __name__ == "__main__":
    main()
