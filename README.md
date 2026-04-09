# KernelDB

A library for querying data within AMD CDNA kernel binaries. KernelDB loads HSACO and HIP fat binary files, disassembles the kernels, and uses DWARF debug info to map source lines to ISA instructions.

Use it to inspect what the compiler actually emitted for a given line of your kernel — loads, stores, ALU ops, and how they were optimized.

## Install

```bash
pip install git+https://github.com/AMDResearch/kerneldb.git
```

Or from a local clone:

```bash
git clone https://github.com/AMDResearch/kerneldb.git
cd kerneldb
pip install -e .
```

This builds the C++ library with CMake and installs the Python bindings.

## Python API

```python
from kerneldb import KernelDB

kdb = KernelDB("my_kernel.hsaco")

for kernel_name in kdb.get_kernels():
    kernel = kdb.get_kernel(kernel_name)

    # Source lines that map to instructions
    for line in kdb.get_kernel_lines(kernel_name):
        instructions = kdb.get_instructions_for_line(kernel_name, line)
        for inst in instructions:
            print(f"[{inst.line}:{inst.column}] {inst.disassembly}")

    # Filter for memory operations
    for line in kdb.get_kernel_lines(kernel_name):
        mem_ops = kdb.get_instructions_for_line(kernel_name, line, ".*(load|store).*")
```

### Kernel arguments

KernelDB extracts kernel argument metadata from DWARF info, including nested structs and typedef resolution:

```python
args = kdb.get_kernel_arguments(kernel_name)
for arg in args:
    print(f"{arg.name}: {arg.type_name} ({arg.size} bytes)")

# Resolve typedefs to underlying types
args = kdb.get_kernel_arguments(kernel_name, resolve_typedefs=True)
```

### Kernel wrapper

The `Kernel` object provides convenient properties:

```python
kernel = kdb.get_kernel(kernel_name)
kernel.name          # kernel name/signature
kernel.lines         # source line numbers
kernel.assembly      # full disassembly as list of strings
kernel.files         # source files referenced
kernel.arguments     # kernel arguments from DWARF
kernel.get_basic_blocks()
kernel.get_instructions_for_line(line, filter_pattern=None)
```

See the `examples/` directory for complete runnable examples.

## Building the C++ library only

KernelDB depends on LLVM for disassembly. Use `rocm-llvm-dev` or the Triton LLVM (typically under `~/.triton/llvm`):

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/.local ..
make && make install
```

This installs `libkernelDB64.so` to `${CMAKE_INSTALL_PREFIX}/lib` and headers to `${CMAKE_INSTALL_PREFIX}/include`.

## C++ API

```c++
#include "inc/kernelDB.h"

hsa_init();
hsa_agent_t agent = /* get a GPU agent */;
kernelDB::kernelDB kdb(agent, "my_kernel.hsaco");

std::vector<std::string> kernels;
kdb.getKernels(kernels);

for (auto& kernel : kernels) {
    std::vector<uint32_t> lines;
    kdb.getKernelLines(kernel, lines);
    for (auto line : lines) {
        auto instructions = kdb.getInstructionsForLine(kernel, line);
        for (auto& inst : instructions) {
            std::cout << inst.disassembly_ << std::endl;
        }
    }
}
```

## License

MIT
