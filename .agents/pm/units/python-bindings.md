# Python Bindings

## Responsibility

Provides a Python API for kernelDB via pybind11. The C++ pybind11 wrapper exposes all core classes and types as a `_kerneldb` extension module. A pure-Python `kerneldb` package wraps the extension module with a higher-level `KernelDB` class and a `Kernel` convenience wrapper.

## Key Source Files

- `src/pybind11_wrapper.cc` — pybind11 bindings for all C++ types: instruction_t, basicBlock, CDNAKernel, kernelDB, KernelArgument, kdb_arch_descriptor_t, hsa_agent_t; helper functions hsa_init() and get_first_gpu_agent()
- `kerneldb/api.py` — pure-Python `KernelDB` class (handles HSA init, agent selection, lazy/eager construction) and `Kernel` wrapper with convenience properties
- `kerneldb/__init__.py` — package init; re-exports all public types; uses importlib.metadata for versioning

## Key Types and Classes

- `_kerneldb.KernelDB` (C++) — pybind11 binding of `kernelDB::kernelDB`
- `_kerneldb.CDNAKernel` (C++) — pybind11 binding of `kernelDB::CDNAKernel`
- `_kerneldb.BasicBlock` (C++) — pybind11 binding of `kernelDB::basicBlock`
- `_kerneldb.Instruction` (C++) — pybind11 binding of `kernelDB::instruction_t`
- `_kerneldb.KernelArgument` (C++) — pybind11 binding of `KernelArgument`
- `_kerneldb.ArchDescriptor` (C++) — pybind11 binding of `kdb_arch_descriptor_t`
- `_kerneldb.HsaAgent` (C++) — pybind11 binding of `hsa_agent_t`
- `kerneldb.KernelDB` (Python) — high-level wrapper; manages HSA init and agent selection
- `kerneldb.Kernel` (Python) — convenience wrapper around CDNAKernel with properties: name, lines, assembly, hip_source, files, arguments, signature

## Key Functions and Entry Points

- `KernelDB.__init__(binary_path, agent_id, lazy)` — initializes HSA, gets GPU agent, creates C++ kernelDB instance
- `KernelDB.add_file(path, filter, lazy)` — adds a binary with optional lazy loading (default: lazy=True)
- `KernelDB.get_kernels()` — returns list of kernel name strings
- `KernelDB.get_kernel(name)` — returns `Kernel` wrapper instance
- `KernelDB.get_kernel_lines(kernel_name)` — returns list of source line numbers
- `KernelDB.get_instructions_for_line(kernel_name, line, filter_pattern)` — returns list of Instruction objects
- `KernelDB.get_kernel_arguments(kernel_name, resolve_typedefs)` — returns list of KernelArgument
- `KernelDB.scan_code_object(co_file)` — manual code object scan
- `KernelDB.has_kernel(name)` — existence check
- `Kernel.lines` — property returning source line numbers
- `Kernel.assembly` — property returning full disassembly as list of strings
- `Kernel.files` — property returning sorted list of source files referenced
- `Kernel.arguments` — property returning kernel arguments from DWARF
- `Kernel.get_basic_blocks()` — returns list of BasicBlock objects
- `Kernel.get_instructions_for_line(line, filter_pattern)` — per-line instruction query

## Data Flow

1. Python user creates `KernelDB(binary_path)` or `KernelDB(lazy=True)`
2. Constructor calls `_kerneldb.hsa_init()` and `_kerneldb.get_first_gpu_agent()`
3. Creates `_kerneldb.KernelDB(agent, path)` or `_kerneldb.KernelDB(agent)` for lazy
4. All query methods delegate to the underlying C++ object
5. `get_kernel()` wraps the returned CDNAKernel in a Python `Kernel` object
6. The `Kernel` wrapper provides convenience properties that iterate underlying C++ data

## Invariants

- HSA must be initialized before creating a KernelDB instance; failure raises RuntimeError
- At least one GPU agent must be available (or explicitly provide agent_id)
- The `_kerneldb` C++ module must be compiled and importable (ImportError with helpful message on failure)
- Return value policies: `reference_internal` for references into C++ containers (instructions, blocks, arguments); copy for filtered instruction lists
- KernelArgument.members is recursive (pybind11 handles the nested vector automatically)

## Dependencies

- **pybind11** (build-time) — C++/Python binding generation
- **kernelDB C++ library** — the compiled `_kerneldb` extension module
- **HSA runtime** — must be available at import/construction time
- **setuptools + CMake** — build system for compiling the extension (see `pyproject.toml`, `setup.py`)

## Negative Knowledge

- **The Python `KernelDB` class is NOT the same as `_kerneldb.KernelDB`**. The Python class adds HSA initialization, agent management, and wraps results in `Kernel` objects. The C++ binding is the raw interface.
- **`Kernel.assembly` iterates all lines and concatenates instructions**. For large kernels this can be slow. The C++ `getDisassembly()` returns the raw text but is not exposed via the Python Kernel wrapper (only via CDNAKernel directly).
- **Version falls back to "0.0.0+unknown"** if the package is not pip-installed (e.g., development mode without setuptools-scm).

## Open Questions

- None currently.

## Last Verified

- 2026-04-27 by agent (migration from KT to v0.3 PM)
