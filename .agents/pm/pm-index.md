# Project Memory Index

## Units

| Unit | Facet | Purpose | Load when... | Dependencies |
|------|-------|---------|-------------|--------------|
| `core-library.md` | code | Main kernelDB class, kernel management, query API, lazy/eager loading | Working on kernel queries, addFile, lazy loading, or any core API | disassembly, dwarf-address-map, code-object-extraction |
| `disassembly.md` | code | llvm-objdump invocation for .hsaco disassembly | Working on disassembly output, objdump flags, or temp file handling | (none) |
| `dwarf-address-map.md` | code | DWARF parsing for address-to-source mapping and kernel argument extraction | Working on source-line mapping, argument metadata, or struct/typedef handling | (none) |
| `code-object-extraction.md` | code | Code object extraction from HIP fatbins (uncompressed and CCOB-compressed) | Working on fatbin parsing, CCOB support, or code object discovery | (none) |
| `python-bindings.md` | code | pybind11 C++ bindings and Python KernelDB/Kernel wrapper API | Working on Python interface, adding new bindings, or changing the Python API | core-library |
| `build-and-packaging.md` | code | CMake build system, ROCm dependencies, Python packaging (setuptools/pyproject.toml) | Working on build configuration, adding dependencies, or packaging | (none) |
| `testing.md` | code | pytest test suite (kernels, arguments, lazy loading) and C++ integration test | Working on tests, adding test coverage, or debugging CI | python-bindings, core-library |

## Recommended Read Order

1. `core-library.md` — understand the central architecture
2. `code-object-extraction.md` — how binaries are ingested
3. `disassembly.md` — how code objects become text
4. `dwarf-address-map.md` — how instructions get source locations
5. `python-bindings.md` — the user-facing API layer
6. `build-and-packaging.md` — build requirements
7. `testing.md` — validation strategy
