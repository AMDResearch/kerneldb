# Project Memory Current State

## Summary

kernelDB is a mature library for extracting and correlating GPU kernel information from HIP/ROCm executables. It parses disassembly (via llvm-objdump), DWARF debug info (via libdwarf), and ELF structures to provide ISA-level introspection mapped to source code.

## Recent Activity

- Lazy loading support added: `addFile(..., lazy=true)` indexes kernel names from ELF symbol tables without disassembly, then disassembles on demand per-kernel via `--disassemble-symbols`
- CCOB (compressed code object bundle) support added for ROCm 6.x fatbin format changes
- Kernel argument extraction with nested struct member traversal and typedef resolution
- Python test suite expanded with argument, lazy loading, and import tests

## Active Work Areas

- Python bindings and Python-first API (`kerneldb/api.py`)
- Lazy loading path for performance with large binaries containing many kernels

## Current Risks

- Tests require GPU hardware and ROCm stack; no mock/offline testing path exists
- CCOB V1 format is effectively unsupported (no FileSize field in header)
- `tmpnam()` usage in co_extract.cc is inherently racy (contrast with `mkstemp()` used elsewhere)

## Migrated From

KT dossier (`.agents/kt/architecture.md`) migrated to v0.3 PM on 2026-04-27. Original KT content archived at `.agents/kt.archive/`.

## Changed Assumptions

None since migration.
