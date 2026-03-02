# kerneldb Architecture

## Overview
Kernel database library for extracting and correlating GPU kernel information. Parses disassembly, DWARF debug info, and ELF structures from HIP/ROCm executables to provide ISA-level introspection.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HIP Executable (.exe)                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  .hip_fatbin section (embedded code objects)                 │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │   │
│  │  │ gfx906 CO   │ │ gfx90a CO   │ │ gfx942 CO   │            │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘            │   │
│  └─────────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         kernelDB                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Disassembly  │  │ DWARF Parser │  │ ELF Parser   │              │
│  │ (via comgr)  │  │ (libdwarf)   │  │ (libelf)     │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                       │
│         └────────────────►├◄────────────────┘                       │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  CDNAKernel                                                  │   │
│  │  - basicBlocks[]                                             │   │
│  │  - line_map: line → instructions                             │   │
│  │  - arguments: KernelArgument[]                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. Constructor receives executable path + HSA agent
2. `extractCodeObjects()` finds `.hip_fatbin`, extracts per-arch code objects
3. `getDisassembly()` uses amd_comgr to disassemble code object
4. `parseDisassembly()` parses into kernels, basic blocks, instructions
5. `buildDwarfAddressMap()` parses DWARF for addr → source location mapping
6. `processKernelsWithAddressMap()` annotates instructions with file/line/col
7. `extractKernelArguments()` parses DWARF for kernel argument metadata

## Key Invariants

- One kernelDB per executable + agent combination
- Code object must match target architecture
- DWARF info required for source correlation (-g flag at compile time)
- Instrumented kernels prefixed with `__amd_crk_`

## Subsystems

| Component | Files | Description |
|-----------|-------|-------------|
| kernelDB class | `src/kernelDB.cc` | Main API, kernel management |
| Address Map | `src/addressMap.cc` | DWARF parsing, addr → source mapping |
| Disassembly | `src/disassemble.cc` | comgr-based disassembly |
| CO Extract | `src/co_extract.cc` | Code object extraction from fatbin |
| Python bindings | `src/pybind11_wrapper.cc` | Python API |

## Key Classes

### kernelDB
Main database class.
- `getKernel(name)` — get CDNAKernel by name
- `getInstructionsForLine(kernel, line)` — ISA for source line
- `addFile(name, agent, filter)` — load executable

### CDNAKernel
Single kernel representation.
- `getBasicBlocks()` — control flow blocks
- `getInstructionsForLine(line)` — instructions for source line
- `getArguments()` — kernel parameter metadata

### basicBlock
Instruction container.
- `getInstructions()` — instruction list

### instruction_t
Parsed instruction.
- `prefix_`, `type_`, `size_`, `inst_` — instruction parts
- `address_` — ISA address
- `line_`, `column_`, `file_name_` — source location

## Dependencies
- libdwarf (DWARF parsing)
- libelf (ELF parsing)
- amd_comgr (disassembly)
- HSA runtime (agent info)

## Build
- CMake-based
- Produces: shared library + Python module (optional)

## Last Verified
Date: 2026-03-02
