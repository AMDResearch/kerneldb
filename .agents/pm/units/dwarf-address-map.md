# DWARF Address Map and Argument Extraction

## Responsibility

Parses DWARF debug information from .hsaco code objects to build address-to-source-location mappings and to extract kernel argument (parameter) metadata including nested struct members and typedef resolution.

## Key Source Files

- `src/addressMap.cc` — buildDwarfAddressMap, getSourceLocation, extractKernelArguments, DWARF DIE traversal helpers (~1000 lines)

## Key Types and Classes

- `SourceLocation` (defined in `include/kernelDB.h`) — holds fileName, lineNumber, columnNumber from DWARF
- `KernelArgument` (defined in `include/kernelDB.h`) — recursive struct: name, type_name, size, offset, alignment, position, and `members` vector for nested structs
- `std::map<Dwarf_Addr, SourceLocation>` — the address map: ISA address to source location

## Key Functions and Entry Points

- `buildDwarfAddressMap(filename, offset, hsaco_length, addressMap)` — opens .hsaco ELF with libdwarf, iterates compilation units and line tables, populates addressMap with all address-to-source mappings
- `getSourceLocation(addrMap, addr)` — looks up an ISA address in the map; uses floor-key search (`find_floor_key`) to find the nearest preceding address
- `extractKernelArguments(filename, offset, hsaco_length, kernelArgsMap, resolve_typedefs)` — traverses DWARF DIEs for DW_TAG_subprogram entries, reads DW_TAG_formal_parameter children, resolves types recursively for struct members
- `find_floor_key(map, target)` — template helper for finding the largest key <= target in an ordered map
- `is_subprogram(die, error)` — checks if a DIE has tag DW_TAG_subprogram
- `process_function_die(dbg, die, decl_line)` — extracts declaration line from function DIE attributes

## Data Flow

1. `buildDwarfAddressMap` receives a .hsaco file path (with optional offset/length for embedded code objects)
2. Opens the file with `dwarf_init_path` (or via fd for offset-based access)
3. Iterates all compilation units via `dwarf_next_cu_header_d`
4. For each CU, reads line tables via `dwarf_srclines` and extracts (address, file, line, column) tuples
5. Populates `std::map<Dwarf_Addr, SourceLocation>` — sorted by address
6. `processKernelsWithAddressMap` (in kernelDB.cc) then walks each kernel's instructions and calls `getSourceLocation()` to annotate them
7. For argument extraction: `extractKernelArguments` iterates DIEs looking for DW_TAG_subprogram, then DW_TAG_formal_parameter children; resolves types via DW_AT_type chains including DW_TAG_typedef and DW_TAG_structure_type

## Invariants

- DWARF info is only present if the binary was compiled with `-g` (debug info flag)
- `MISSING_SOURCE_INFO` (0xffffffff) is used as sentinel for instructions with no DWARF mapping
- Address map uses floor-key lookup: an instruction at address A is mapped to the source location of the largest DWARF address <= A
- Argument extraction handles: pointers, typedefs/using aliases, nested structs (recursive member traversal), template instantiations
- `resolve_typedefs=true` follows DW_TAG_typedef chains to the underlying type; `false` preserves the alias name as written in source
- Ubuntu vs non-Ubuntu libdwarf differences are handled via `UBUNTU_LIBDWARF` compile define (affects API compatibility)

## Dependencies

- **libdwarf** — DWARF debugging format parsing (libdwarf-0 or libdwarf)
- **libelf / gelf** — ELF binary access for section reading and fd-based DWARF init
- Platform-specific libdwarf API differences handled via `UBUNTU_LIBDWARF` macro

## Negative Knowledge

- **The address map is built per code object, not per kernel.** All addresses from all kernels in a single .hsaco end up in one map. The per-kernel association happens when `processKernelsWithAddressMap` iterates instructions in kernelDB.cc.
- **When offset and hsaco_length are both 0**, the function treats the entire file as the code object (used for standalone .hsaco files). Non-zero values indicate an embedded code object within a larger binary.

## Open Questions

- None currently.

## Last Verified

- 2026-04-27 by agent (migration from KT to v0.3 PM)
