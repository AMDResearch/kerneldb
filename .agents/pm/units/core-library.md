# Core Library (kernelDB class)

## Responsibility

Central database class that manages GPU kernel representations. Accepts HIP executables or HSACO files, orchestrates code object extraction, disassembly, DWARF source-line mapping, and kernel argument extraction. Provides the query API for looking up kernels, mapping source lines to ISA instructions, and retrieving kernel arguments. Supports both eager (full load at construction) and lazy (on-demand per-kernel disassembly) loading modes.

## Key Source Files

- `src/kernelDB.cc` — main implementation (~600 lines); constructors, addFile, parseDisassembly, ensureKernelLoaded, lazy loading, query methods
- `include/kernelDB.h` — all class definitions, struct declarations, free function prototypes, inline utilities (ltrim/rtrim/trim/split)

## Key Types and Classes

- `kernelDB::kernelDB` — main database class; holds `kernels_` map, `lazy_kernels_` index, HSA agent, file map, scanned code objects set
- `kernelDB::CDNAKernel` — single kernel representation; owns basic blocks, line-to-instruction map, file names, kernel arguments
- `kernelDB::basicBlock` — instruction container with thread-safe accessors
- `kernelDB::instruction_t` — parsed instruction struct: prefix, type, size, inst, operands, disassembly text, address, source location (line/column/file)
- `kernelDB::parse_mode` — enum (BEGIN, KERNEL, BBLOCK, BRANCH) for disassembly line classification
- `KernelArgument` — recursive struct for kernel parameter metadata (name, type, size, offset, alignment, position, nested members)
- `kdb_arch_descriptor_t` — GPU architecture descriptor (ISA string, XCCs, SEs, CUs, SIMDs, wave size, max waves)
- `SourceLocation` — DWARF source mapping (fileName, lineNumber, columnNumber)
- `LazyKernelEntry` — internal struct holding hsaco_path, logical_file, and elf_symbol for deferred disassembly

## Key Functions and Entry Points

- `kernelDB::kernelDB(agent, fileName)` — eager constructor; loads executable (or all shared libs if empty path)
- `kernelDB::kernelDB(agent)` — lazy constructor; empty DB for subsequent add_file calls
- `kernelDB::addFile(name, agent, filter, lazy)` — main file ingestion; extracts code objects, either indexes (lazy) or disassembles+maps (eager)
- `kernelDB::getKernel(name)` — returns CDNAKernel reference; triggers ensureKernelLoaded for lazy kernels
- `kernelDB::ensureKernelLoaded(name)` — thread-safe lazy loading: checks lazy_kernels_, acquires loading_mutex_, calls scanCodeObjectForKernel, removes from lazy index
- `kernelDB::scanCodeObject(co_file)` — idempotent full scan of a .hsaco: disassembly + DWARF + arguments
- `kernelDB::scanCodeObjectForKernel(co_file, kernelName)` — targeted single-kernel scan using --disassemble-symbols
- `kernelDB::parseDisassembly(text)` — parses llvm-objdump output into CDNAKernel/basicBlock/instruction_t structures
- `kernelDB::parseDisassemblyForKernel(text, targetKernel)` — same, but only for one kernel
- `kernelDB::mapDisassemblyToSource(agent, elfFilePath)` — builds DWARF address map and annotates instructions
- `kernelDB::getInstructionsForLine(kernel, line [, match])` — source-line query, with optional regex filter
- `kernelDB::hasKernel(name)` — checks both loaded and lazy-indexed kernels
- `kernelDB::getKernelArguments(kernel, resolve_typedefs)` — extracts argument metadata from DWARF
- `kernelDB::getKernelNamesFromElf(fileName)` — reads .symtab to index kernel names without disassembly (static)
- `kernelDB::getCodeObjectInfo(agent, bits)` — gets code object offsets/sizes from fatbin data (static)
- `kernelDB::getElfSectionBits(fileName, sectionName, offset, sectionData)` — reads raw ELF section bytes (static)
- `demangleName(name)` — C++ name demangling with special handling for OMNIPROBE_PREFIX (`__amd_crk_`)
- `getKernelName(name)` — canonical kernel name (strips trailing suffixes after last `)` or `.`)
- `getIsaList(agent)` — queries HSA for supported ISA strings

## Data Flow

1. Constructor receives executable path + HSA agent (or empty path for process-wide scan)
2. `addFile()` calls `extractCodeObjects()` to find `.hip_fatbin`, extract per-arch code objects to temp files
3. **Eager path**: `getDisassembly()` invokes llvm-objdump on each .hsaco; `parseDisassembly()` tokenizes output into CDNAKernel/basicBlock/instruction_t; `mapDisassemblyToSource()` builds DWARF address map and annotates instructions
4. **Lazy path**: `getKernelNamesFromElf()` reads .symtab symbols only; stores `LazyKernelEntry` per kernel
5. On query (`getKernel`, `getInstructionsForLine`, etc.), `ensureKernelLoaded()` triggers `scanCodeObjectForKernel()` which does targeted disassembly + DWARF + argument extraction for just that kernel
6. `processKernelsWithAddressMap()` walks each instruction's address, looks up nearest DWARF address, and sets file/line/column fields
7. `extractKernelArguments()` reads DWARF DIEs for subprogram parameters and struct member breakdowns

## Invariants

- One kernelDB instance per executable+agent combination
- Code object must match target GPU architecture (ISA must be in agent's ISA list)
- DWARF info is optional for disassembly but required for source correlation (-g compile flag)
- Instrumented kernels use `__amd_crk_` prefix (OMNIPROBE_PREFIX) which must be handled during demangling
- Kernel names are canonicalized via `getKernelName()` for consistent map lookups
- Thread safety: `shared_mutex` for kernels_ and lazy_kernels_; separate `loading_mutex_` + condition_variable to prevent concurrent disassembly of the same kernel
- Destructor unlinks temporary .hsaco files (but not the original input files)
- `scanCodeObject` is idempotent (tracked via `scanned_code_objects_` set); `scanCodeObjectForKernel` is not (to allow other kernels from same CO to load later)

## Dependencies

- **libdwarf** — DWARF debug info parsing for address-to-source mapping and argument extraction. Also load: `dwarf-address-map.md`
- **libelf / gelf** — ELF binary parsing for section extraction and symbol table reading
- **amd_comgr** — AMD code object manager for code object info queries
- **HSA runtime** — `hsa_agent_t` for GPU agent identification, ISA queries
- **llvm-objdump** — external process invoked for disassembly (located via `ROCM_PATH`). Also load: `disassembly.md`
- **Code Object Extraction** — `extractCodeObjects()` handles fatbin and CCOB decompression. Also load: `code-object-extraction.md`

## Negative Knowledge

- **The regex-filtered `getInstructionsForLine` creates a copy**, not a reference. The unfiltered overload returns a const reference into the kernel's internal line_map_. Do not treat them identically regarding lifetime.
- **Empty file path in constructor means "load entire running process"**, not "do nothing". It calls `getExecutablePath()` then `getSharedLibraries()` and loads everything. Use the agent-only constructor for an empty DB.
- **`getKernelNamesFromElf` is a static method** that reads the .symtab section directly with libelf. It does not use DWARF or comgr.
- **The destructor calls `unlink()` on temp files** stored in `file_map_`. Files where the key equals the value (meaning the original file was an .hsaco, not extracted) are not deleted.

## Open Questions

- The `getBasicBlocks(kernel, vector)` method is a stub (always returns true) -- unclear if it was meant to have a different purpose from CDNAKernel::getBasicBlocks().
- Column marker generation (`genColumnMarkers`) is defined but usage is not visible in the core flow.

## Last Verified

- 2026-04-27 by agent (migration from KT to v0.3 PM)
