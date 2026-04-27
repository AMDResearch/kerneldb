# Code Object Extraction

## Responsibility

Extracts per-architecture GPU code objects (.hsaco files) from HIP executables. Handles both standard uncompressed `.hip_fatbin` sections (using amd_comgr) and CCOB-compressed fatbin sections (using clang-offload-bundler). Also handles standalone CCOB-compressed code objects (e.g., Tensile .co files).

## Key Source Files

- `src/co_extract.cc` — extractCodeObjects, CCOB detection and unbundling, ELF section reading helpers

## Key Types and Classes

- No new public types. Uses `hsa_agent_t` and `amd_comgr_code_object_info_t` internally.

## Key Functions and Entry Points

- `extractCodeObjects(agent, fileName)` — main entry point; returns a vector of .hsaco file paths (may be temp files that need cleanup)
- `isCCOB(data, size)` / `isCCOBFile(fileName)` — detect CCOB magic header ("CCOB") in buffer or file
- `getCCOBBlockSize(data, available)` — parse CCOB header to determine block size (supports V1, V2, V3 formats with different field layouts)
- `findOffloadBundler()` — locates clang-offload-bundler via `ROCM_PATH` env or PATH
- `buildTarget(agent)` — constructs `hipv4-amdgcn-amd-amdhsa--gfxNNN` target string from agent ISA list
- `unbundleCCOB(bundler, input_file, target)` — runs `clang-offload-bundler --unbundle` to decompress a CCOB block into a .hsaco temp file
- `extractFromCCOBSection(bits, bundler, target)` — iterates 4096-aligned CCOB blocks within a fatbin section
- `createTempFileFromBuffer(data, size)` — writes bytes to a temp file for unbundling
- `create_temp_file_segment(filename, offset, length)` — extracts a segment of a file to a temp file (for uncompressed code objects)

## Data Flow

1. `extractCodeObjects()` checks if the input file is a bare .hsaco (returns it directly)
2. If input is CCOB-compressed standalone: calls `unbundleCCOB` directly and returns result
3. Otherwise reads the `.hip_fatbin` ELF section via `kernelDB::getElfSectionBits()`
4. If fatbin contains CCOB data: iterates 4096-aligned blocks via `extractFromCCOBSection`, unbundling each with clang-offload-bundler
5. If fatbin is standard (uncompressed): uses `kernelDB::getCodeObjectInfo(agent, bits)` via amd_comgr to get offsets/sizes, then extracts each code object to a temp file via `create_temp_file_segment`
6. Returns vector of .hsaco paths; temp files are tracked in `file_map_` for cleanup by kernelDB destructor

## Invariants

- CCOB format versions: V1 (20-byte header, no FileSize field), V2 (24-byte header, 32-bit FileSize), V3 (32-byte header, 64-bit FileSize)
- CCOB blocks within a fatbin section are 4096-byte aligned
- clang-offload-bundler must be available for CCOB decompression (typically at `$ROCM_PATH/llvm/bin/clang-offload-bundler`)
- Standard (non-CCOB) code objects are extracted using amd_comgr for offset/size discovery
- Temp file cleanup is done by the kernelDB destructor via `unlink()`

## Dependencies

- **amd_comgr** — for `getCodeObjectInfo()` on uncompressed fatbin sections
- **clang-offload-bundler** — external tool for decompressing CCOB-format code objects
- **libelf** — for reading `.hip_fatbin` ELF section
- **HSA runtime** — for ISA list to select target architecture

## Negative Knowledge

- **CCOB V1 blocks have no FileSize field**, so `getCCOBBlockSize` returns 0 for V1, which causes the extraction loop to break. V1 CCOB is effectively unsupported.
- **The `tmpnam()` calls in co_extract.cc are inherently racy** (TOCTOU). The kernelDB.cc code uses `mkstemp()` instead. This inconsistency exists but has not caused practical issues.

## Open Questions

- Should V1 CCOB support be added, or is V1 considered legacy/obsolete?

## Last Verified

- 2026-04-27 by agent (migration from KT to v0.3 PM)
