# Disassembly Subsystem

## Responsibility

Invokes llvm-objdump to disassemble .hsaco code objects into textual assembly. Provides both full-file disassembly and targeted per-symbol disassembly (using `--disassemble-symbols`). Handles temporary file I/O for capturing objdump output.

## Key Source Files

- `src/disassemble.cc` — getDisassembly, getDisassemblyForSymbol, invokeProgram, readFileToString

## Key Types and Classes

- No new types defined. Uses `hsa_agent_t` for GPU identification and std::string for I/O.

## Key Functions and Entry Points

- `getDisassembly(agent, fileName, out)` — full disassembly of a .hsaco file; queries agent name for `--mcpu` flag; invokes `llvm-objdump -d --arch-name=amdgcn --mcpu=<gpu>`; writes output to temp file then reads into string
- `getDisassemblyForSymbol(agent, fileName, symbolName, out)` — targeted disassembly of a single symbol using `--disassemble-symbols=<name>`; used by lazy loading path for single-kernel extraction
- `invokeProgram(programName, params, outputFileName)` — generic external process runner; constructs shell command with single-quoted parameters (escaping embedded quotes); uses `popen/pclose`; redirects stdout to outputFileName
- `readFileToString(filename, content)` — reads entire file into a std::string (binary mode)
- `init_disassembler_path()` — resolves llvm-objdump path from `ROCM_PATH` env var or defaults to `/opt/rocm/llvm/bin/llvm-objdump`

## Data Flow

1. Caller passes HSA agent and .hsaco file path
2. `getDisassembly` queries `hsa_agent_get_info` for GPU name (e.g., gfx90a)
3. Constructs llvm-objdump command: `-d --arch-name=amdgcn --mcpu=gfx90a <file>`
4. `invokeProgram` runs the command via popen, capturing output to a temp file
5. `readFileToString` loads the temp file into the output string
6. Temp file is unlinked
7. Caller (kernelDB::addFile or scanCodeObjectForKernel) passes the text to `parseDisassembly()`

## Invariants

- Requires llvm-objdump from ROCm LLVM installation (rocm-llvm-dev or Triton LLVM)
- `ROCM_PATH` must be set or `/opt/rocm` must exist for disassembler lookup
- Shell metacharacters in kernel names (parentheses, angle brackets from templates) are safely handled via single-quoting in invokeProgram
- Temp files use mkstemp pattern `/tmp/kdb_dis_XXXXXX` and are always unlinked after use
- `invokeProgram` throws `std::runtime_error` on failure (non-zero exit, missing output)

## Dependencies

- **ROCm LLVM** — llvm-objdump binary at `$ROCM_PATH/llvm/bin/llvm-objdump`
- **HSA runtime** — for querying agent GPU name

## Negative Knowledge

- **This does NOT use amd_comgr for disassembly** despite comgr being linked. The comgr dependency is used for code object info extraction only. Disassembly is done by invoking llvm-objdump as an external process.
- **The `disassembly_params` global** (`{"-d", "--arch-name=amdgcn"}`) is used only by `getDisassembly`; `getDisassemblyForSymbol` constructs its own param list (omitting `-d`, using `--disassemble-symbols` instead).

## Open Questions

- None currently.

## Last Verified

- 2026-04-27 by agent (migration from KT to v0.3 PM)
