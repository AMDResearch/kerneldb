# Build System and Packaging

## Responsibility

CMake-based build system that produces a shared library (`libkernelDB64.so`) and an optional Python extension module (`_kerneldb`). Also provides CPack packaging for DEB, RPM, and TGZ distribution. Python packaging uses setuptools with setuptools-scm for version management.

## Key Source Files

- `CMakeLists.txt` — root CMake configuration; project setup, dependencies, build/install/packaging
- `src/CMakeLists.txt` — library target definition; source files, link libraries, install rules
- `cmake_modules/env.cmake` — build environment setup (compiler flags, platform detection)
- `cmake_modules/utils.cmake` — version parsing utilities
- `pyproject.toml` — Python package metadata; build-system requirements (setuptools, cmake, pybind11)
- `setup.py` — Python build script; invokes CMake for C++ compilation, copies .so into package
- `MANIFEST.in` — source distribution file inclusion rules

## Key Types and Classes

N/A (build configuration, no runtime types)

## Key Functions and Entry Points

- CMake `project(kernelDB ...)` — project definition with CXX and HIP languages
- CMake `find_package(hip REQUIRED)` — locates ROCm HIP installation
- CMake `find_path(LIBDWARF_INCLUDE_DIR ...)` — finds libdwarf headers
- CMake target `kernelDB64` — the shared library target
- Python `setup.py` build — invokes CMake, builds C++ library, compiles pybind11 extension
- `pip install -e .` — editable install with automatic cmake build

## Data Flow

1. CMake reads `ROCM_PATH` (env or `/opt/rocm` default) for ROCm toolchain
2. Finds `hipcc` as the C++ compiler
3. Locates libdwarf headers (searches multiple paths including `/usr/include/libdwarf-0`)
4. Detects Ubuntu vs non-Ubuntu via `/etc/os-release` and sets `UBUNTU_LIBDWARF` macro
5. Includes `src/CMakeLists.txt` to build `libkernelDB64.so` from C++ sources
6. Optionally builds test executable (`kdbtest`) and examples
7. For Python: setuptools calls cmake, builds pybind11 module `_kerneldb.*.so`, installs into `kerneldb/` package

## Invariants

- Requires ROCm installation (hipcc, HSA headers, amd_comgr)
- Requires libdwarf and libelf development headers
- C++20 standard required
- `hipcc` is used as the C++ compiler (not g++ or clang++ directly)
- Ubuntu libdwarf has different API signatures; detected automatically via OS detection
- Version managed by setuptools-scm: tags for releases, `.postN+hash` for dev builds
- The library name is `kernelDB64` (64-bit convention)

## Dependencies

- **ROCm** — hipcc compiler, HSA runtime, amd_comgr, hsa headers
- **libdwarf** — DWARF parsing (development headers: libdwarf-0 or libdwarf)
- **libelf** — ELF binary parsing (development headers)
- **pybind11** — Python binding generation (for Python extension)
- **CMake >= 3.15** — build system
- **setuptools, setuptools-scm** — Python packaging

## Negative Knowledge

- **`CMAKE_INSTALL_PREFIX` defaults to "/"** which is unusual. Most consumer builds override this.
- **The test executable `kdbtest` is NOT built by default** -- requires `KERNELDB_BUILD_TESTING=ON`.
- **There is an inconsistency**: `INTERCEPTOR_TOPLEVEL_PROJECT` is checked on line 84 but the variable that gets set is `KERNELDB_TOPLEVEL_PROJECT`. The `INTERCEPTOR_TOPLEVEL_PROJECT` check will always be false (the variable is never set). This means the C++ standard settings always take the `else` branch.

## Open Questions

- The `INTERCEPTOR_TOPLEVEL_PROJECT` vs `KERNELDB_TOPLEVEL_PROJECT` inconsistency should likely be corrected.

## Last Verified

- 2026-04-27 by agent (migration from KT to v0.3 PM)
