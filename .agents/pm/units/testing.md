# Testing

## Responsibility

Validates kernelDB functionality across C++ and Python interfaces. The C++ test (`kdbtest`) is a manual integration test. The Python test suite (`tests/`) uses pytest and covers kernel discovery, instruction mapping, filtering, basic blocks, kernel arguments (nested structs, typedefs, templates), and lazy loading correctness.

## Key Source Files

- `test/kdbtest.cc` — C++ integration test; loads an executable, iterates kernels, prints blocks and instructions
- `test/CMakeLists.txt` — CMake build for C++ test
- `tests/conftest.py` — pytest fixtures; `requires_rocm` marker for GPU-dependent tests
- `tests/test_imports.py` — import smoke tests for the Python package
- `tests/test_kernels.py` — kernel discovery, line mapping, instruction extraction, filtering, basic blocks, Kernel wrapper
- `tests/test_arguments.py` — kernel argument extraction: basic args, nested structs, typedef resolution, template instantiations
- `tests/test_lazy_loading.py` — lazy vs eager loading equivalence tests
- `examples/` — runnable example programs (01_hip_kernels through 06_typedef_resolution)

## Key Types and Classes

N/A (test code, no production types)

## Key Functions and Entry Points

- `test_kernel_discovery()` — verifies get_kernels returns both vector_add and vector_multiply
- `test_kernel_lines()` — verifies source line extraction and kernel differentiation
- `test_instructions()` — verifies instruction attributes (disassembly, line, column, file_name)
- `test_instruction_filtering()` — verifies regex filter `.*(load|store).*` returns subset
- `test_basic_blocks_and_kernel_wrapper()` — verifies Kernel wrapper properties
- `test_kernel_arguments()` — verifies basic argument metadata (name, type, size, alignment)
- `test_nested_struct_arguments()` — verifies recursive Point3D/BoundingBox member extraction
- `test_typedef_resolution()` — verifies resolve_typedefs=True strips MyInt/MyFloat/MyDouble
- `test_template_kernel_arguments()` — verifies scale_values<float> and scale_values<double> argument types
- `test_lazy_kernel_discovery()` — verifies lazy and eager produce same kernel list
- `test_lazy_has_kernel()` — verifies has_kernel works before lazy loading
- `test_lazy_get_kernel_matches_eager()` — verifies lazy assembly matches eager assembly
- `test_lazy_instructions_for_line()` — verifies lazy per-line instructions match eager

## Data Flow

1. Python tests compile HIP source code with `hipcc -g` into temp binaries
2. Create KernelDB instances pointing at the compiled binaries
3. Query kernels, lines, instructions, arguments and assert correctness
4. Lazy loading tests create both eager and lazy instances and compare results for equivalence
5. Tests are skipped if ROCm/hipcc is not available (`requires_rocm` marker)

## Invariants

- All Python tests require a working ROCm environment with hipcc and at least one GPU
- Test binaries must be compiled with `-g` for DWARF info
- Lazy and eager loading must produce identical results (kernel names, assembly, instructions, lines)
- The `conftest.py` `requires_rocm` marker controls test skipping on non-GPU systems

## Dependencies

- **pytest** — test framework
- **hipcc** — HIP compiler for building test binaries
- **ROCm runtime** — GPU access for HSA initialization
- **kerneldb Python package** — the module under test (importorskip pattern)

## Negative Knowledge

- **Tests cannot run in CI without a GPU** — all meaningful tests are gated by `requires_rocm`.
- **The C++ test (`kdbtest`) is not a unit test** — it is a manual integration tool that takes a binary path as argv[1] and prints output to stdout.

## Open Questions

- None currently.

## Last Verified

- 2026-04-27 by agent (migration from KT to v0.3 PM)
