# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for KernelDB kernel argument extraction:
  - Basic argument metadata (name, type, size, alignment)
  - Nested struct member recursion
  - Typedef / using-alias resolution
  - Template kernel instantiations

All tests in this module require a ROCm environment with hipcc and a GPU.
"""

import pytest
from conftest import requires_rocm

kerneldb = pytest.importorskip("kerneldb", reason="kernelDB C++ extension not available")
KernelDB = kerneldb.KernelDB


def _find_kernel(kernels, fragment):
    matches = [k for k in kernels if fragment in k]
    if not matches:
        pytest.skip(f"No kernel containing {fragment!r} found in binary")
    return matches[0]


def _arg_by_name(arguments, name):
    return next((a for a in arguments if a.name == name), None)


# ---------------------------------------------------------------------------
# Basic argument metadata
# ---------------------------------------------------------------------------


@requires_rocm
def test_kernel_arguments(arguments_binary):
    """kernel_with_args must expose 4 named arguments with expected types/sizes."""
    kdb = KernelDB(arguments_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "kernel_with_args")
    args = kdb.get_kernel_arguments(kernel_name)

    assert isinstance(args, list) and len(args) == 4
    for arg in args:
        assert isinstance(arg.name, str) and arg.name
        assert isinstance(arg.type_name, str) and arg.type_name
        assert isinstance(arg.size, int) and arg.size > 0
        assert isinstance(arg.alignment, int) and arg.alignment > 0

    names = [a.name for a in args]
    for expected in ("a", "b", "c", "n"):
        assert expected in names, f"Expected argument {expected!r} in {names}"

    ptr_arg = _arg_by_name(args, "a")
    assert ptr_arg.size == 8, f"Pointer size must be 8 bytes, got {ptr_arg.size}"

    n_arg = _arg_by_name(args, "n")
    assert n_arg.size == 4, f"int size must be 4 bytes, got {n_arg.size}"
    assert "int" in n_arg.type_name.lower()


@requires_rocm
def test_kernel_wrapper_arguments(arguments_binary):
    """Kernel.has_arguments() and .arguments must reflect the extracted args."""
    kdb = KernelDB(arguments_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "kernel_with_args")
    kernel = kdb.get_kernel(kernel_name)

    assert kernel.has_arguments() is True
    assert isinstance(kernel.arguments, list) and kernel.arguments


# ---------------------------------------------------------------------------
# Nested struct member recursion
# ---------------------------------------------------------------------------


@requires_rocm
def test_nested_struct_arguments(nested_structs_binary):
    """update_bounds args must include a BoundingBox arg exposing Point3D members."""
    kdb = KernelDB(nested_structs_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "update_bounds")
    args = kdb.get_kernel_arguments(kernel_name)

    # Every argument must have a members attribute
    for arg in args:
        assert hasattr(arg, "members")

    # Helper: recursively search for a type by name
    def find_type(arg_list, type_fragment):
        for arg in arg_list:
            if type_fragment in arg.type_name:
                return arg
            if arg.members:
                found = find_type(arg.members, type_fragment)
                if found:
                    return found
        return None

    point = find_type(args, "Point3D")
    if point is None:
        pytest.skip("Point3D not found in argument tree (may be optimised away)")
    assert len(point.members) == 3, f"Point3D should have 3 members, got {len(point.members)}"


# ---------------------------------------------------------------------------
# Typedef / using-alias resolution
# ---------------------------------------------------------------------------


@requires_rocm
def test_typedef_resolution(typedef_binary):
    """typedef_kernel must have 4 args; typedefs preserved or resolved correctly."""
    kdb = KernelDB(typedef_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "typedef_kernel")

    args_raw = kdb.get_kernel_arguments(kernel_name, resolve_typedefs=False)
    assert len(args_raw) == 4

    type_names_raw = [a.type_name for a in args_raw]
    assert any(n in ("MyInt", "MyFloat", "MyDouble") for n in type_names_raw), (
        f"Expected typedef names to be preserved, got: {type_names_raw}"
    )

    args_resolved = kdb.get_kernel_arguments(kernel_name, resolve_typedefs=True)
    type_names_resolved = [a.type_name for a in args_resolved]
    for alias in ("MyInt", "MyFloat", "MyDouble"):
        assert alias not in type_names_resolved, (
            f"Alias {alias!r} should have been resolved, got: {type_names_resolved}"
        )

    primitive_types = {"int", "float", "double", "unsigned int"}
    for type_name in type_names_resolved:
        base = type_name.replace("*", "").strip()
        assert any(p in base for p in primitive_types), (
            f"Resolved type {type_name!r} does not look like a C primitive"
        )


# ---------------------------------------------------------------------------
# Template kernel instantiations
# ---------------------------------------------------------------------------


@requires_rocm
def test_template_kernel_arguments(template_binary):
    """scale_values must be instantiated for at least float and double."""
    kdb = KernelDB(template_binary)
    kernels = kdb.get_kernels()

    scale_kernels = [k for k in kernels if "scale_values" in k]
    assert len(scale_kernels) >= 2, f"Expected ≥2 template instantiations, got {scale_kernels}"

    for type_fragment in ("float", "double"):
        matching = [k for k in scale_kernels if type_fragment in k]
        if not matching:
            pytest.skip(f"scale_values<{type_fragment}> not found in binary")
        args = kdb.get_kernel_arguments(matching[0])
        assert len(args) == 4, f"Expected 4 args for scale_values<{type_fragment}>"
        non_int = [a for a in args if a.name != "n"]
        assert all(type_fragment in a.type_name.lower() for a in non_int), (
            f"Non-count args should be {type_fragment}: {[a.type_name for a in non_int]}"
        )
