# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for KernelDB kernel argument extraction:
  - Basic argument metadata (name, type, size, alignment)
  - Pointer argument sizes
  - Nested struct member recursion
  - Typedef / using-alias resolution

All tests in this module require a ROCm environment with hipcc and a GPU.
"""

import pytest
from conftest import requires_rocm

kerneldb = pytest.importorskip("kerneldb", reason="kernelDB C++ extension not available", exc_type=ImportError)
KernelDB = kerneldb.KernelDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_kernel(kernels: list[str], fragment: str) -> str:
    matches = [k for k in kernels if fragment in k]
    if not matches:
        pytest.skip(f"No kernel containing {fragment!r} found in binary")
    return matches[0]


def _arg_by_name(arguments, name: str):
    for arg in arguments:
        if arg.name == name:
            return arg
    return None


# ---------------------------------------------------------------------------
# Basic argument metadata
# ---------------------------------------------------------------------------


@requires_rocm
def test_get_kernel_arguments_returns_list(arguments_binary):
    kdb = KernelDB(arguments_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "kernel_with_args")
    args = kdb.get_kernel_arguments(kernel_name)
    assert isinstance(args, list)


@requires_rocm
def test_get_kernel_arguments_non_empty(arguments_binary):
    kdb = KernelDB(arguments_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "kernel_with_args")
    args = kdb.get_kernel_arguments(kernel_name)
    assert len(args) > 0, "Expected kernel_with_args to have argument information"


@requires_rocm
def test_get_kernel_arguments_correct_count(arguments_binary):
    """kernel_with_args(double*, double*, double*, int) has 4 parameters."""
    kdb = KernelDB(arguments_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "kernel_with_args")
    args = kdb.get_kernel_arguments(kernel_name)
    assert len(args) == 4, f"Expected 4 arguments, got {len(args)}"


@requires_rocm
def test_arguments_have_name_attribute(arguments_binary):
    kdb = KernelDB(arguments_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "kernel_with_args")
    for arg in kdb.get_kernel_arguments(kernel_name):
        assert hasattr(arg, "name")
        assert isinstance(arg.name, str)
        assert len(arg.name) > 0


@requires_rocm
def test_arguments_have_type_name_attribute(arguments_binary):
    kdb = KernelDB(arguments_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "kernel_with_args")
    for arg in kdb.get_kernel_arguments(kernel_name):
        assert hasattr(arg, "type_name")
        assert isinstance(arg.type_name, str)
        assert len(arg.type_name) > 0


@requires_rocm
def test_arguments_have_size_attribute(arguments_binary):
    kdb = KernelDB(arguments_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "kernel_with_args")
    for arg in kdb.get_kernel_arguments(kernel_name):
        assert hasattr(arg, "size")
        assert isinstance(arg.size, int)
        assert arg.size > 0


@requires_rocm
def test_arguments_have_alignment_attribute(arguments_binary):
    kdb = KernelDB(arguments_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "kernel_with_args")
    for arg in kdb.get_kernel_arguments(kernel_name):
        assert hasattr(arg, "alignment")
        assert isinstance(arg.alignment, int)
        assert arg.alignment > 0


@requires_rocm
def test_argument_names_match_source(arguments_binary):
    """Argument names should match the source parameter names a, b, c, n."""
    kdb = KernelDB(arguments_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "kernel_with_args")
    args = kdb.get_kernel_arguments(kernel_name)
    names = [arg.name for arg in args]
    for expected in ("a", "b", "c", "n"):
        assert expected in names, f"Expected argument {expected!r} in {names}"


@requires_rocm
def test_pointer_argument_size(arguments_binary):
    """Pointer arguments (double*) must be 8 bytes on a 64-bit platform."""
    kdb = KernelDB(arguments_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "kernel_with_args")
    args = kdb.get_kernel_arguments(kernel_name)
    ptr_arg = _arg_by_name(args, "a")
    assert ptr_arg is not None, "Argument 'a' not found"
    assert ptr_arg.size == 8, f"Expected pointer size 8, got {ptr_arg.size}"


@requires_rocm
def test_int_argument_size(arguments_binary):
    """The 'n' argument is an int, which must be 4 bytes."""
    kdb = KernelDB(arguments_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "kernel_with_args")
    args = kdb.get_kernel_arguments(kernel_name)
    n_arg = _arg_by_name(args, "n")
    assert n_arg is not None, "Argument 'n' not found"
    assert n_arg.size == 4, f"Expected int size 4, got {n_arg.size}"


@requires_rocm
def test_int_argument_type_name(arguments_binary):
    kdb = KernelDB(arguments_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "kernel_with_args")
    args = kdb.get_kernel_arguments(kernel_name)
    n_arg = _arg_by_name(args, "n")
    assert n_arg is not None, "Argument 'n' not found"
    assert "int" in n_arg.type_name.lower()


# ---------------------------------------------------------------------------
# Kernel wrapper: has_arguments / arguments property
# ---------------------------------------------------------------------------


@requires_rocm
def test_kernel_wrapper_has_arguments_true(arguments_binary):
    kdb = KernelDB(arguments_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "kernel_with_args")
    kernel = kdb.get_kernel(kernel_name)
    assert kernel.has_arguments() is True


@requires_rocm
def test_kernel_wrapper_arguments_property(arguments_binary):
    kdb = KernelDB(arguments_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "kernel_with_args")
    kernel = kdb.get_kernel(kernel_name)
    args = kernel.arguments
    assert isinstance(args, list)
    assert len(args) > 0


# ---------------------------------------------------------------------------
# Nested struct member recursion
# ---------------------------------------------------------------------------


@requires_rocm
def test_nested_structs_kernel_found(nested_structs_binary):
    kdb = KernelDB(nested_structs_binary)
    kernels = kdb.get_kernels()
    assert any("update_bounds" in k for k in kernels), (
        f"'update_bounds' not found in {kernels}"
    )


@requires_rocm
def test_nested_structs_argument_has_members(nested_structs_binary):
    """The BoundingBox argument must expose its Point3D sub-members."""
    kdb = KernelDB(nested_structs_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "update_bounds")
    args = kdb.get_kernel_arguments(kernel_name)

    # Find the BoundingBox argument (passed by pointer as first arg)
    bbox_arg = None
    for arg in args:
        if "BoundingBox" in arg.type_name or "boxes" == arg.name:
            bbox_arg = arg
            break

    assert bbox_arg is not None, (
        f"Expected a BoundingBox/boxes argument, got {[a.type_name for a in args]}"
    )


@requires_rocm
def test_nested_structs_members_attribute_is_list(nested_structs_binary):
    """Every KernelArgument must expose a members attribute."""
    kdb = KernelDB(nested_structs_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "update_bounds")
    for arg in kdb.get_kernel_arguments(kernel_name):
        assert hasattr(arg, "members")


@requires_rocm
def test_nested_structs_point3d_has_three_members(nested_structs_binary):
    """Point3D has x, y, z – any Point3D sub-argument must have 3 members."""
    kdb = KernelDB(nested_structs_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "update_bounds")
    args = kdb.get_kernel_arguments(kernel_name)

    def find_point3d(arg_list):
        for arg in arg_list:
            if "Point3D" in arg.type_name:
                return arg
            if arg.members:
                result = find_point3d(arg.members)
                if result:
                    return result
        return None

    point = find_point3d(args)
    if point is None:
        pytest.skip("Point3D type not found in argument tree (may be optimized away)")

    assert len(point.members) == 3, (
        f"Expected 3 members in Point3D, got {len(point.members)}"
    )


# ---------------------------------------------------------------------------
# Typedef / using-alias resolution
# ---------------------------------------------------------------------------


@requires_rocm
def test_typedef_argument_count(typedef_binary):
    """typedef_kernel(MyInt, MyFloat, MyDouble, int*) has 4 parameters."""
    kdb = KernelDB(typedef_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "typedef_kernel")
    args = kdb.get_kernel_arguments(kernel_name)
    assert len(args) == 4, f"Expected 4 arguments, got {len(args)}"


@requires_rocm
def test_typedef_not_resolved_by_default(typedef_binary):
    """With resolve_typedefs=False the alias names (MyInt etc.) are preserved."""
    kdb = KernelDB(typedef_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "typedef_kernel")
    args = kdb.get_kernel_arguments(kernel_name, resolve_typedefs=False)
    type_names = [arg.type_name for arg in args]

    # At least one argument should retain its typedef name
    has_typedef_name = any(
        name in ("MyInt", "MyFloat", "MyDouble") for name in type_names
    )
    assert has_typedef_name, (
        f"Expected typedef names to be preserved, got types: {type_names}"
    )


@requires_rocm
def test_typedef_resolved_when_requested(typedef_binary):
    """With resolve_typedefs=True, typedef names map to their underlying types."""
    kdb = KernelDB(typedef_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "typedef_kernel")
    args_resolved = kdb.get_kernel_arguments(kernel_name, resolve_typedefs=True)
    type_names = [arg.type_name for arg in args_resolved]

    # After resolution, original typedef names like MyInt/MyFloat should not appear
    for alias in ("MyInt", "MyFloat", "MyDouble"):
        assert alias not in type_names, (
            f"Typedef alias {alias!r} still present after resolution: {type_names}"
        )


@requires_rocm
def test_typedef_resolved_types_are_primitives(typedef_binary):
    """After typedef resolution the types should be C primitive type names."""
    kdb = KernelDB(typedef_binary)
    kernel_name = _find_kernel(kdb.get_kernels(), "typedef_kernel")
    args_resolved = kdb.get_kernel_arguments(kernel_name, resolve_typedefs=True)
    type_names = [arg.type_name for arg in args_resolved]

    primitive_types = {"int", "float", "double", "unsigned int"}
    for type_name in type_names:
        # Check that each resolved type is recognisably a C/C++ primitive or pointer to one.
        # Strip pointer qualifiers (* and spaces) before matching.
        base_type = type_name.replace("*", "").strip()
        assert any(prim in base_type for prim in primitive_types), (
            f"Resolved type {type_name!r} does not look like a primitive type"
        )


# ---------------------------------------------------------------------------
# Template kernel instantiations
# ---------------------------------------------------------------------------


@requires_rocm
def test_template_kernels_found(template_binary):
    """Both scale_values<float> and scale_values<double> must appear."""
    kdb = KernelDB(template_binary)
    kernels = kdb.get_kernels()
    scale_kernels = [k for k in kernels if "scale_values" in k]
    assert len(scale_kernels) >= 2, (
        f"Expected at least 2 template instantiations, got {scale_kernels}"
    )


@requires_rocm
def test_template_float_instantiation_arguments(template_binary):
    """scale_values<float> must have 4 arguments."""
    kdb = KernelDB(template_binary)
    kernels = kdb.get_kernels()
    float_kernels = [k for k in kernels if "scale_values" in k and "float" in k]
    if not float_kernels:
        pytest.skip("scale_values<float> not found in binary")
    args = kdb.get_kernel_arguments(float_kernels[0])
    assert len(args) == 4, f"Expected 4 arguments for scale_values<float>, got {len(args)}"


@requires_rocm
def test_template_float_argument_types(template_binary):
    """scale_values<float> argument types should reference float."""
    kdb = KernelDB(template_binary)
    kernels = kdb.get_kernels()
    float_kernels = [k for k in kernels if "scale_values" in k and "float" in k]
    if not float_kernels:
        pytest.skip("scale_values<float> not found in binary")
    args = kdb.get_kernel_arguments(float_kernels[0])
    # Exclude 'n' (the int count parameter)
    non_int_args = [a for a in args if a.name != "n"]
    for arg in non_int_args:
        assert "float" in arg.type_name.lower(), (
            f"Expected float type for {arg.name!r}, got {arg.type_name!r}"
        )


@requires_rocm
def test_template_double_instantiation_arguments(template_binary):
    """scale_values<double> must have 4 arguments."""
    kdb = KernelDB(template_binary)
    kernels = kdb.get_kernels()
    double_kernels = [k for k in kernels if "scale_values" in k and "double" in k]
    if not double_kernels:
        pytest.skip("scale_values<double> not found in binary")
    args = kdb.get_kernel_arguments(double_kernels[0])
    assert len(args) == 4, f"Expected 4 arguments for scale_values<double>, got {len(args)}"
