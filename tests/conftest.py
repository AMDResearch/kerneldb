# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

import shutil

import pytest


def _hipcc_available():
    return shutil.which("hipcc") is not None


def _gpu_available():
    try:
        from kerneldb import _kerneldb

        if _kerneldb.hsa_init() != 0:
            return False
        return _kerneldb.get_first_gpu_agent().handle != 0
    except Exception:
        return False


requires_rocm = pytest.mark.skipif(
    not (_hipcc_available() and _gpu_available()),
    reason="Requires hipcc and a GPU",
)
