# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build kerneldb")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

        # Create build directory
        build_temp = Path(self.build_temp).absolute()
        build_temp.mkdir(parents=True, exist_ok=True)

        # CMake configuration
        import sysconfig
        python_include = sysconfig.get_path('include')
        source_pkg_dir = Path(__file__).parent / 'kerneldb'
        source_pkg_dir.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={source_pkg_dir}',
            f'-DCMAKE_INSTALL_PREFIX={source_pkg_dir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DPython3_EXECUTABLE={sys.executable}',
            f'-DPython3_INCLUDE_DIR={python_include}',
            '-DCMAKE_BUILD_TYPE=Release',
            '-DBUILD_PYTHON_BINDINGS=ON',
        ]

        # Build arguments
        build_args = ['--config', 'Release']

        # Add parallel build flag
        if hasattr(self, 'parallel') and self.parallel:
            build_args += ['-j', str(self.parallel)]
        else:
            build_args += ['-j4']

        # Run CMake
        subprocess.check_call(
            ['cmake', str(Path(__file__).parent.absolute())] + cmake_args,
            cwd=build_temp
        )

        # Build both the C++ library and Python module
        print(f"\n{'='*60}")
        print(f"Building directly into: {source_pkg_dir}")
        print(f"{'='*60}\n")

        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=build_temp
        )

        print(f"\n{'='*60}")
        print("✓ Build complete! No copying needed.")
        print(f"{'='*60}\n")


setup(
    ext_modules=[CMakeExtension('kerneldb')],
    cmdclass={'build_ext': CMakeBuild},
)
