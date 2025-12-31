# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
import subprocess
import shutil
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
        
        # Build to a temporary directory
        cmake_output_dir = build_temp / 'output'
        cmake_output_dir.mkdir(parents=True, exist_ok=True)

        # Get Python library path for linking
        python_lib = sysconfig.get_config_var('LIBDIR')
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={cmake_output_dir}',
            f'-DCMAKE_INSTALL_PREFIX={cmake_output_dir}',
            f'-DPython3_EXECUTABLE={sys.executable}',
            f'-DPython3_INCLUDE_DIR={python_include}',
            f'-DPython3_LIBRARY={python_lib}/libpython{python_version}.so',
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
        print(f"Building into: {cmake_output_dir}")
        print(f"{'='*60}\n")

        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=build_temp
        )

        print(f"\n{'='*60}")
        print("✓ Build complete!")
        print(f"Extension directory: {extdir}")
        print(f"{'='*60}\n")

        # Copy the built files to the extension directory
        extdir.mkdir(parents=True, exist_ok=True)
        
        # Copy the Python extension module
        for so_file in cmake_output_dir.glob('_kerneldb*.so'):
            dest = extdir / so_file.name
            print(f"Copying {so_file} -> {dest}")
            shutil.copy2(so_file, dest)
        
        # Copy the C++ library (needed by the Python module)
        for lib_file in cmake_output_dir.glob('libkernelDB64.so*'):
            dest = extdir / lib_file.name
            print(f"Copying {lib_file} -> {dest}")
            shutil.copy2(lib_file, dest)


setup(
    ext_modules=[CMakeExtension('kerneldb._kerneldb')],
    cmdclass={'build_ext': CMakeBuild},
)
