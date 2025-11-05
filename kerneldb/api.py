# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

"""
Python API for kernelDB

Provides high-level Python classes wrapping the kernelDB C++ library via pybind11.
"""

from typing import List, Optional
from dataclasses import dataclass

# Import the pybind11 compiled module
try:
    from . import _kerneldb
except ImportError as e:
    raise ImportError(
        "Failed to import _kerneldb module. "
        "Make sure kernelDB was compiled with Python bindings. "
        f"Error: {e}"
    )


# Re-export pybind11 types with better names
Instruction = _kerneldb.Instruction
BasicBlock = _kerneldb.BasicBlock
CDNAKernel = _kerneldb.CDNAKernel
ArchDescriptor = _kerneldb.ArchDescriptor
HsaAgent = _kerneldb.HsaAgent


class KernelDB:
    """
    Main interface for analyzing HIP/ROCm kernel binaries

    This class loads HSACO or HIP fat binary files and provides methods
    to query instruction-level information mapped to source lines.
    """

    def __init__(self, binary_path: Optional[str] = None, agent_id: Optional[int] = None):
        """
        Initialize KernelDB

        Args:
            binary_path: Path to HSACO or HIP binary file. If empty string or None,
                        will search in the running process for fat binaries.
            agent_id: HSA agent handle (if None, will use first GPU)
        """
        # Initialize HSA
        status = _kerneldb.hsa_init()
        if status != 0:
            raise RuntimeError(f"HSA initialization failed with status {status}")

        # Get HSA agent
        if agent_id is not None:
            self.agent = HsaAgent()
            self.agent.handle = agent_id
        else:
            self.agent = _kerneldb.get_first_gpu_agent()
            if self.agent.handle == 0:
                raise RuntimeError("No GPU agent found")

        # Create kernelDB instance (analysis happens in constructor)
        binary_path = binary_path or ""
        self._kdb = _kerneldb.KernelDB(self.agent, binary_path)
        self.binary_path = binary_path

    def get_kernels(self) -> List[str]:
        """
        Get list of all kernel names found in the binary

        Returns:
            List of kernel names
        """
        return self._kdb.get_kernels()

    def get_kernel(self, name: str):
        """
        Get a kernel object by name

        Args:
            name: Kernel name

        Returns:
            Kernel wrapper object with convenient properties
        """
        cdna_kernel = self._kdb.get_kernel(name)
        return Kernel(cdna_kernel, self)

    def get_kernel_lines(self, kernel_name: str) -> List[int]:
        """
        Get all source line numbers that have instructions in a kernel

        Args:
            kernel_name: Name of the kernel

        Returns:
            List of line numbers
        """
        return self._kdb.get_kernel_lines(kernel_name)

    def get_instructions_for_line(
        self,
        kernel_name: str,
        line: int,
        filter_pattern: Optional[str] = None
    ) -> List[Instruction]:
        """
        Get instructions for a specific source line in a kernel

        Args:
            kernel_name: Name of the kernel
            line: Source line number
            filter_pattern: Optional regex to filter instructions (e.g., ".*(load|store).*")

        Returns:
            List of Instruction objects
        """
        if filter_pattern:
            return self._kdb.get_instructions_for_line(kernel_name, line, filter_pattern)
        return self._kdb.get_instructions_for_line(kernel_name, line)

    def get_file_name(self, kernel_name: str, index: int) -> str:
        """
        Get source file name by index

        Args:
            kernel_name: Name of the kernel
            index: File path index

        Returns:
            File path string
        """
        return self._kdb.get_file_name(kernel_name, index)


class Kernel:
    """
    High-level wrapper around CDNAKernel with convenient properties

    This provides a more Pythonic interface similar to the Nexus API.
    """

    def __init__(self, cdna_kernel: CDNAKernel, kdb_instance: KernelDB):
        self._kernel = cdna_kernel
        self._kdb = kdb_instance
        self.name = cdna_kernel.get_name()

    @property
    def signature(self) -> str:
        """Get kernel signature (name for now)"""
        return self.name

    @property
    def lines(self) -> List[int]:
        """Get all line numbers that have instructions in this kernel"""
        return self._kernel.get_line_numbers()

    @property
    def assembly(self) -> List[str]:
        """Get the full disassembly of this kernel"""
        instructions = []
        for line in self.lines:
            line_instructions = self._kernel.get_instructions_for_line(line)
            instructions.extend([inst.disassembly for inst in line_instructions])
        return instructions

    @property
    def hip_source(self) -> List[str]:
        """Get the HIP/source code lines for this kernel"""
        return self._kernel.get_source_code()

    @property
    def files(self) -> List[str]:
        """Get list of source files referenced by this kernel"""
        file_set = set()
        for line in self.lines:
            instructions = self._kernel.get_instructions_for_line(line)
            for inst in instructions:
                if inst.file_name:
                    file_set.add(inst.file_name)
        return sorted(list(file_set))

    def get_instructions_for_line(self, line: int, filter_pattern: Optional[str] = None) -> List[Instruction]:
        """
        Get instructions for a specific source line

        Args:
            line: Source line number
            filter_pattern: Optional regex pattern to filter instructions

        Returns:
            List of Instruction objects
        """
        if filter_pattern:
            return self._kernel.get_instructions_for_line(line, filter_pattern)
        return self._kernel.get_instructions_for_line(line)

    def get_basic_blocks(self) -> List[BasicBlock]:
        """Get all basic blocks in this kernel"""
        return list(self._kernel.get_basic_blocks())


class KernelTrace:
    """
    Container for traced kernel information

    Compatible with Nexus-style trace interface for easy example adaptation.
    """

    def __init__(self):
        self.kernels: List[Kernel] = []

    def __len__(self) -> int:
        return len(self.kernels)

    def __iter__(self):
        return iter(self.kernels)

    def __getitem__(self, idx: int) -> Kernel:
        return self.kernels[idx]

    def save(self, filename: str):
        """Save trace to JSON file"""
        import json

        data = {
            "kernels": [
                {
                    "name": kernel.name,
                    "signature": kernel.signature,
                    "lines": kernel.lines,
                    "files": kernel.files,
                    "assembly_count": len(kernel.assembly),
                    "hip_source_lines": len(kernel.hip_source),
                }
                for kernel in self.kernels
            ]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Trace saved to {filename}")


# High-level API compatible with Nexus-style usage
def analyze_binary(binary_path: str) -> KernelTrace:
    """
    Analyze a HIP binary and return kernel trace information

    Args:
        binary_path: Path to HSACO or HIP binary

    Returns:
        KernelTrace object containing kernel information
    """
    kdb = KernelDB(binary_path)

    trace = KernelTrace()
    for kernel_name in kdb.get_kernels():
        cdna_kernel = kdb.get_kernel(kernel_name)
        kernel = Kernel(cdna_kernel, kdb)
        trace.kernels.append(kernel)

    return trace
