// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "include/kernelDB.h"

namespace py = pybind11;

PYBIND11_MODULE(_kerneldb, m) {
    m.doc() = "Python bindings for kernelDB - CDNA kernel analysis library";

    // Bind hsa_agent_t as an opaque type
    py::class_<hsa_agent_t>(m, "HsaAgent")
        .def(py::init<>())
        .def_readwrite("handle", &hsa_agent_t::handle);

    // Bind KernelArgument struct (recursive - used for both top-level args and struct members)
    py::class_<KernelArgument>(m, "KernelArgument")
        .def(py::init<>())
        .def(py::init<const std::string&, const std::string&, size_t, size_t, size_t, uint32_t>(),
             py::arg("name"), py::arg("type"), py::arg("size"), py::arg("offset"),
             py::arg("alignment"), py::arg("position"))
        .def_readonly("name", &KernelArgument::name)
        .def_readonly("type", &KernelArgument::type)
        .def_readonly("size", &KernelArgument::size)
        .def_readonly("offset", &KernelArgument::offset)      // 0 for top-level, actual offset for members
        .def_readonly("alignment", &KernelArgument::alignment) // relevant for top-level args
        .def_readonly("position", &KernelArgument::position)   // 0-based for top-level, 0 for members
        .def_readonly("members", &KernelArgument::members)     // recursive!
        .def("__repr__", [](const KernelArgument &arg) {
            return "<KernelArgument '" + arg.name + "': " + arg.type +
                   " (size=" + std::to_string(arg.size) +
                   ", offset=" + std::to_string(arg.offset) +
                   ", pos=" + std::to_string(arg.position) + ")>";
        });

    // Bind instruction_t struct
    py::class_<kernelDB::instruction_t>(m, "Instruction")
        .def(py::init<>())
        .def_readonly("prefix", &kernelDB::instruction_t::prefix_)
        .def_readonly("type", &kernelDB::instruction_t::type_)
        .def_readonly("size", &kernelDB::instruction_t::size_)
        .def_readonly("inst", &kernelDB::instruction_t::inst_)
        .def_readonly("operands", &kernelDB::instruction_t::operands_)
        .def_readonly("disassembly", &kernelDB::instruction_t::disassembly_)
        .def_readonly("address", &kernelDB::instruction_t::address_)
        .def_readonly("line", &kernelDB::instruction_t::line_)
        .def_readonly("column", &kernelDB::instruction_t::column_)
        .def_readonly("path_id", &kernelDB::instruction_t::path_id_)
        .def_readonly("file_name", &kernelDB::instruction_t::file_name_)
        .def("__repr__", [](const kernelDB::instruction_t &inst) {
            return "<Instruction: " + inst.disassembly_ + ">";
        });

    // Bind basicBlock class
    py::class_<kernelDB::basicBlock>(m, "BasicBlock")
        .def("get_instructions", &kernelDB::basicBlock::getInstructions,
             "Get all instructions in this basic block",
             py::return_value_policy::reference_internal)
        .def("__len__", [](kernelDB::basicBlock &block) {
            return block.getInstructions().size();
        })
        .def("__repr__", [](kernelDB::basicBlock &block) {
            return "<BasicBlock with " + std::to_string(block.getInstructions().size()) + " instructions>";
        });

    // Bind CDNAKernel class
    py::class_<kernelDB::CDNAKernel>(m, "CDNAKernel")
        .def("get_name", &kernelDB::CDNAKernel::getName,
             "Get the kernel name")
        .def("get_block_count", &kernelDB::CDNAKernel::getBlockCount,
             "Get the number of basic blocks")
        .def("get_basic_blocks", &kernelDB::CDNAKernel::getBasicBlocks,
             "Get all basic blocks",
             py::return_value_policy::reference_internal)
        .def("get_line_numbers", [](kernelDB::CDNAKernel &kernel) {
            std::vector<uint32_t> lines;
            kernel.getLineNumbers(lines);
            return lines;
        }, "Get all source line numbers")
        .def("get_instructions_for_line",
             py::overload_cast<uint32_t>(&kernelDB::CDNAKernel::getInstructionsForLine),
             "Get instructions for a source line",
             py::return_value_policy::reference_internal)
        .def("get_instructions_for_line",
             py::overload_cast<uint32_t, const std::string&>(&kernelDB::CDNAKernel::getInstructionsForLine),
             "Get instructions for a source line with regex filter",
             py::arg("line"), py::arg("pattern"))
        .def("get_file_name", &kernelDB::CDNAKernel::getFileName,
             "Get source file name by index",
             py::arg("index"))
        .def("get_source_code", [](kernelDB::CDNAKernel &kernel) {
            std::vector<std::string> lines;
            kernel.getSourceCode(lines);
            return lines;
        }, "Get source code lines")
        .def("get_disassembly", &kernelDB::CDNAKernel::getDisassembly,
             "Get full disassembly")
        .def("get_arguments", &kernelDB::CDNAKernel::getArguments,
             "Get kernel arguments",
             py::return_value_policy::reference_internal)
        .def("has_arguments", &kernelDB::CDNAKernel::hasArguments,
             "Check if kernel has argument information")
        .def("__repr__", [](kernelDB::CDNAKernel &kernel) {
            return "<CDNAKernel: " + kernel.getName() + ">";
        });

    // Bind kdb_arch_descriptor_t
    py::class_<kdb_arch_descriptor_t>(m, "ArchDescriptor")
        .def(py::init<>())
        .def_readonly("isa", &kdb_arch_descriptor_t::isa_)
        .def_readonly("xccs", &kdb_arch_descriptor_t::xccs_)
        .def_readonly("ses", &kdb_arch_descriptor_t::ses_)
        .def_readonly("cus", &kdb_arch_descriptor_t::cus_)
        .def_readonly("simds", &kdb_arch_descriptor_t::simds_)
        .def_readonly("wave_size", &kdb_arch_descriptor_t::wave_size_)
        .def_readonly("max_waves", &kdb_arch_descriptor_t::max_waves_)
        .def("__repr__", [](const kdb_arch_descriptor_t &arch) {
            return "<ArchDescriptor: " + arch.isa_ + ">";
        });

    // Bind main kernelDB class
    py::class_<kernelDB::kernelDB>(m, "KernelDB")
        .def(py::init<hsa_agent_t, const std::string&>(),
             "Create KernelDB instance with agent and filename",
             py::arg("agent"), py::arg("filename"))
        .def(py::init<hsa_agent_t>(),
             "Create KernelDB instance with agent (searches process)",
             py::arg("agent"))
        .def("get_kernel", &kernelDB::kernelDB::getKernel,
             "Get a kernel by name",
             py::arg("name"),
             py::return_value_policy::reference_internal)
        .def("get_kernels", [](kernelDB::kernelDB &db) {
            std::vector<std::string> kernels;
            db.getKernels(kernels);
            return kernels;
        }, "Get all kernel names")
        .def("get_kernel_lines", [](kernelDB::kernelDB &db, const std::string &kernel) {
            std::vector<uint32_t> lines;
            db.getKernelLines(kernel, lines);
            return lines;
        }, "Get source lines for a kernel",
           py::arg("kernel"))
        .def("get_instructions_for_line",
             py::overload_cast<const std::string&, uint32_t>(
                 &kernelDB::kernelDB::getInstructionsForLine),
             "Get instructions for a line in a kernel",
             py::arg("kernel_name"), py::arg("line"),
             py::return_value_policy::reference_internal)
        .def("get_instructions_for_line",
             py::overload_cast<const std::string&, uint32_t, const std::string&>(
                 &kernelDB::kernelDB::getInstructionsForLine),
             "Get instructions for a line in a kernel with regex filter",
             py::arg("kernel_name"), py::arg("line"), py::arg("pattern"))
        .def("get_file_name", &kernelDB::kernelDB::getFileName,
             "Get source file name",
             py::arg("kernel"), py::arg("index"))
        .def("get_kernel_arguments", &kernelDB::kernelDB::getKernelArguments,
             "Get kernel arguments",
             py::arg("kernel_name"))
        .def("__repr__", [](const kernelDB::kernelDB &db) {
            return "<KernelDB instance>";
        });

    // Helper function to initialize HSA
    m.def("hsa_init", []() -> int {
        return static_cast<int>(hsa_init());
    }, "Initialize HSA runtime");

    // Helper function to get first GPU agent
    m.def("get_first_gpu_agent", []() -> hsa_agent_t {
        hsa_agent_t agent;
        agent.handle = 0;

        auto callback = [](hsa_agent_t ag, void* data) -> hsa_status_t {
            hsa_device_type_t device_type;
            hsa_agent_get_info(ag, HSA_AGENT_INFO_DEVICE, &device_type);

            if (device_type == HSA_DEVICE_TYPE_GPU) {
                hsa_agent_t* ret = reinterpret_cast<hsa_agent_t*>(data);
                *ret = ag;
                return HSA_STATUS_INFO_BREAK;  // Stop iteration
            }
            return HSA_STATUS_SUCCESS;
        };

        hsa_iterate_agents(callback, &agent);
        return agent;
    }, "Get the first GPU agent");

    // Version info
    m.attr("__version__") = "1.0.0";
}

