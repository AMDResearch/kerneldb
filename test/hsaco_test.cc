
#include <hsa/hsa.h>

#include <filesystem>
#include <iostream>

#include "include/kernelDB.h"

hsa_agent_t get_first_gpu_agent() {
  hsa_agent_t agent = {};

  hsa_status_t status = hsa_iterate_agents(
      [](hsa_agent_t a, void* data) -> hsa_status_t {
        hsa_device_type_t type;
        if (hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &type) !=
            HSA_STATUS_SUCCESS) {
          return HSA_STATUS_ERROR;
        }

        if (type == HSA_DEVICE_TYPE_GPU) {
          *reinterpret_cast<hsa_agent_t*>(data) = a;
          return HSA_STATUS_INFO_BREAK;
        }

        return HSA_STATUS_SUCCESS;
      },
      &agent);

  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
    const char* err_str = nullptr;
    hsa_status_string(status, &err_str);
    std::cerr << "Failed to find a GPU agent. HSA error: "
              << (err_str ? err_str : "Unknown error") << "\n";
    std::exit(1);
  }
  return agent;
}

int main() {
  hsa_init();

  std::filesystem::path file_path =
      "/opt/rocm-6.3.1/lib/hipblaslt/library/"
      "Kernels.so-000-gfx90a-xnack+.hsaco";

  if (!std::filesystem::exists(file_path)) {
    std::cout << "File " << file_path << " does not exist.\n";
    return 1;
  }

  auto gpu_agent = get_first_gpu_agent();

  kernelDB::kernelDB test(gpu_agent);
  std::cout << "Adding the file" << std::endl;
  auto result = test.addFile(file_path.c_str(), gpu_agent, "");

  std::cout << "Result: " << result << std::endl;
  std::vector<std::string> kernels;
  std::vector<uint32_t> lines;

  std::cout << "Get kernels: " << std::endl;
  test.getKernels(kernels);

  std::cout << "Number of kernels: " << kernels.size() << std::endl;
  int kenrel_id = 0;
  for (const auto& kernel_str : kernels) {
    auto& kernel = test.getKernel(kernel_str);

    std::cout << kernel.getName() << std::endl;

    std::vector<std::string> outputLine;
    const auto& bb = kernel.getBasicBlocks();

    std::cout << "Number of lines: " << outputLine.size() << std::endl;
    kenrel_id++;

    for (const auto& b : bb) {
      const auto& isa = b->getInstructions();
      for (auto& inst : isa){
        std::cout << inst.disassembly_ << std::endl;
      }
    }

    if (kenrel_id  == 2){
      break;
    }
    std::vector<uint32_t> lines;
    // const auto gcn = test.getInstructions(kernel, lines);
    // for (auto i : gcn){
    //   std::cout << i.disassembly_ << std::endl;
    // }
    // for (auto& line : lines) {
    //   std::cout << "Line for " << kernel << " " << line << std::endl;
    //   try {
    //     // Old Style
    //     const auto& inst = test.getInstructionsForLine(kernel, line);
    //     for (size_t idx = 0; idx < inst.size(); idx++)
    //     // for (const auto& item : inst)
    //     {
    //       std::cout << "Default Disassembly[" << inst[idx].column_
    //                 << "]: " << inst[idx].disassembly_ << std::endl;
    //       std::cout << test.getFileName(kernel, inst[idx].path_id_)
    //                 << std::endl;
    //     }

    //     // Filtered
    //     std::vector<kernelDB::instruction_t> filtered =
    //         test.getInstructionsForLine(kernel, line,
    //                                     std::string(".*(load|store).*"));
    //     for (size_t idx = 0; idx < filtered.size(); idx++)
    //     // for (const auto& item : inst)
    //     {
    //       std::cout << "Filtered Disassembly: " << filtered[idx].disassembly_
    //                 << std::endl;
    //     }

    //   } catch (std::runtime_error e) {
    //     std::cout << "Error: " << e.what() << std::endl;
    //   }
    // }
  }
  return 0;
}
