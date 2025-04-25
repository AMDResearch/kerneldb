
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

  test.addFile(file_path.c_str(), gpu_agent, "");

  return 0;
}
