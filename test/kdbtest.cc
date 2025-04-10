#include <iostream>
#include <string>
#include "include/kernelDB.h"

int main(int argc, char **argv)
{
    hsa_init();
    if (argc > 1)
    {
        std::string str(argv[1]);
        hsa_agent_t agent;
        if(hsa_iterate_agents ([](hsa_agent_t agent, void *data){
                    hsa_agent_t *this_agent  = reinterpret_cast<hsa_agent_t *>(data);
                    *this_agent = agent;
                    return HSA_STATUS_SUCCESS;
                }, reinterpret_cast<void *>(&agent))== HSA_STATUS_SUCCESS)
        {
            kernelDB::kernelDB test(agent,str);
            std::vector<std::string> kernels;
            std::vector<uint32_t> lines;
            test.getKernels(kernels);
            for (auto kernel : kernels)
            {
                std::vector<uint32_t> lines;
                test.getKernelLines(kernel, lines);
                for (auto& line : lines)
                {
                    std::cout << "Line for " << kernel << " " << line << std::endl;
                    try
                    {
                        // Old Style
                        const auto& inst = test.getInstructionsForLine(kernel, line);
                        for(size_t idx = 0; idx < inst.size(); idx++)
                        //for (const auto& item : inst)
                        {
                            std::cout << "Default Disassembly[" << inst[idx].column_ << "]: " << inst[idx].disassembly_ << std::endl;
                                std::cout << test.getFileName(kernel, inst[idx].path_id_) << std::endl;
                        }

                        // Filtered
                        std::vector<kernelDB::instruction_t> filtered = test.getInstructionsForLine(kernel, line, std::string(".*(load|store).*"));
                        for(size_t idx = 0; idx < filtered.size(); idx++)
                        //for (const auto& item : inst)
                        {
                            std::cout << "Filtered Disassembly: " << filtered[idx].disassembly_ << std::endl;
                        }

                    }
                    catch(std::runtime_error e)
                    {
                        std::cout << "Error: " << e.what() << std::endl;
                    }
                }
            }
        }
    }
    else
        std::cout << "Usage: kdbtest <hsaco or HIP binary to test>\n";
}
