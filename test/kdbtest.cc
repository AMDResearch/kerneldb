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
                    auto inst = test.getInstructionsForLine(kernel, line);
                    for (auto item : inst)
                    {
                        std::cout << "Disassembly: " << item.disassembly_ << std::endl;
                    }
                }
            }
        }
    }
    else
        std::cout << "Usage: kdbtest <hsaco or HIP binary to test>\n";
}
