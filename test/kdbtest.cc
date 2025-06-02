
/******************************************************************************
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/
#include <iostream>
#include <string>
#include "include/kernelDB.h"

int main(int argc, char **argv)
{
    hsa_init();
    if (argc > 1)
    {
        try
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
                    auto& thisKernel = test.getKernel(kernel);
                    std::cout << "Blocks for kernel " << kernel << ":\n\t" << thisKernel.getBlockCount() << std::endl;
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
        catch(const std::runtime_error& exception)
        {
            std::cout << "Cannot process " << argv[1] << "\n\t (Usually that means there is no debug info in the file.)\n";
        }
    }
    else
        std::cout << "Usage: kdbtest <hsaco or HIP binary to test>\n";
}
