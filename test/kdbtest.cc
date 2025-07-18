
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

std::vector<std::string> readFileLines(const std::string& filename, uint32_t startLine, uint32_t endLine) {
    std::vector<std::string> lines;

    // Validate input parameters
    if (startLine == 0 || startLine > endLine) {
        throw std::invalid_argument("Invalid line range: startLine must be >= 1 and <= endLine");
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::string line;
    uint32_t currentLine = 0;

    // Read until we reach startLine or EOF
    while (currentLine < startLine - 1 && std::getline(file, line)) {
        ++currentLine;
    }

    // Read lines from startLine to endLine inclusive
    while (currentLine < endLine && std::getline(file, line)) {
        lines.push_back(line);
        ++currentLine;
    }

    file.close();
    return lines;
}


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
                    if (kernel.find("_amd_crk") != std::string::npos)
                        continue;
                    std::vector<uint32_t> lines;
                    test.getKernelLines(kernel, lines);
                    auto& thisKernel = test.getKernel(kernel);
                    std::cout << "Blocks for kernel " << kernel << ":\n\t" << thisKernel.getBlockCount() << std::endl;
                    const auto& blocks = thisKernel.getBasicBlocks();
                    uint32_t idx = 0;
                    for (auto& block : blocks)
                    {
                        kernelDB::basicBlock *thisBlock = block.get();
                        auto inst = thisBlock->getInstructions();
                        std::cout << "\tBlock " << idx++ << std::endl;
                        thisKernel.printBlock(std::cout, thisBlock, std::string("label"));
                        for (auto& one_inst : inst)
                        {
                            std::cout << "\t\t[" << one_inst.line_ << ":" << one_inst.column_ << "] " << one_inst.disassembly_ << std::endl;
                        }
                    }
                    for (auto& line : lines)
                    {
                        std::cout << "Line for " << kernel << " " << line << std::endl;
                        try
                        {
                            // Old Style
                            const auto& inst = test.getInstructionsForLine(kernel, line);
                            for(size_t idx = 0; idx < inst.size(); idx++)
                            {
                                std::cout << "Default Disassembly[" << inst[idx].line_ << "|" << inst[idx].column_ << "]: " << inst[idx].disassembly_ << std::endl;
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
                            std::cout << "Error: "  << e.what() << std::endl;
                        }
                    }
                }
            }
        }
        catch(const std::runtime_error& exception)
        {
            std::cout << "Cannot process " << argv[1] << "\n\t (Usually that means there is no debug info in the file.)\n";
            std::cout << exception.what() << std::endl;
        }
    }
    else
        std::cout << "Usage: kdbtest <hsaco or HIP binary to test>\n";
}
