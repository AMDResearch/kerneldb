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
#include <sstream>
#include "include/kernelDB.h"

std::vector<std::string> disassembly_params = {"-d ", "--arch-name=amdgcn"};

void readFileToString(const std::string& filename, std::string& content) {
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Get file size
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read file contents into string
    content.resize(size, ' ');
    file.read(&content[0], size);

    file.close();
    return;
}


std::string init_disassembler_path()
{
    const char *path = getenv("ROCM_PATH");
    std::string result;
    if (path)
        result = std::string(path) + "/llvm/bin/llvm-objdump";
    else
        result = "/opt/rocm/llvm/bin/llvm-objdump";
    return result;
}

std::string disassembler = init_disassembler_path();

bool getDisassembly(hsa_agent_t agent, const std::string& fileName, std::string& out)
{
    std::vector<std::string> parms = disassembly_params;
    std::stringstream ss;
    char name[64];
    memset(name, 0, sizeof(name));
    hsa_status_t status = hsa_agent_get_info(agent,HSA_AGENT_INFO_NAME, name); 
    if (status == HSA_STATUS_SUCCESS)
    {
        ss << "--mcpu=" << name;
        parms.push_back(ss.str());
        parms.push_back(fileName);
         // Create a temporary file using tmpnam (note: tmpnam is not the most secure option)
        char temp_filename[L_tmpnam];
        if (tmpnam(temp_filename) == nullptr) 
            throw std::runtime_error("Failed to generate temporary filename");
        else if (invokeProgram(disassembler, parms, temp_filename))
        {
            // read file contents here
            readFileToString(temp_filename, out);
            unlink(temp_filename);
            return out.length() != 0;
        }
        else
            return false;
    }
    else
        return false;
    
    return true;
}

bool invokeProgram(const std::string& programName, const std::vector<std::string>& params, const std::string& outputFileName) {
    // Construct the command string
    std::stringstream command;
    command << programName;
    for (const auto& param : params) {
        // Basic escaping of parameters (assumes no spaces in params; enhance if needed)
        command << " " << param;
    }
    // Redirect stdout to outputFileName
    command << " > " << outputFileName;

    // Execute the command synchronously using popen
    // Note: popen is used here to ensure compatibility with POSIX systems
    FILE* pipe = popen(command.str().c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("Failed to execute command: " + command.str());
    }

    // Since we're redirecting stdout to a file, we don't need to read from the pipe
    // Just wait for the command to finish
    int returnCode = pclose(pipe);
    if (returnCode != 0) {
        // Non-zero return code indicates failure
        throw std::runtime_error("Command failed with return code: " + std::to_string(returnCode));
    }

    // Verify the output file was created
    std::ifstream outputFile(outputFileName);
    if (!outputFile.good()) {
        throw std::runtime_error("Output file was not created or is inaccessible: " + outputFileName);
    }
    outputFile.close();

    return true;
}
