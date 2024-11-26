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
#pragma once
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolSize.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <elf.h>

#include <assert.h>
#include <cxxabi.h>
#include <dirent.h>
#include <pthread.h>
#include <random>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <filesystem>
#include <sys/syscall.h>   /* For SYS_xxx definitions */
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <limits.h>
#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <fstream>
#include <list>
#include <map>
#include <unordered_set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <thread>
#include <mutex>
#include <utility>
#include <shared_mutex>
#include <filesystem>
#include <ios>
#include <ctime>
#include <algorithm>
#include <regex>
#include <fcntl.h>
#include <sys/stat.h>
#include <link.h>

#include <hsa.h>
#include <hsa_ven_amd_aqlprofile.h>
#include <hsa_ven_amd_loader.h>
#include <amd_comgr/amd_comgr.h>


namespace kernelDB {

typedef struct instruction_s{
    std::string prefix_;
    std::string type_;
    std::string size_;
    std::string inst_;
    std::vector<std::string> operands_;
    std::string disassembly_;
    uint64_t address_;
}instruction_t;


enum parse_mode {
    BEGIN,
    KERNEL,
    BBLOCK,
    BRANCH
};


class __attribute__((visibility("default"))) basicBlock {
public: 
    basicBlock();
    ~basicBlock() = default;
    void addInstruction(const instruction_t& instruction);
private:
    uint16_t block_id;
    std::string disassembly_;
    std::vector<instruction_t> instructions_;
    std::map<std::string, uint64_t> counts_;
};

class __attribute__((visibility("default"))) CDNAKernel {
public:
    CDNAKernel(const std::string& name);
    ~CDNAKernel() = default;
    size_t addBlock(std::unique_ptr<basicBlock> block);
    size_t getBlockCount() { return blocks_.size();}
    std::string getName() { return name_;}
    const std::vector<instruction_t>& getInstructionsForLine(uint64_t);
private:
    std::string name_;
    std::string disassembly_;
    std::vector<std::unique_ptr<basicBlock>> blocks_;
    std::map<uint64_t, std::vector<instruction_t>> line_map_;
};

class __attribute__((visibility("default"))) kernelDB {
public:
    kernelDB(hsa_agent_t agent, const std::string& fileName);
    kernelDB(hsa_agent_t agent, std::vector<uint8_t> bits);
    ~kernelDB();
    bool getBasicBlocks(const std::string& name, std::vector<basicBlock>&);
    const CDNAKernel& getKernel(const std::string& name);
    bool addFile(const std::string& name, hsa_agent_t agent, const std::string& strFilter);
    bool parseDisassembly(const std::string& text);
    void mapDisassemblyToSource(hsa_agent_t agent, const char *elfFilePath);
    void addKernel(std::unique_ptr<CDNAKernel> kernel);
    static void dumpDwarfInfo(const char *elfFilePath, llvm::MemoryBuffer *pVal);
    static amd_comgr_code_object_info_t getCodeObjectInfo(hsa_agent_t agent, std::vector<uint8_t>& bits);
    static void getElfSectionBits(const std::string &fileName, const std::string &sectionName, std::vector<uint8_t>& sectionData );
private:
    parse_mode getLineType(std::string& line);
    static bool isBranch(const std::string& instruction);
private:
    std::map<std::string, std::unique_ptr<CDNAKernel>> kernels_;
    amd_comgr_data_t executable_;
    hsa_agent_t agent_;
    std::string fileName_;
};


static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

static inline size_t split(std::string const& s,
             std::vector<std::string> &container,
             const char * delimiter,
             bool keepBlankFields)
{
    size_t n = 0;
    std::string::const_iterator it = s.begin(), end = s.end(), first;
    for (first = it; it != end; ++it)
    {
        // Examine each character and if it matches the delimiter
        if (*delimiter == *it)
        {
            if (keepBlankFields || first != it)
            {
                // extract the current field from the string and
                // append the current field to the given container
                container.push_back(std::string(first, it));
                ++n;
                
                // skip the delimiter
                first = it + 1;
            }
            else
            {
                ++first;
            }
        }
    }
    if (keepBlankFields || first != it)
    {
        // extract the last field from the string and
        // append the last field to the given container
        container.push_back(std::string(first, it));
        ++n;
    }
    return n;
}
}//kernelDB
