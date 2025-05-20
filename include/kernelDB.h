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
extern "C"{
#include "libdwarf.h"
#include "dwarf.h"
}

// Structure to hold source location info
struct SourceLocation {
    std::string fileName;
    Dwarf_Unsigned lineNumber;
    Dwarf_Unsigned columnNumber;

    SourceLocation(const std::string& file = "", Dwarf_Unsigned line = 0, Dwarf_Unsigned col = 0)
        : fileName(file), lineNumber(line), columnNumber(col) {}
};

bool buildDwarfAddressMap(const char* filename, size_t offset, size_t hsaco_length, std::map<Dwarf_Addr, SourceLocation>& addressMap);
SourceLocation getSourceLocation(std::map<Dwarf_Addr, SourceLocation>& addrMap, Dwarf_Addr addr);

namespace kernelDB {

class basicBlock;
    
std::string getKernelName(const std::string& name);

typedef struct instruction_s{
    std::string prefix_;
    std::string type_;
    std::string size_;
    std::string inst_;
    std::vector<std::string> operands_;
    std::string disassembly_;
    uint64_t address_;
    uint32_t line_;
    uint32_t column_;
    size_t path_id_;
    std::string file_name_;
    basicBlock *block_;
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
    const std::vector<instruction_t>& getInstructions();
    std::vector<instruction_t>& getModifiableInstructions();
private:
private:
    uint16_t block_id;
    std::string disassembly_;
    std::vector<instruction_t> instructions_;
    std::map<std::string, uint64_t> counts_;
    std::shared_mutex mutex_;
};


class __attribute__((visibility("default"))) CDNAKernel {
public:
    CDNAKernel(const std::string& name);
    ~CDNAKernel() = default;
    size_t addBlock(uint32_t global_index, std::unique_ptr<basicBlock> block);
    size_t getBlockCount();
    std::string getName() { return name_;}
    const std::vector<std::unique_ptr<basicBlock>>& getBasicBlocks() {return blocks_;}
    void addInstructionForLine(uint64_t, const instruction_t& instruction);
    void addLine(uint32_t line, const instruction_t& instruction);
    size_t addFileName(const std::string& name);
    void getLineNumbers(std::vector<uint32_t>& out);
    const std::vector<instruction_t>& getInstructionsForLine(uint32_t line);
    std::vector<instruction_t> getInstructionsForLine(uint32_t line, const std::string& match);
    std::string getFileName(size_t index) {assert(index <= file_names_.size()); return file_names_[index-1];}
    const basicBlock *getBasicBlock(uint32_t idx) { assert(idx < blocks_.size()); return blocks_[idx].get();}
    void getSourceCode(std::vector<std::string>& outputLines);
private:
    std::string name_;
    std::string disassembly_;
    std::vector<std::unique_ptr<basicBlock>> blocks_;
    std::map<uint32_t, std::vector<instruction_t>> line_map_;
    std::map<uint32_t, basicBlock *> block_map_;
    std::map<std::string, size_t> file_map_;
    std::vector<std::string> file_names_;
    std::shared_mutex mutex_;
};

class __attribute__((visibility("default"))) kernelDB {
public:
    kernelDB(hsa_agent_t agent, const std::string& fileName);
    kernelDB(hsa_agent_t agent);
    ~kernelDB();
    bool getBasicBlocks(const std::string& name, std::vector<basicBlock>&);
    CDNAKernel& getKernel(const std::string& name);
    bool addFile(const std::string& name, hsa_agent_t agent, const std::string& strFilter);
    bool parseDisassembly(const std::string& text);
    void mapDisassemblyToSource(hsa_agent_t agent, const char *elfFilePath);
    bool addKernel(std::unique_ptr<CDNAKernel> kernel);
    const std::vector<instruction_t>& getInstructionsForLine(const std::string& kernel_name, uint32_t line);
    std::vector<instruction_t> getInstructionsForLine(const std::string& kernel_name, uint32_t line, const std::string& match);
    void getKernels(std::vector<std::string>& out);
    void getKernelLines(const std::string& kernel, std::vector<uint32_t>& out);
    std::string getFileName(const std::string& kernel, size_t index);
    static amd_comgr_code_object_info_t getCodeObjectInfo(hsa_agent_t agent, std::vector<uint8_t>& bits);
    static void getElfSectionBits(const std::string &fileName, const std::string &sectionName, size_t& offset, std::vector<uint8_t>& sectionData );
private:
    void buildLineMap(size_t offset, size_t hsaco_length, const char *elfFilePath);
    parse_mode getLineType(std::string& line);
    static bool isBranch(const std::string& instruction);
private:
    std::map<std::string, std::unique_ptr<CDNAKernel>> kernels_;
    amd_comgr_data_t executable_;
    hsa_agent_t agent_;
    std::string fileName_;
    std::shared_mutex mutex_;
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
