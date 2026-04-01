
/******************************************************************************
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
#include <elf.h>
extern "C"
{
#include "dwarf.h"
#include "libdwarf.h"
}

#include "include/kernelDB.h"



#define FATBIN_SECTION ".hip_fatbin"

#define CHECK_COMGR(call)                                                                          \
  if (amd_comgr_status_s status = call) {                                                          \
    const char* reason = "";                                                                       \
    amd_comgr_status_string(status, &reason);                                                      \
    std::cerr << __LINE__ << " code: " << status << std::endl;                                     \
    std::cerr << __LINE__ << " failed: " << reason << std::endl;                                   \
    exit(1);                                                                                       \
  }


namespace kernelDB {


static std::unordered_set<std::string> branch_instructions = {"s_branch", "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz", "s_cbranch_vccnz", "s_cbranch_execz", "s_cbranch_execnz",
    "s_setpc_b64", "s_call_b64", "s_return_b64", "s_trap", "s_endpgm"};

#define OMNIPROBE_PREFIX "__amd_crk_"



size_t readFile(const std::string& filename, std::vector<std::string>& lines) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        // Missing source files should not be fatal (e.g., repro binaries built on another machine)
        return 0;
    }

    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    file.close();
    return lines.size();
}


std::string genColumnMarkers(std::vector<uint32_t>& cols)
{
    // Sort the vector in ascending order
    std::sort(cols.begin(), cols.end());

    // Get the maximum value (last element after sorting, if vector is not empty)
    uint32_t max_val = cols.empty() ? 0 : cols.back();

    // Generate a string of spaces with length equal to max_val
    std::string result(max_val, ' ');

    // Replace space with '^' at each index from the vector
    for (uint32_t index : cols) {
        if (index - 1 < max_val) {
            result[index - 1] = '^';
        }
    }

    return result;
}


std::string demangleName(const char *name)
{
   int status;
   std::string result;
   char *realname = abi::__cxa_demangle(name, 0, 0, &status);
   if (status == 0)
   {
       if (realname)
       {
           result = realname;
           free(realname);
       }
   }
   else
   {
        if (realname)
            free(realname);
        // We're going through these gyrations here because the OMNIPROBE_PREFIX is being
        // prepended by the LLVM plugin AFTER kernel name mangling has already occurred.
        // So we have to do some special kind of non-standard de-mangling by stripping the
        // OMNIPROBE_PREFIX from the name before demangling, then adding it back in the
        // appropriate place.
        if (!strncmp(name, OMNIPROBE_PREFIX, strlen(OMNIPROBE_PREFIX)))
        {
            realname = abi::__cxa_demangle(&name[strlen(OMNIPROBE_PREFIX)],0,0, &status);
            if (status == 0 && realname)
            {
                result = realname;
                size_t pos = result.find_first_of(" ");
                size_t ret_type = result.find_first_of("(");
                // If pos > ret_type this means that there's no return type in the kernel name
                if (pos > ret_type)
                    pos = -1;
                result.insert(pos+1, OMNIPROBE_PREFIX);
                free(realname);
            }
        }
   }
   return result.length() ? result : std::string(name);
}

std::string getKernelName(const std::string& name)
{
    std::string result = name;
    size_t pos = result.find_last_of(')');
    if (pos != std::string::npos)
        result.erase(pos + 1);
    else
    {
        pos = result.find_last_of('.');
        if (pos != std::string::npos)
            result.erase(pos);
    }
    return result;
}

std::string getExecutablePath() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    return std::string(result, (count > 0) ? count : 0);
}

/* A helper function to create a list of all the shared libraries in use by the current
 * process. This is needed for HIP-style applications where, rather than utilizing
 * a code object cache of .hsaco files (e.g. the way Triton works), the application
 * is s HIP application where the instrumented clones are bound to the executable in
 * a fat binary */
void getSharedLibraries(std::vector<std::string>& libraries) {
    dl_iterate_phdr([](struct dl_phdr_info *info, size_t size, void *data){
        std::vector<std::string>* p_libraries = static_cast<std::vector<std::string>*>(data);

        if (info->dlpi_name && *info->dlpi_name) {  // Filter out empty names
            p_libraries->push_back(std::string(info->dlpi_name));
        }

        return 0;  // Continue iteration
    }, &libraries);
    return;
}

std::vector<std::string> getIsaList(hsa_agent_t agent)
{
    std::vector<std::string> list;
    hsa_agent_iterate_isas(agent,[](hsa_isa_t isa, void *data){
        std::vector<std::string> *pList = reinterpret_cast<std::vector<std::string> *> (data);
           uint32_t length;
           hsa_status_t status = hsa_isa_get_info(isa, HSA_ISA_INFO_NAME_LENGTH, 0, &length);
           if (status == HSA_STATUS_SUCCESS)
           {
                char *pName = static_cast<char *>(malloc(length + 1));
                if (pName)
                {
                    pName[length] = '\0';
                    status = hsa_isa_get_info(isa, HSA_ISA_INFO_NAME, 0, pName);
                    //std::cerr << "Isa name: " << pName << std::endl;
                    if (status == HSA_STATUS_SUCCESS)
                        pList->push_back(std::string(pName));
                    free(pName);
                }
                else
                {
                    std::cout << "The system is somehow out of memory at line " << __LINE__ << " so I'm aborting this run." << std::endl;
                    abort();
                }
           }
           return HSA_STATUS_SUCCESS;
        }, reinterpret_cast<void *>(&list));
    return list;
}

bool kernelDB::isBranch(const std::string& instruction)
{
    return branch_instructions.find(instruction) != branch_instructions.end();
}


kernelDB::kernelDB(hsa_agent_t agent, const std::string& fileName)
{
    agent_ = agent;
    std::string empty("");
    if (fileName.length() == 0)
    {
        fileName_ = getExecutablePath();
        addFile(fileName_, agent, empty);
        std::vector<std::string> shared_libs;
        getSharedLibraries(shared_libs);
        for (auto& lib : shared_libs)
            addFile(lib, agent, empty);
    }
    else
    {
        fileName_ = fileName;
        addFile(fileName_, agent, empty);
    }
}

kernelDB::kernelDB(hsa_agent_t agent) : agent_{agent} {}

kernelDB::~kernelDB()
{
   std::cout << "Ending kernelDB\n";
   std::cout << "Found " << kernels_.size() << " kernels.\n";
   auto it = kernels_.begin();
   while(it != kernels_.end())
   {
       it++;
   }
   for (auto it : file_map_)
   {
       // it.second is now a vector of file paths
       for (const auto& file_path : it.second)
       {
           // Only unlink temp files (where original != extracted)
           if (it.first != file_path)
               unlink(file_path.c_str());
       }
   }
}

CDNAKernel& kernelDB::getKernel(const std::string& name)
{
    ensureKernelLoaded(name);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = kernels_.find(getKernelName(name));
    if (it != kernels_.end())
        return *(it->second.get());
    throw std::runtime_error(name + " kernel does not exist.");
}


bool kernelDB::getBasicBlocks(const std::string& kernel, std::vector<basicBlock>&)
{
    return true;
}

bool kernelDB::addKernel(std::unique_ptr<CDNAKernel> kernel)
{
    bool result = true;
    std::unique_lock<std::shared_mutex> lock(mutex_);
    std::string strName = kernel.get()->getName();
    std::string canonical = getKernelName(strName);  // same key form as lazy_kernels_ and getKernel lookup
    if (kernels_.find(canonical) == kernels_.end())
    {
        kernels_[canonical] = std::move(kernel);
    }
    else
    {
        kernels_[canonical] = std::move(kernel);
        result = false;
    }
    return result;
}

void kernelDB::ensureKernelLoaded(const std::string& name)
{
    std::string canonical = getKernelName(name);
    std::string hsaco_path, elf_symbol;
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = lazy_kernels_.find(canonical);
        if (it == lazy_kernels_.end())
        {
            return;
        }
        hsaco_path = it->second.hsaco_path;
        elf_symbol = it->second.elf_symbol;
    }
    if (!scanCodeObjectForKernel(hsaco_path, elf_symbol))
    {
        return;
    }
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        lazy_kernels_.erase(canonical);
    }
}

bool kernelDB::addFile(const std::string& name, hsa_agent_t agent, const std::string& strFilter, bool lazy)
{
    bool bReturn = true;
    amd_comgr_data_t executable;
    bool bValidExecutable = false;
    std::vector<std::string> isas = ::kernelDB::getIsaList(agent);
    std::cout << "Adding " << name << (lazy ? " (lazy)" : "") << std::endl;

    if (name.ends_with(".hsaco"))
    {
        {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            file_map_[name] = {name};
        }
        bValidExecutable = true;
    }
    else
    {
        std::vector<std::string> tmp_hsacos = extractCodeObjects(agent, name);
        if (!tmp_hsacos.empty())
        {
            {
                std::unique_lock<std::shared_mutex> lock(mutex_);
                file_map_[name] = tmp_hsacos;
            }
            bValidExecutable = true;
        }
    }

    if (!bValidExecutable || !isas.size())
    {
        bReturn = false;
        return bReturn;
    }

    if (lazy)
    {
        // Only index: get kernel names from ELF symbol table; no disassembly yet.
        for (const auto& hsaco : file_map_[name])
        {
            std::vector<std::string> rawNames = getKernelNamesFromElf(hsaco);
            std::unique_lock<std::shared_mutex> lock(mutex_);
            for (const auto& raw : rawNames)
            {
                std::string demangled = demangleName(raw.c_str());
                std::string canonical = getKernelName(demangled);
                if (canonical.empty())
                    continue;
                lazy_kernels_[canonical] = {hsaco, name, raw};
            }
        }
        return true;
    }

    // Eager: disassemble and parse each code object, then map to source
    for (const auto& hsaco : file_map_[name])
    {
        std::string strDisassembly;
        getDisassembly(agent, hsaco, strDisassembly);
        parseDisassembly(strDisassembly);
    }
    try
    {
        mapDisassemblyToSource(agent, name.c_str());
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << "Error adding " << name << "\n\t" << e.what() << std::endl;
        bReturn = false;
    }
    return bReturn;
}

bool kernelDB::scanCodeObject(const std::string& co_file)
{
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        if (scanned_code_objects_.count(co_file))
            return true;
    }

    std::string strDisassembly;
    if (!getDisassembly(agent_, co_file, strDisassembly))
        return false;

    parseDisassembly(strDisassembly);

    std::map<Dwarf_Addr, SourceLocation> addrMap;
    buildDwarfAddressMap(co_file.c_str(), 0, 0, addrMap);

    if (!addrMap.empty())
        processKernelsWithAddressMap(addrMap);

    std::map<std::string, std::vector<KernelArgument>> kernelArgsMap;
    extractKernelArguments(co_file.c_str(), 0, 0, kernelArgsMap, false);

    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        for (const auto& entry : kernelArgsMap)
        {
            std::string demangledName = demangleName(entry.first.c_str());
            std::string searchName = getKernelName(demangledName);

            auto it = kernels_.find(searchName);
            if (it == kernels_.end())
            {
                for (auto& k : kernels_)
                {
                    if (k.first.compare(0, searchName.length(), searchName) == 0 &&
                        (k.first.length() == searchName.length() || k.first[searchName.length()] == '('))
                    {
                        k.second.get()->setArguments(entry.second);
                        break;
                    }
                }
            }
            else
            {
                it->second.get()->setArguments(entry.second);
            }
        }
        scanned_code_objects_.insert(co_file);
    }

    return true;
}

bool kernelDB::scanCodeObjectForKernel(const std::string& co_file, const std::string& kernelName)
{
    std::string strDisassembly;
    // Use targeted disassembly (--disassemble-symbols) to extract only the
    // requested kernel instead of disassembling the entire code object.
    // Try the FUNC symbol name first, then with .kd suffix (in case
    // the caller passed a kernel descriptor name).
    bool gotDisasm = false;
    std::string funcName = kernelName;
    if (funcName.size() > 3 && funcName.substr(funcName.size() - 3) == ".kd")
        funcName = funcName.substr(0, funcName.size() - 3);
    for (const auto& sym : {funcName, funcName + ".kd"})
    {
        try {
            gotDisasm = getDisassemblyForSymbol(agent_, co_file, sym, strDisassembly);
        } catch (...) {
            gotDisasm = false;
        }
        if (gotDisasm)
            break;
        strDisassembly.clear();
    }
    if (!gotDisasm)
    {
        return false;
    }

    parseDisassemblyForKernel(strDisassembly, kernelName);

    std::map<Dwarf_Addr, SourceLocation> addrMap;
    buildDwarfAddressMap(co_file.c_str(), 0, 0, addrMap);

    if (!addrMap.empty())
    {
        processKernelsWithAddressMap(addrMap, kernelName);
    }

    // Extract arguments only for the target kernel
    std::map<std::string, std::vector<KernelArgument>> kernelArgsMap;
    extractKernelArguments(co_file.c_str(), 0, 0, kernelArgsMap, false);

    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        for (const auto& entry : kernelArgsMap)
        {
            std::string demangledName = demangleName(entry.first.c_str());
            std::string searchName = getKernelName(demangledName);

            auto it = kernels_.find(searchName);
            if (it == kernels_.end())
            {
                for (auto& k : kernels_)
                {
                    if (k.first.compare(0, searchName.length(), searchName) == 0 &&
                        (k.first.length() == searchName.length() || k.first[searchName.length()] == '('))
                    {
                        k.second.get()->setArguments(entry.second);
                        break;
                    }
                }
            }
            else
            {
                it->second.get()->setArguments(entry.second);
            }
        }
        // Do NOT mark co_file in scanned_code_objects_ — other kernels from this
        // code object may still need loading later.
    }

    return true;
}

bool kernelDB::hasKernel(const std::string& name)
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::string key = getKernelName(name);
    return kernels_.find(key) != kernels_.end() || lazy_kernels_.find(key) != lazy_kernels_.end();
}

std::string kernelDB::extractKernelName(const std::string& line)
{
    std::string name;
    if (line.ends_with(":"))
    {
        std::vector<std::string> tokens;
        if (line.find_first_of(" ") != std::string::npos)
        {
            split(line, tokens, " ", false);
            if (tokens.size() == 2)
            {
                name = tokens[1].substr(0, tokens[1].length() - 1);
                if (name.starts_with("<") && name.ends_with(">"))
                {
                    name = name.substr(1, name.length() - 2);
                }
                if (name == ".text")
                    name = "";
            }
        }
        else
            name = line.substr(0, line.length() - 1);
    }
    return name;
}

parse_mode kernelDB::getLineType(std::string& line)
{
    parse_mode result = BBLOCK;
    auto it = line.begin();
    if (*it == ':' || line.starts_with(".text:"))
        result = BEGIN;
    else
    {
        it = --(line.end());
        if (*it == ':')
        {
            // It's only a valid kernel if there are no spaces in lines that end with ':'
            if(line.find_first_of(" ") == std::string::npos || *(--it) == '>')
            {
                if (line.ends_with("<.text>:"))
                    result = BEGIN;
                else
                    result = KERNEL;
            }
            else
                result = BEGIN;
        }
    }
    return result;
}

void kernelDB::getBlockMarkers(const std::string& disassembly, std::map<std::string, std::set<uint64_t>>& markers)
{
    std::istringstream in(disassembly);
    std::string line;
    uint64_t base_addr;
    std::getline(in, line);
    while (getLineType(line) != KERNEL)
        std::getline(in, line);
    std::string name;
    std::map<std::string, std::set<uint64_t>>::iterator it;
    do
    {
        parse_mode mode = getLineType(line);
        if (mode == KERNEL)
        {
            name = extractKernelName(line);
            name = demangleName(name.c_str());
            it = markers.find(name);
            if (it == markers.end())
            {
                markers[name] = std::set<uint64_t>();
                it = markers.find(name);
            }
            base_addr = 0;
        }
        else
        {
           std::vector<std::string> tokens;
           split(line, tokens, " ", false);
           if (tokens.size() && tokens[0].find("_cbranch_") != std::string::npos)
           {
               std::string addr = tokens[tokens.size() - 1];
               std::vector<std::string> tmp;
               split(addr, tmp, "+", false);
               assert(tmp.size() == 2);
               tmp[1].pop_back();
               it->second.insert(base_addr + std::stoull(tmp[1], nullptr, 16));
           }
           else if (base_addr == 0)
           {
                size_t i = 1;
                while (tokens[i].find("//") == std::string::npos)
                    i++;
                std::string strAddress = tokens[++i];
                // remove the ending colon
                strAddress.pop_back();
                base_addr = std::stoull(strAddress, nullptr, 16);
           }
        }
    }while(std::getline(in,line));
    // std::cout << "Found " << markers.size() << " kernels\n";
}

bool kernelDB::parseDisassembly(const std::string& text)
{
    bool bReturn = true;
    std::istringstream in(text);
    std::string line;
    parse_mode mode = BEGIN;
    std::string strKernel;
    uint32_t block_count = 0;
    std::unique_ptr<CDNAKernel> kernel;
    CDNAKernel *current_kernel = nullptr;
    std::unique_ptr<basicBlock> block;
    basicBlock *current_block = nullptr;
    std::map<std::string, std::set<uint64_t>> markers;
    getBlockMarkers(text, markers);
    //std::cout << "Found " << markers.size() << " in getBlockMarkers";
    std::map<std::string, std::set<uint64_t>>::iterator mit;
    bool bDoingKernels = false;
    while(std::getline(in,line))
    {
        bool blockCreated = false;
        std::vector<std::string> tokens;
        mode = getLineType(line);
        switch(mode)
        {
            case BEGIN:
                mode = KERNEL;
                block_count = 0;
                break;
            case KERNEL:
                bDoingKernels = true;
                strKernel = extractKernelName(line);
                //strKernel = line.substr(0, line.length() - 1);
                kernel = std::make_unique<CDNAKernel>(demangleName(strKernel.c_str()));
                mit = markers.find(demangleName(strKernel.c_str()));
                assert(mit != markers.end());
                current_kernel = kernel.get();
                mode=BBLOCK;
                addKernel(std::move(kernel));

                break;
            case BBLOCK:
                if (!bDoingKernels)
                    break;
                split(line, tokens, " ", false);
                if (tokens.size())
                {
                    if (!current_block)
                    {
                        //std::cout << "Starting a new block:\n\t" << line << std::endl;
                        block = std::make_unique<basicBlock>();
                        block_count++;
                        current_block = block.get();
                        blockCreated = true;
                    }
                    trim(tokens[0]);
                    if (isBranch(tokens[0]))
                    {
                        if (current_kernel)
                            current_kernel->addBlock(block_count, std::move(block));
                        else
                        {
                            std::cout << "Disassembly parsing error. Processing a branch instruction when there's not a kernel currently defined.\n";
                            std::cout << line << std::endl;
                            abort();
                        }
                        if (tokens[0].find("s_endpgm") != std::string::npos)
                        {
                            blockCreated = true;
                          //  std::cout << "New Block at endpgm\n\t" << line << std::endl;
                            block = std::make_unique<basicBlock>();
                            block_count++;
                            current_block = block.get();
                            // Going to let this drop down to add the s_endpgrm instruction to the block.
                            // This is the only branch instruction we include
                            // This should always be the last block and last instruction in the kernel
                        }
                        else
                        {
                            current_block = nullptr;
                            continue;
                        }
                    }
                    std::vector<std::string> inst_tokens;
                    instruction_t inst;

                    // If there is more than one token and the first token contains underscores that means it's an instruction line
                    if (tokens.size() > 1 && tokens[0].find("_") != std::string::npos)
                    {
                        split(tokens[0], inst_tokens, "_", false);
                        if (inst_tokens.size() > 2 && (inst_tokens[1] == "load" || inst_tokens[1] == "store"))
                        {
                            inst.prefix_ = inst_tokens[0];
                            inst.type_ = inst_tokens[1];
                            inst.size_ = inst_tokens[2];
                        }
                        inst.inst_ = tokens[0];
                        inst.disassembly_ = line;
                        for (size_t i = 1; i < tokens.size() && tokens[i].find("//") == std::string::npos; i++)
                            inst.operands_.push_back(tokens[i]);
                        size_t i = 1;
                        while (tokens[i].find("//") == std::string::npos)
                            i++;
                        std::string strAddress = tokens[++i];
                        // remove the ending colon
                        strAddress.pop_back();
                        inst.address_ = std::stoull(strAddress, nullptr, 16);
                        if (!blockCreated && mit != markers.end() && (mit->second.find(inst.address_) != mit->second.end()))
                        {
                            // This is the first line of a new block
                            // So add the current block and create a new one
                            // Not all blocks end in branches, so when we come to
                            // an address identified as the target of a conditional branch
                            // we don't care if the current block ends with a branch instruction
                            // we just save it and create a new one.
                            //std::cout << "Starting block at " << strAddress << std::endl;
                            if (current_block)
                                current_kernel->addBlock(block_count, std::move(block));
                            block = std::make_unique<basicBlock>();
                            block_count++;
                            current_block = block.get();
                        }
                        inst.block_ = current_block;
                        current_block->addInstruction(inst);
                        if (inst.inst_ == "s_endpgm")
                        {
                            if (current_kernel && current_block)
                            {
                                current_kernel->addBlock(block_count, std::move(block));
                                current_block = nullptr;
                            }
                            else
                                std::cerr << "Error parsing disassembly - s_endpgm without current kernel or block\n";
                        }
                    }
                }
                break;
            case BRANCH:
                current_block = nullptr;
                break;
            default:
                break;
        }
    }

    std::cout << "Marker Count: " << markers.size() << " Kernel Count: " << kernels_.size() << std::endl;

    return bReturn;
}

bool kernelDB::parseDisassemblyForKernel(const std::string& text, const std::string& targetKernel)
{
    bool bReturn = true;
    std::istringstream in(text);
    std::string line;
    parse_mode mode = BEGIN;
    std::string strKernel;
    uint32_t block_count = 0;
    std::unique_ptr<CDNAKernel> kernel;
    CDNAKernel *current_kernel = nullptr;
    std::unique_ptr<basicBlock> block;
    basicBlock *current_block = nullptr;
    std::map<std::string, std::set<uint64_t>> markers;
    getBlockMarkers(text, markers);
    std::map<std::string, std::set<uint64_t>>::iterator mit;
    bool bDoingKernels = false;
    bool skip = false;  // true when current kernel is not the target
    while(std::getline(in,line))
    {
        bool blockCreated = false;
        std::vector<std::string> tokens;
        mode = getLineType(line);
        switch(mode)
        {
            case BEGIN:
                mode = KERNEL;
                block_count = 0;
                break;
            case KERNEL:
            {
                bDoingKernels = true;
                strKernel = extractKernelName(line);
                std::string demangledName = demangleName(strKernel.c_str());
                std::string canonical = getKernelName(demangledName);
                if (canonical != targetKernel)
                {
                    skip = true;
                    current_kernel = nullptr;
                    current_block = nullptr;
                    break;
                }
                skip = false;
                kernel = std::make_unique<CDNAKernel>(demangledName);
                mit = markers.find(demangledName);
                assert(mit != markers.end());
                current_kernel = kernel.get();
                mode = BBLOCK;
                addKernel(std::move(kernel));
                break;
            }
            case BBLOCK:
                if (!bDoingKernels || skip)
                {
                    break;
                }
                split(line, tokens, " ", false);
                if (tokens.size())
                {
                    if (!current_block)
                    {
                        block = std::make_unique<basicBlock>();
                        block_count++;
                        current_block = block.get();
                        blockCreated = true;
                    }
                    trim(tokens[0]);
                    if (isBranch(tokens[0]))
                    {
                        if (current_kernel)
                        {
                            current_kernel->addBlock(block_count, std::move(block));
                        }
                        else
                        {
                            std::cout << "Disassembly parsing error. Processing a branch instruction when there's not a kernel currently defined.\n";
                            std::cout << line << std::endl;
                            abort();
                        }
                        if (tokens[0].find("s_endpgm") != std::string::npos)
                        {
                            blockCreated = true;
                            block = std::make_unique<basicBlock>();
                            block_count++;
                            current_block = block.get();
                        }
                        else
                        {
                            current_block = nullptr;
                            continue;
                        }
                    }
                    std::vector<std::string> inst_tokens;
                    instruction_t inst;

                    if (tokens.size() > 1 && tokens[0].find("_") != std::string::npos)
                    {
                        split(tokens[0], inst_tokens, "_", false);
                        if (inst_tokens.size() > 2 && (inst_tokens[1] == "load" || inst_tokens[1] == "store"))
                        {
                            inst.prefix_ = inst_tokens[0];
                            inst.type_ = inst_tokens[1];
                            inst.size_ = inst_tokens[2];
                        }
                        inst.inst_ = tokens[0];
                        inst.disassembly_ = line;
                        for (size_t i = 1; i < tokens.size() && tokens[i].find("//") == std::string::npos; i++)
                        {
                            inst.operands_.push_back(tokens[i]);
                        }
                        size_t i = 1;
                        while (tokens[i].find("//") == std::string::npos)
                        {
                            i++;
                        }
                        std::string strAddress = tokens[++i];
                        strAddress.pop_back();
                        inst.address_ = std::stoull(strAddress, nullptr, 16);
                        if (!blockCreated && mit != markers.end() && (mit->second.find(inst.address_) != mit->second.end()))
                        {
                            if (current_block)
                            {
                                current_kernel->addBlock(block_count, std::move(block));
                            }
                            block = std::make_unique<basicBlock>();
                            block_count++;
                            current_block = block.get();
                        }
                        inst.block_ = current_block;
                        current_block->addInstruction(inst);
                        if (inst.inst_ == "s_endpgm")
                        {
                            if (current_kernel && current_block)
                            {
                                current_kernel->addBlock(block_count, std::move(block));
                                current_block = nullptr;
                            }
                            else
                            {
                                std::cerr << "Error parsing disassembly - s_endpgm without current kernel or block\n";
                            }
                        }
                    }
                }
                break;
            case BRANCH:
                current_block = nullptr;
                break;
            default:
                break;
        }
    }

    return bReturn;
}

void kernelDB::getElfSectionBits(const std::string &fileName, const std::string &sectionName, size_t& offset, std::vector<uint8_t>& sectionData ) {
    std::ifstream file(fileName, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file: " + fileName);
    }

    // Read ELF header
    Elf64_Ehdr elfHeader;
    file.read(reinterpret_cast<char*>(&elfHeader), sizeof(elfHeader));

    // Check if it's an ELF file
    if (memcmp(elfHeader.e_ident, ELFMAG, SELFMAG) != 0) {
        throw std::runtime_error("Not a valid ELF file");
    }

    // Seek to the section header table
    file.seekg(elfHeader.e_shoff, std::ios::beg);

    // Read all section headers
    std::vector<Elf64_Shdr> sectionHeaders(elfHeader.e_shnum);
    file.read(reinterpret_cast<char*>(sectionHeaders.data()), elfHeader.e_shnum * sizeof(Elf64_Shdr));

    // Seek to the section header string table
    const Elf64_Shdr &shstrtab = sectionHeaders[elfHeader.e_shstrndx];
    std::vector<char> shstrtabData(shstrtab.sh_size);
    file.seekg(shstrtab.sh_offset, std::ios::beg);
    file.read(shstrtabData.data(), shstrtab.sh_size);

    // Find the section by name
    for (const auto &section : sectionHeaders) {
        std::string currentSectionName(&shstrtabData[section.sh_name]);

        if (currentSectionName == sectionName) {
            offset = section.sh_offset;
            // Read the section data
            sectionData.resize(section.sh_size);
            file.seekg(section.sh_offset, std::ios::beg);
            file.read(reinterpret_cast<char*>(sectionData.data()), section.sh_size);
            return;  // Return the section bits
        }
    }

    throw std::runtime_error("Section not found: " + sectionName);
}

// AMDGPU kernel symbol type (STT_LOOS = 10); standard ELF STT_FUNC = 2
#ifndef STT_AMDGPU_HSA_KERNEL
#define STT_AMDGPU_HSA_KERNEL 10
#endif

std::vector<std::string> kernelDB::getKernelNamesFromElf(const std::string& fileName)
{
    std::vector<std::string> names;
    std::ifstream file(fileName, std::ios::binary);
    if (!file)
        return names;

    Elf64_Ehdr elfHeader;
    file.read(reinterpret_cast<char*>(&elfHeader), sizeof(elfHeader));
    if (file.gcount() != sizeof(elfHeader) || memcmp(elfHeader.e_ident, ELFMAG, SELFMAG) != 0)
        return names;

    file.seekg(elfHeader.e_shoff, std::ios::beg);
    std::vector<Elf64_Shdr> sectionHeaders(elfHeader.e_shnum);
    file.read(reinterpret_cast<char*>(sectionHeaders.data()), elfHeader.e_shnum * sizeof(Elf64_Shdr));
    if (file.gcount() != static_cast<std::streamsize>(elfHeader.e_shnum * sizeof(Elf64_Shdr)))
        return names;

    const Elf64_Shdr& shstrtab = sectionHeaders[elfHeader.e_shstrndx];
    std::vector<char> shstrtabData(shstrtab.sh_size);
    file.seekg(shstrtab.sh_offset, std::ios::beg);
    file.read(shstrtabData.data(), shstrtab.sh_size);

    Elf64_Word textShndx = SHN_UNDEF;
    size_t symtabOffset = 0, symtabSize = 0, strtabOffset = 0, strtabSize = 0;

    for (Elf64_Word i = 0; i < elfHeader.e_shnum; ++i)
    {
        std::string secName(&shstrtabData[sectionHeaders[i].sh_name]);
        if (secName == ".text")
            textShndx = i;
        else if (secName == ".symtab")
        {
            symtabOffset = sectionHeaders[i].sh_offset;
            symtabSize = sectionHeaders[i].sh_size;
        }
        else if (secName == ".strtab")
        {
            strtabOffset = sectionHeaders[i].sh_offset;
            strtabSize = sectionHeaders[i].sh_size;
        }
    }

    if (textShndx == SHN_UNDEF || symtabSize == 0 || strtabSize == 0)
        return names;

    std::vector<char> strtabData(strtabSize);
    file.seekg(strtabOffset, std::ios::beg);
    file.read(strtabData.data(), strtabSize);

    const size_t numSyms = symtabSize / sizeof(Elf64_Sym);
    file.seekg(symtabOffset, std::ios::beg);
    for (size_t s = 0; s < numSyms; ++s)
    {
        Elf64_Sym sym;
        file.read(reinterpret_cast<char*>(&sym), sizeof(sym));
        if (file.gcount() != static_cast<std::streamsize>(sizeof(sym)))
            break;
        if (sym.st_shndx == SHN_UNDEF)
            continue;
        uint8_t type = ELF64_ST_TYPE(sym.st_info);
        bool isCode = (type == STT_FUNC || type == STT_AMDGPU_HSA_KERNEL);
        bool isObject = (type == STT_OBJECT);
        if (!isCode && !isObject)
            continue;
        if (isCode && sym.st_shndx != textShndx)
            continue;
        if (sym.st_name >= strtabSize)
            continue;
        const char* nameStr = &strtabData[sym.st_name];
        if (!nameStr[0])
            continue;
        std::string symName(nameStr);
        // Only keep .kd kernel descriptors from OBJECT symbols
        if (isObject && (symName.size() < 3 || symName.substr(symName.size() - 3) != ".kd"))
            continue;
        names.push_back(symName);
    }
    return names;
}

//using namespace llvm;
//using namespace llvm::object;

// Pair of (bundle_offset, bundle_size) for a single __CLANG_OFFLOAD_BUNDLE__ in .hip_fatbin
using BundleOffsetSize = std::pair<size_t, size_t>;

std::vector<BundleOffsetSize> findCodeObjectOffsets(hsa_agent_t agent, std::vector<uint8_t>& bits)
{
    // std::cout << "Bundle size: " << bits.size() << " bytes" << std::endl;

    const char* CLANG_OFFLOAD_MAGIC = "__CLANG_OFFLOAD_BUNDLE__";
    const size_t MAGIC_SIZE = 24;
    const size_t ALIGNMENT = 4096; // 4096-byte alignment

    std::vector<BundleOffsetSize> bundle_offset_sizes;
    size_t search_offset = 0;

    // Search for Clang Offload Bundles using calculated positions
    while (search_offset < bits.size()) {
        // Check if there's a valid bundle at this position
        if (search_offset + MAGIC_SIZE > bits.size()) {
            break;
        }

        // Validate that the data starts with "__CLANG_OFFLOAD_BUNDLE__"
        if (memcmp(bits.data() + search_offset, CLANG_OFFLOAD_MAGIC, MAGIC_SIZE) != 0) {
            std::cout << "No bundle found at expected position 0x" << std::hex << search_offset << std::dec << ", stopping search" << std::endl;
            break;
        }

        if (search_offset + MAGIC_SIZE + 8 > bits.size()) {
            break;
        }

        uint64_t num_bundles = *reinterpret_cast<const uint64_t*>(bits.data() + search_offset + MAGIC_SIZE);
        // std::cout << "Number of sub-bundles: " << num_bundles << std::endl;

        size_t offset = search_offset + MAGIC_SIZE + 8;
        uint64_t last_bundle_end = 0; // Track the furthest end position relative to bundle start

        // Parse sub-bundles to find their details and the maximum end position
        for (uint64_t i = 0; i < num_bundles && offset + 24 <= bits.size(); i++) {
            uint64_t bundle_offset = *reinterpret_cast<const uint64_t*>(bits.data() + offset);
            uint64_t bundle_size = *reinterpret_cast<const uint64_t*>(bits.data() + offset + 8);
            uint64_t triple_size = *reinterpret_cast<const uint64_t*>(bits.data() + offset + 16);

            offset += 24;
            if (offset + triple_size > bits.size()) break;

            // Read triple string
            std::string triple(reinterpret_cast<const char*>(bits.data() + offset), triple_size);
            if (!triple.empty() && triple.back() == '\0') {
                triple.pop_back(); // Remove null terminator
            }
            offset += triple_size;

            uint64_t bundle_end = bundle_offset + bundle_size;

            // std::cout << "Sub-bundle " << i << ":" << std::endl;
            // std::cout << "  Triple: " << triple << std::endl;
            // std::cout << "  Offset: 0x" << std::hex << bundle_offset << std::dec << " (" << bundle_offset << ")" << std::endl;
            // std::cout << "  Size: " << bundle_size << " bytes" << std::endl;
            // std::cout << "  End: 0x" << std::hex << bundle_end << std::dec << std::endl;
            // std::cout << "  Status: " << (search_offset + bundle_end <= bits.size() ? "Valid" : "Invalid") << std::endl;
            // std::cout << std::endl;

            // Track the furthest sub-bundle end position (relative to bundle start)
            if (bundle_end > last_bundle_end) {
                last_bundle_end = bundle_end;
            }
        }

        // No valid code objects in this bundle; nothing more to search for
        if (last_bundle_end == 0) {
            break;
        }

        // Record this bundle's offset and size so COMGR receives only this bundle's bytes
        bundle_offset_sizes.push_back({search_offset, static_cast<size_t>(last_bundle_end)});

        // Calculate the absolute end of all code objects in this bundle
        uint64_t absolute_end = search_offset + last_bundle_end;

        // Try the common case of 4096-byte aligned bundles first
        uint64_t next_aligned = ((absolute_end + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
        if (next_aligned + MAGIC_SIZE <= bits.size() &&
            memcmp(bits.data() + next_aligned, CLANG_OFFLOAD_MAGIC, MAGIC_SIZE) == 0) {
            search_offset = next_aligned;
        } else {
            // Fall back to a sequential scan from the true end of the bundle,
            // bounded to one alignment unit to avoid scanning the entire section.
            search_offset = bits.size(); // default: not found
            size_t scan_limit = std::min(absolute_end + ALIGNMENT, bits.size());
            for (size_t scan = absolute_end; scan + MAGIC_SIZE <= scan_limit; ++scan) {
                if (memcmp(bits.data() + scan, CLANG_OFFLOAD_MAGIC, MAGIC_SIZE) == 0) {
                    search_offset = scan;
                    break;
                }
            }
        }
    }

    std::cout << "Total Clang Offload Bundles found: " << bundle_offset_sizes.size() << std::endl;
    return bundle_offset_sizes;
}


std::vector<amd_comgr_code_object_info_t> kernelDB::getCodeObjectInfo(hsa_agent_t agent, std::vector<uint8_t>& bits)
{
    std::vector<amd_comgr_code_object_info_t> results;
    auto bundle_offset_sizes = findCodeObjectOffsets(agent, bits);

    for(size_t co_idx = 0; co_idx != bundle_offset_sizes.size(); ++co_idx)
    {
        size_t bundle_offset = bundle_offset_sizes[co_idx].first;
        size_t bundle_size  = bundle_offset_sizes[co_idx].second;

        amd_comgr_data_t bundle;
        std::vector<std::string> isas = getIsaList(agent);
        CHECK_COMGR(amd_comgr_create_data(AMD_COMGR_DATA_KIND_FATBIN, &bundle));
        // Pass only this bundle's bytes so COMGR decodes this bundle only
        CHECK_COMGR(amd_comgr_set_data(bundle, bundle_size,
                                       reinterpret_cast<const char *>(bits.data() + bundle_offset)));
        if (isas.size())
        {
            std::vector<amd_comgr_code_object_info_t> ql;
            for (int i = 0; i < isas.size(); i++)
                ql.push_back({isas[i].c_str(), 0, 0});

            CHECK_COMGR(amd_comgr_lookup_code_object(bundle, static_cast<amd_comgr_code_object_info_t *>(ql.data()), ql.size()));

            for (auto co : ql)
            {
                co.offset += bundle_offset;
                /* Collect all ISA-compatible code objects */
                if (co.size != 0)
                {
                    results.push_back(co);
                    break;  // Only take the first compatible ISA for this bundle offset
                }
            }
        }
        CHECK_COMGR(amd_comgr_release_data(bundle));
    }

    return results;
}

void CDNAKernel::getSourceCode(std::vector<std::string>& outputLines)
{
}


void kernelDB::processKernelsWithAddressMap(const std::map<Dwarf_Addr, SourceLocation>& addrMap, const std::string& targetKernel)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = kernels_.begin();
    while(it != kernels_.end())
    {
        if (!targetKernel.empty() && it->first != targetKernel)
        {
            it++;
            continue;
        }
        const auto& blocks = it->second.get()->getBasicBlocks();
        for (const auto& block : blocks)
        {
            assert(block.get());
            auto& instructions = block.get()->getModifiableInstructions();
            for(auto& instruction : instructions)
            {
                try
                {
                    SourceLocation source = getSourceLocation(addrMap, instruction.address_);
                    instruction_t inst = instruction;
                    instruction.line_ = inst.line_ = source.lineNumber;
                    instruction.column_ = inst.column_ = source.columnNumber;
                    instruction.block_ = inst.block_ = block.get();
                    instruction.path_id_ = inst.path_id_ = it->second.get()->addFileName(source.fileName);
                    it->second.get()->addLine(source.lineNumber, inst);
                }
                catch(const std::runtime_error& e)
                {
                    instruction_t inst = instruction;
                    instruction.line_ = inst.line_ = MISSING_SOURCE_INFO;
                    instruction.column_ = inst.column_ = MISSING_SOURCE_INFO;
                    instruction.block_ = inst.block_ = block.get();
                    instruction.path_id_ = inst.path_id_ = MISSING_SOURCE_INFO;
                    it->second.get()->addLine(MISSING_SOURCE_INFO, inst);
                }
            }
        }
        it++;
    }
}

void kernelDB::mapDisassemblyToSource(hsa_agent_t agent, const char *elfFilePath) {
    std::string strFile(elfFilePath);
    /*
        the file_map_ maps binaries to the extracted code object (hsaco) contents
        that were written to /tmp.  We do this because we need to disassemble
        the code objects and we need libdwarf to read the debug information in code
        objects. So we extract a single copy of the code object and write it to tmp.
        file_map_ correlates the original executable or .so file name with the tmp file
        that contains the extracted hsaco for the ISA on the system where this is
        being run. This way we're not extracting code objects multiple times when
        we want to use them multiple times.  All of this is a workaround to solve
        the problem of comgr disassembly having an upper bound of how many kernels
        it can disassemble. Disassembly of code objects with large #'s of kernels will
        randomly not contain all of the kernels in a hsaco. llvm-objdump works more
        reliably but requires an encapsulated hsaco binary (i.e. it can't cope with
        fat binaries evidently.
    */
    std::cerr << "Mapping disassembly to source for " << strFile << std::endl;

    std::map<Dwarf_Addr, SourceLocation> combined_addrMap;

    // file_map_[strFile] contains extracted code object temp files (from addFile/extractCodeObjects)
    // or the original .hsaco file path. Iterate over all entries and build DWARF maps.
    for (const auto& hsaco_file : file_map_[strFile])
    {
        std::map<Dwarf_Addr, SourceLocation> addrMap;
        if (buildDwarfAddressMap(hsaco_file.c_str(), 0, 0, addrMap))
        {
            combined_addrMap.insert(addrMap.begin(), addrMap.end());
        }
    }

    if (combined_addrMap.empty())
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        if (!kernels_.empty())
        {
            std::cerr << "Warning: Unable to build address map for " << elfFilePath
                      << " (no debug information available). Kernels will lack source line mapping." << std::endl;
        }
        // No address map available - kernels will have MISSING_SOURCE_INFO
        return;
    }
    extractArgumentsFromDwarf(agent, elfFilePath, false);

    // Process all kernels once with the combined address map
    if (!combined_addrMap.empty())
    {
        processKernelsWithAddressMap(combined_addrMap);
    }
    else
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        if (!kernels_.empty())
        {
            std::cerr << "Warning: Unable to build address map for " << elfFilePath 
                      << " (no debug information available). Kernels will lack source line mapping." << std::endl;
        }
        // No address map available - kernels will have MISSING_SOURCE_INFO
    }
}


std::string kernelDB::getFileName(const std::string& kernel, size_t index)
{
    ensureKernelLoaded(kernel);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = kernels_.find(getKernelName(kernel));
    if (it != kernels_.end())
    {
        if (index != MISSING_SOURCE_INFO)
            return it->second.get()->getFileName(index);
        else
            return std::string("<unknown>");
    }
    else
        return "";
}

std::vector<instruction_t> kernelDB::getInstructionsForLine(const std::string& kernel_name, uint32_t line, const std::string& match)
{
    ensureKernelLoaded(kernel_name);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = kernels_.find(getKernelName(kernel_name));
    if (it != kernels_.end())
        return it->second.get()->getInstructionsForLine(line, match);
    else
        throw std::runtime_error("Unable to find kernel " + kernel_name);
}

const std::vector<instruction_t>& kernelDB::getInstructionsForLine(const std::string& kernel_name, uint32_t line)
{
    ensureKernelLoaded(kernel_name);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = kernels_.find(getKernelName(kernel_name));
    if (it != kernels_.end())
        return it->second.get()->getInstructionsForLine(line);
    else
        throw std::runtime_error("Unable to find kernel " + kernel_name);
}

void kernelDB::getKernels(std::vector<std::string>& out)
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    for (const auto& p : kernels_)
        out.push_back(p.first);
    for (const auto& p : lazy_kernels_)
        out.push_back(p.first);
}

void kernelDB::getKernelLines(const std::string& kernel, std::vector<uint32_t>& out)
{
    ensureKernelLoaded(kernel);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = kernels_.find(getKernelName(kernel));
    if (it != kernels_.end())
        it->second.get()->getLineNumbers(out);
}

std::vector<KernelArgument> kernelDB::getKernelArguments(const std::string& kernel_name, bool resolve_typedefs)
{
    std::string logical_file;
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto lit = lazy_kernels_.find(getKernelName(kernel_name));
        if (lit != lazy_kernels_.end())
            logical_file = lit->second.logical_file;
    }
    ensureKernelLoaded(kernel_name);
    if (resolve_typedefs && !logical_file.empty())
        extractArgumentsFromDwarf(agent_, logical_file.c_str(), true);
    else if (resolve_typedefs && !fileName_.empty())
        extractArgumentsFromDwarf(agent_, fileName_.c_str(), true);

    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = kernels_.find(getKernelName(kernel_name));
    if (it != kernels_.end())
    {
        return it->second.get()->getArguments();
    }
    else
        throw std::runtime_error("Unable to find kernel " + kernel_name);
}

void kernelDB::extractArgumentsFromDwarf(hsa_agent_t agent, const char *elfFilePath, bool resolve_typedefs)
{
    std::string strFile(elfFilePath);
    std::map<std::string, std::vector<KernelArgument>> kernelArgsMap;

    try
    {
        // Extract kernel arguments from DWARF info in the already-extracted code objects
        for (const auto& hsaco_file : file_map_[strFile])
        {
            extractKernelArguments(hsaco_file.c_str(), 0, 0, kernelArgsMap, resolve_typedefs);
        }

        // Store the arguments in the kernel objects
        std::unique_lock<std::shared_mutex> lock(mutex_);
        for (const auto& entry : kernelArgsMap)
        {
            std::string demangledName = demangleName(entry.first.c_str());
            std::string searchName = getKernelName(demangledName);

            // Try to match by full name first
            auto it = kernels_.find(searchName);

            // If not found, try to match by prefix (for cases where DWARF has simple name
            // but kernels_ has full signature)
            if (it == kernels_.end())
            {
                for (auto& k : kernels_)
                {
                    // Check if kernel name starts with the DWARF name
                    if (k.first.compare(0, searchName.length(), searchName) == 0 &&
                        (k.first.length() == searchName.length() || k.first[searchName.length()] == '('))
                    {
                        k.second.get()->setArguments(entry.second);
                        break;
                    }
                }
            }
            else
            {
                it->second.get()->setArguments(entry.second);
            }
        }
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << "Warning: Unable to extract kernel arguments from " << elfFilePath << ": " << e.what() << std::endl;
    }
}

basicBlock::basicBlock()
{
}


const std::vector<instruction_t>& basicBlock::getInstructions()
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return instructions_;
}

std::vector<instruction_t>& basicBlock::getModifiableInstructions()
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return instructions_;
}

void basicBlock::addInstruction(const instruction_t& instruction)
{
   std::unique_lock<std::shared_mutex> lock(mutex_);
   instructions_.push_back(instruction);
   if (counts_.find(instruction.inst_) != counts_.end())
        counts_[instruction.inst_]++;
   else
        counts_[instruction.inst_] = 1;
}

CDNAKernel::CDNAKernel(const std::string& name)
{
    name_ = name;
}

void CDNAKernel::getLineNumbers(std::vector<uint32_t>& out)
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = line_map_.begin();
    while(it != line_map_.end())
    {
        out.push_back(it->first);
        it++;
    }
}

void CDNAKernel::printBlock(std::ostream& out, basicBlock *block, const std::string& label)
{
   assert(block);
   std::map<std::string, std::map<uint32_t, std::vector<uint32_t>>> columnMarkers;
   auto& instructions = block->getInstructions();
   std::unique_lock<std::shared_mutex> lock(mutex_);
   for (auto& inst : instructions)
   {
       if (inst.path_id_ != MISSING_SOURCE_INFO)
       {
           std::string filename = getFileName(inst.path_id_);
           auto it = source_cache_.find(filename);
           if (it == source_cache_.end())
           {
               std::vector<std::string> contents;
               readFile(filename, contents);
               source_cache_[filename] = contents;
           }
           auto jt = columnMarkers.find(filename);
           if (jt == columnMarkers.end())
           {
                columnMarkers[filename][inst.line_] = {inst.column_};
           }
           else
               jt->second[inst.line_].push_back(inst.column_);
       }
   }

   std::set<uint32_t> processed;
   for (auto inst : instructions)
   {
       if (inst.line_ != MISSING_SOURCE_INFO)
       {
           if (processed.find(inst.line_) == processed.end())
           {
               std::string filename = getFileName(inst.path_id_);
               auto it = source_cache_.find(filename);
               if (it != source_cache_.end() && it->second.size() > inst.line_) {
                   if (inst.line_)
                       out << it->second[inst.line_ - 1] << std::endl;
                   else
                       out << "No source line reference for this instruction: " << inst.disassembly_ << std::endl;
                   out << genColumnMarkers(columnMarkers[filename][inst.line_]) << std::endl;
               }
               processed.insert(inst.line_);
           }
       }
   }

}

size_t CDNAKernel::addBlock(uint32_t global_index, std::unique_ptr<basicBlock> block)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    assert(block.get());
    block_map_[global_index] = block.get();
    blocks_.push_back(std::move(block));
    return blocks_.size();
}

size_t CDNAKernel::getBlockCount()
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (block_map_.size() != blocks_.size())
        std::cout << "block_map_.size() == " << block_map_.size() << " while blocks_.size() == " << blocks_.size() << std::endl;
    return block_map_.size();
}

void CDNAKernel::addLine(uint32_t line, const instruction_t& instruction)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = line_map_.find(line);
    if (it != line_map_.end())
        it->second.push_back(instruction);
    else
        line_map_[line] = {instruction};
}

// 1-based index so that 0 indicates an error
size_t CDNAKernel::addFileName(const std::string& name)
{
    size_t result = 0;
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = file_map_.find(name);
    if (it == file_map_.end())
    {
        file_names_.push_back(name);
        result = file_names_.size();
        file_map_[name] = result;
    }
    else
        result = it->second;
    return result;
}

void CDNAKernel::addInstructionForLine(uint64_t line, const instruction_t& instruction)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = line_map_.find(line);
    if (it == line_map_.end())
        line_map_[line] = {instruction};
    else
        it->second.push_back(instruction);
}

std::vector<instruction_t> CDNAKernel::getInstructionsForLine(uint32_t line, const std::string& match)
{
   std::shared_lock<std::shared_mutex> lock(mutex_);
   auto it = line_map_.find(line);
   std::vector<instruction_t> result;
   if (it != line_map_.end())
   {
       if (match.length())
       {
           std::regex pattern(match);
           for(auto inst : it->second)
           {
               if (std::regex_match(inst.inst_, pattern))
                   result.push_back(inst);
           }
       }
       else
           result = it->second;
       return result;
   }
   else
       throw std::runtime_error("Unable to find instructions for line.");
}

const std::vector<instruction_t>& CDNAKernel::getInstructionsForLine(uint32_t line)
{
   std::shared_lock<std::shared_mutex> lock(mutex_);
   auto it = line_map_.find(line);
   if (it != line_map_.end())
   {
       return it->second;
   }
   else
       throw std::runtime_error("Unable to find instructions for line.");
}

void CDNAKernel::setArguments(const std::vector<KernelArgument>& args)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    arguments_ = args;
}

const std::vector<KernelArgument>& CDNAKernel::getArguments() const
{
    // No lock needed for const access
    return arguments_;
}

bool CDNAKernel::hasArguments() const
{
    return !arguments_.empty();
}

}//kernelDB
