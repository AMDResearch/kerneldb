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

kernelDB::~kernelDB()
{
   std::cout << "Ending kernelDB\n";
   std::cout << "Found " << kernels_.size() << " kernels.\n";
   auto it = kernels_.begin();
   while(it != kernels_.end())
   {
       it++;
   }
}
    
const CDNAKernel& kernelDB::getKernel(const std::string& name)
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = kernels_.find(getKernelName(name));
    if (it != kernels_.end())
    {
        return *(it->second.get());
    }
    else
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
    if (kernels_.find(strName) == kernels_.end())
    {
        kernels_[strName] = std::move(kernel);
    }
    else
    {
        std::cout << "You're adding kernel \"" << strName << "\" which we've seen before. Something may be wrong." << std::endl;
        kernels_[strName] = std::move(kernel);
        result = false;
    }
    return result;
}

bool kernelDB::addFile(const std::string& name, hsa_agent_t agent, const std::string& strFilter)
{
    bool bReturn = true;
    amd_comgr_data_t executable;
    std::vector<std::string> isas = ::kernelDB::getIsaList(agent);
    std::cout << "Adding " << name << std::endl;
    if (name.ends_with(".hsaco"))
    {
        std::vector<char> buff;
        std::ifstream file(name, std::ios::binary | std::ios::ate);

        if (!file.is_open()) {
            std::cerr << "Failed to open the file: " << name << std::endl;
        }

        std::streamsize fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        // Resize the buffer to fit the file content
        buff.resize(fileSize);

        // Read the file content into the buffer
        if (!file.read(buff.data(), fileSize)) {
            std::cerr << "Failed to read the file content" << std::endl;
        }
        file.close();
        CHECK_COMGR(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &executable));
        CHECK_COMGR(amd_comgr_set_data(executable, buff.size(), buff.data()));
    }
    else
    {
        amd_comgr_data_t bundle;
        std::vector<uint8_t> bits;
        try
        {
            getElfSectionBits(name, FATBIN_SECTION, bits);
        }
        catch (const std::runtime_error e)
        {
            //getElfSectionBits will throw a runtime error if it can't find the file. 
            return false;
        }
        CHECK_COMGR(amd_comgr_create_data(AMD_COMGR_DATA_KIND_FATBIN, &bundle));
        CHECK_COMGR(amd_comgr_set_data(bundle, bits.size(), reinterpret_cast<const char *>(bits.data())));
        if (isas.size())
        {
            std::vector<amd_comgr_code_object_info_t> ql;
            for (int i = 0; i < isas.size(); i++)
                ql.push_back({isas[i].c_str(),0,0});
            CHECK_COMGR(amd_comgr_lookup_code_object(bundle,static_cast<amd_comgr_code_object_info_t *>(ql.data()), ql.size()));
            for (auto co : ql)
            {
                /* Use the first code object that is ISA-compatible with this agent */
                if (co.size != 0)
                {
                    CHECK_COMGR(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &executable));
                    CHECK_COMGR(amd_comgr_set_data(executable, co.size, reinterpret_cast<const char *>(bits.data() + co.offset)));
                    break;
                }
            }   
        }
        CHECK_COMGR(amd_comgr_release_data(bundle));
    }
    if(isas.size())
    {
        amd_comgr_data_set_t dataSetIn, dataSetOut;
        amd_comgr_data_t dataOutput;
        amd_comgr_action_info_t dataAction;
        CHECK_COMGR(amd_comgr_create_data_set(&dataSetIn));
        CHECK_COMGR(amd_comgr_set_data_name(executable, "RB_DATAIN"));
        CHECK_COMGR(amd_comgr_data_set_add(dataSetIn, executable));
        CHECK_COMGR(amd_comgr_create_data_set(&dataSetOut));
        CHECK_COMGR(amd_comgr_create_action_info(&dataAction));
        CHECK_COMGR(amd_comgr_action_info_set_isa_name(dataAction,isas[0].c_str()));
    	CHECK_COMGR(amd_comgr_do_action(AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE,
                            dataAction, dataSetIn, dataSetOut));
		CHECK_COMGR(amd_comgr_destroy_data_set(dataSetIn));
		size_t count,size;
        
		CHECK_COMGR(amd_comgr_action_data_count(dataSetOut, AMD_COMGR_DATA_KIND_SOURCE, &count));
		CHECK_COMGR(amd_comgr_action_data_get_data(dataSetOut, AMD_COMGR_DATA_KIND_SOURCE, 0, &dataOutput));
		CHECK_COMGR(amd_comgr_get_data(dataOutput, &size, NULL));
		
        char *bytes = (char *)malloc(size+1);
        bytes[size] = '\0';
		CHECK_COMGR(amd_comgr_get_data(dataOutput, &size, bytes));
        std::string strDisassembly(bytes);
        free(bytes);
        //CHECK_COMGR(amd_comgr_destroy_data_set(dataSetIn));
        CHECK_COMGR(amd_comgr_release_data(dataOutput));
        CHECK_COMGR(amd_comgr_release_data(executable));
        //std::cout << strDisassembly << std::endl;
        parseDisassembly(strDisassembly);
        mapDisassemblyToSource(agent, name.c_str());
    }
    return bReturn;
}

parse_mode kernelDB::getLineType(std::string& line)
{
    parse_mode result = BBLOCK;
    auto it = line.begin();
    if (*it == ':')
        result = BEGIN;
    else
    {
        it = --(line.end());
        if (*it == ':')
        {
            // It's only a valid kernel if there are no spaces in lines that end with ':'
            if(line.find_first_of(" ") == std::string::npos)
            {
                result = KERNEL;
            }
        }
    }
    return result;
}

bool kernelDB::parseDisassembly(const std::string& text)
{
    bool bReturn = true;
    std::istringstream in(text);
    std::string line;
    parse_mode mode = BEGIN;
    std::string strKernel;
    uint16_t block_count = 0;
    std::unique_ptr<CDNAKernel> kernel;
    CDNAKernel *current_kernel = nullptr;
    std::unique_ptr<basicBlock> block;
    basicBlock *current_block = nullptr;
    while(std::getline(in,line))
    {
        std::vector<std::string> tokens;
        mode = getLineType(line);
        switch(mode)
        {
            case BEGIN:
                mode = KERNEL;
                if (block_count)
                    std::cout << std::dec << block_count << " blocks in " << strKernel << std::endl;
                block_count = 0;
                break;
            case KERNEL:
                strKernel = line.substr(0, line.length() - 1);
                kernel = std::make_unique<CDNAKernel>(demangleName(strKernel.c_str()));
                current_kernel = kernel.get();
                mode=BBLOCK;
                block_count++;
                addKernel(std::move(kernel));
                
                break;
            case BBLOCK:
                split(line, tokens, " ", false);
                if (tokens.size())
                {
                    if (!current_block)
                    {
                        block = std::make_unique<basicBlock>();
                        current_block = block.get();
                    }
                    trim(tokens[0]);
                    if (isBranch(tokens[0]))
                    {
                        block_count++;
                        if (current_kernel)
                            current_kernel->addBlock(std::move(block));
                        else
                        {
                            std::cout << "Disassembly parsing error. Processing a branch instruction when there's not a kernel currently defined.\n";
                            std::cout << line << std::endl;
                            abort();
                        }
                        current_block = nullptr;
                        continue;
                    }
                    std::vector<std::string> inst_tokens;
                    instruction_t inst;
                    split(tokens[0], inst_tokens, "_", false);
                    if (inst_tokens.size() > 2 && (inst_tokens[1] == "load" || inst_tokens[1] == "store"))
                    {
                        inst.prefix_ = inst_tokens[0];
                        inst.type_ = inst_tokens[1];
                        inst.size_ = inst_tokens[2];
                        inst.inst_ = tokens[0];
                        inst.disassembly_ = line;
                        for (size_t i = 3; i < inst_tokens.size(); i++)
                            inst.operands_.push_back(inst_tokens[i]);
                        size_t i = 1;
                        while (tokens[i] != "//")
                            i++;
                        std::string strAddress = tokens[++i];
                        // remove the ending colon
                        strAddress.pop_back();
                        inst.address_ = std::stoull(strAddress, nullptr, 16);
                        current_block->addInstruction(inst);
                    }
                }
                break;
            case BRANCH:
                block_count++;
                current_block = nullptr;
                break;
            default:
                break;
        }
    }
    return bReturn;
}

void kernelDB::getElfSectionBits(const std::string &fileName, const std::string &sectionName, std::vector<uint8_t>& sectionData ) {
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
            // Read the section data
            sectionData.resize(section.sh_size);
            file.seekg(section.sh_offset, std::ios::beg);
            file.read(reinterpret_cast<char*>(sectionData.data()), section.sh_size);
            return;  // Return the section bits
        }
    }

    throw std::runtime_error("Section not found: " + sectionName);
}

using namespace llvm;
using namespace llvm::object;

amd_comgr_code_object_info_t kernelDB::getCodeObjectInfo(hsa_agent_t agent, std::vector<uint8_t>& bits)
{
    amd_comgr_data_t executable, bundle;
    std::vector<std::string> isas = getIsaList(agent);
    CHECK_COMGR(amd_comgr_create_data(AMD_COMGR_DATA_KIND_FATBIN, &bundle));
    CHECK_COMGR(amd_comgr_set_data(bundle, bits.size(), reinterpret_cast<const char *>(bits.data())));
    if (isas.size())
    {
        std::vector<amd_comgr_code_object_info_t> ql;
        for (int i = 0; i < isas.size(); i++)
            ql.push_back({isas[i].c_str(),0,0});
        //for(auto co : ql)
        //    std::cerr << "{" << co.isa << "," << co.size << "," << co.offset << "}" << std::endl;
        //std::cerr << "query list size: " << ql.size() << std::endl;
        CHECK_COMGR(amd_comgr_lookup_code_object(bundle,static_cast<amd_comgr_code_object_info_t *>(ql.data()), ql.size()));
        for (auto co : ql)
        {
            //std::cerr << "After query: " << std::endl;
            //std::cerr << "{" << co.isa << "," << co.size << "," << co.offset << "}" << std::endl;
            /* Use the first code object that is ISA-compatible with this agent */
            if (co.size != 0)
            {
                CHECK_COMGR(amd_comgr_release_data(bundle));
                return co;
            }
        }   
    }
    CHECK_COMGR(amd_comgr_release_data(bundle));
    return {0,0,0};
}

void kernelDB::dumpDwarfInfo(const char *elfFilePath, void * val)
{
    llvm::MemoryBuffer *pVal = static_cast<llvm::MemoryBuffer *>(val);
    if (pVal)
    {
        auto ObjOrErr = ObjectFile::createObjectFile(pVal->getMemBufferRef());
        if (!ObjOrErr) {
            errs() << "Error parsing ELF file: " << elfFilePath << "\n";
            return;
        }

        auto *Obj = ObjOrErr->get();
        auto ELF = dyn_cast<ELFObjectFileBase>(Obj);
        if (!ELF) {
            errs() << "File is not an ELF file.\n";
            return;
        }

        // Create DWARF context
        auto DICtx = DWARFContext::create(*ELF);

        // Iterate over compilation units
        for (const auto &CU : DICtx->compile_units()) {
            if (!CU)
                continue;

            // Get the line table
            const auto &LineTable = DICtx->getLineTableForUnit(CU.get());

            if (!LineTable)
                continue;

            // Print source line mappings
            errs() << "Source line mappings for CU:\n";
            LineTable->dump(errs(), DIDumpOptions::getForSingleDIE());

            // Iterate over address mappings
            for (const auto &Row : LineTable->Rows) {
                errs() << "Address: " << format_hex(Row.Address.Address, 10)
                       << " -> File: " << Row.File << ", Line: " << Row.Line
                       << "\n";
            }
        }
    }
}

void kernelDB::buildLineMap(void *buff, const char *elfFilePath)
{
    MemoryBuffer *pVal = static_cast<MemoryBuffer *>(buff);
    if (pVal)
    {
        auto ObjOrErr = ObjectFile::createObjectFile(pVal->getMemBufferRef());
        if (!ObjOrErr) {
            errs() << "Error parsing ELF file: " << elfFilePath << "\n";
            return;
        }

        auto *Obj = ObjOrErr->get();
        auto ELF = dyn_cast<ELFObjectFileBase>(Obj);
        if (!ELF) {
            errs() << "File is not an ELF file.\n";
            return;
        }

        // Create DWARF context
        auto DICtx = DWARFContext::create(*ELF);

        // Iterate over compilation units
        for (const auto &CU : DICtx->compile_units()) {
            if (!CU)
                continue;

            // Get the line table
            const auto &LineTable = DICtx->getLineTableForUnit(CU.get());

            if (!LineTable)
                continue;

        }
        for (const auto &CU : DICtx->compile_units()) {
            if (!CU)
                continue;

            // Get the line table
            const auto &LineTable = DICtx->getLineTableForUnit(CU.get());

            if (!LineTable)
                continue;
            std::unique_lock<std::shared_mutex> lock(mutex_);
            auto it = kernels_.begin();
            while(it != kernels_.end())
            {
                
                const auto& blocks = it->second.get()->getBasicBlocks();
                //for (size_t i=0; i < blocks.size(); i++)
                for (const auto& block : blocks)
                {
                    const auto& instructions = block.get()->getInstructions();
                    for(auto& instruction : instructions)
                    {
                        DILineInfo info;
                        bool bSuccess;

                        #if LLVM_VERSION_MAJOR > 19
                        bSuccess = LineTable->getFileLineInfoForAddress({instruction.address_},false,"", 
                            DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath,info);
                        #else
                        bSuccess = LineTable->getFileLineInfoForAddress({instruction.address_},"", 
                            DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath,info);
                        #endif

                        if (bSuccess)
                        {
                            instruction_t inst = instruction;
                            inst.line_ = info.Line;
                            inst.column_ = info.Column;
                            inst.block_ = block.get();
                            //addFileName returns a 1-based index. 
                            inst.path_id_ = it->second.get()->addFileName(info.FileName) - 1;
                            it->second.get()->addLine(info.Line, inst);
                        }
                    }
                }
                it++;
            }

        }
    }
}

void kernelDB::mapDisassemblyToSource(hsa_agent_t agent, const char *elfFilePath) {
    std::string strFile(elfFilePath);
    std::unique_ptr<MemoryBuffer> pBuff;
    MemoryBuffer *pVal = NULL;
    if (!strFile.ends_with(".hsaco"))
    {
        std::vector<uint8_t> bits;
        getElfSectionBits(strFile, std::string(".hip_fatbin"), bits);
        amd_comgr_code_object_info_t info = getCodeObjectInfo(agent, bits);
        if (info.size)
        {
            llvm::StringRef ref(reinterpret_cast<char *>(bits.data() + info.offset), info.size);
            pBuff = MemoryBuffer::getMemBuffer(ref);
            //dumpDwarfInfo(elfFilePath, pBuff.get());
            buildLineMap(pBuff.get(), elfFilePath);
        }
    }
    else
    {
        // Open HSACO file
        auto FileOrErr = MemoryBuffer::getFile(elfFilePath);
        if (!FileOrErr) {
            errs() << "Error reading file: " << elfFilePath << "\n";
            return;
        }
        else
        {
            //dumpDwarfInfo(elfFilePath, FileOrErr->get());
            buildLineMap(FileOrErr->get(), elfFilePath);
        }
    }
}
    

std::string kernelDB::getFileName(const std::string& kernel, size_t index)
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = kernels_.find(getKernelName(kernel));
    if (it != kernels_.end())
    {
        return it->second.get()->getFileName(index);
    }
    else
        return "";
}
    
const std::vector<instruction_t>& kernelDB::getInstructionsForLine(const std::string& kernel_name, uint32_t line)
{
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
    auto it = kernels_.begin();
    while (it != kernels_.end())
    {
        out.push_back(it->first);
        it++;
    }
}

void kernelDB::getKernelLines(const std::string& kernel, std::vector<uint32_t>& out)
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = kernels_.find(getKernelName(kernel));
    if (it != kernels_.end())
    {
       it->second.get()->getLineNumbers(out); 
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

size_t CDNAKernel::addBlock(std::unique_ptr<basicBlock> block)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    blocks_.push_back(std::move(block));
    return blocks_.size();
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
    
const std::vector<instruction_t>& CDNAKernel::getInstructionsForLine(uint32_t line)
{
   std::shared_lock<std::shared_mutex> lock(mutex_);
   auto it = line_map_.find(line);
   if (it != line_map_.end())
   {
       std::cout << "Kernel: " << name_ << "\tInstruction Count for line " << line << " == " << it->second.size() << std::endl;
       return it->second;
   }
   else
       throw std::runtime_error("Unable to find instructions for line.");
}

}//kernelDB
