
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
#include <cstring>
#include "include/kernelDB.h"

static const char CCOB_MAGIC[] = "CCOB";
static const size_t CCOB_MAGIC_SIZE = 4;

static bool isCCOB(const uint8_t* data, size_t size)
{
    return size >= CCOB_MAGIC_SIZE && memcmp(data, CCOB_MAGIC, CCOB_MAGIC_SIZE) == 0;
}

static bool isCCOBFile(const std::string& fileName)
{
    std::ifstream file(fileName, std::ios::binary);
    if (!file.is_open())
        return false;
    char magic[CCOB_MAGIC_SIZE];
    file.read(magic, CCOB_MAGIC_SIZE);
    return file.good() && memcmp(magic, CCOB_MAGIC, CCOB_MAGIC_SIZE) == 0;
}

// Get the total size of a CCOB block from its header.
// V1: Magic(4) + Version(2) + Method(2) + UncompressedSize(4) + Hash(8) = 20 bytes header; no FileSize field
// V2: Magic(4) + Version(2) + Method(2) + FileSize(4) + UncompressedSize(4) + Hash(8) = 24 bytes header
// V3: Magic(4) + Version(2) + Method(2) + FileSize(8) + UncompressedSize(8) + Hash(8) = 32 bytes header
static size_t getCCOBBlockSize(const uint8_t* data, size_t available)
{
    if (available < 8)
        return 0;
    uint16_t version = *reinterpret_cast<const uint16_t*>(data + 4);
    if (version >= 3 && available >= 16)
        return *reinterpret_cast<const uint64_t*>(data + 8);
    if (version == 2 && available >= 12)
        return *reinterpret_cast<const uint32_t*>(data + 8);
    // V1 has no FileSize field — return 0 to signal unknown
    return 0;
}

// Find the path to clang-offload-bundler from ROCM_PATH
static std::string findOffloadBundler()
{
    std::string rocm_path = "/opt/rocm";
    const char* env = getenv("ROCM_PATH");
    if (env && env[0])
        rocm_path = env;
    std::string bundler = rocm_path + "/llvm/bin/clang-offload-bundler";
    if (std::filesystem::exists(bundler))
        return bundler;
    // Fallback: check if it's in PATH
    bundler = "clang-offload-bundler";
    return bundler;
}

// Build the --targets value from the ISA list.
// getIsaList() returns e.g. "amdgcn-amd-amdhsa--gfx90a"
// clang-offload-bundler expects "hipv4-amdgcn-amd-amdhsa--gfx90a"
static std::string buildTarget(hsa_agent_t agent)
{
    std::vector<std::string> isas = kernelDB::getIsaList(agent);
    if (isas.empty())
        return "";
    return "hipv4-" + isas[0];
}

// Write bytes from a buffer to a temporary file and return the path.
static std::string createTempFileFromBuffer(const uint8_t* data, size_t size)
{
    char temp_filename[L_tmpnam];
    if (tmpnam(temp_filename) == nullptr)
        throw std::runtime_error("Failed to generate temporary filename");

    std::ofstream temp_file(temp_filename, std::ios::binary);
    if (!temp_file.is_open())
        throw std::runtime_error("Failed to create temporary file: " + std::string(temp_filename));

    temp_file.write(reinterpret_cast<const char*>(data), size);
    if (temp_file.fail()) {
        temp_file.close();
        remove(temp_filename);
        throw std::runtime_error("Failed to write to temporary file: " + std::string(temp_filename));
    }
    temp_file.close();
    return std::string(temp_filename);
}

// Unbundle a CCOB file using clang-offload-bundler.
// Returns the path to the decompressed code object (.hsaco), or empty string on failure.
static std::string unbundleCCOB(const std::string& bundler, const std::string& input_file, const std::string& target)
{
    char output_filename[L_tmpnam];
    if (tmpnam(output_filename) == nullptr)
        return "";

    std::string command = bundler +
        " --unbundle --type=o"
        " --input=" + input_file +
        " --output=" + std::string(output_filename) +
        " --targets=" + target +
        " 2>/dev/null";

    int ret = system(command.c_str());
    if (ret != 0) {
        remove(output_filename);
        return "";
    }

    std::ifstream check(output_filename);
    if (!check.good()) {
        return "";
    }
    check.close();

    return std::string(output_filename);
}

// Extract code objects from a CCOB-compressed .hip_fatbin section.
// The section may contain multiple CCOB blocks, each 4096-byte aligned.
static std::vector<std::string> extractFromCCOBSection(const std::vector<uint8_t>& bits,
                                                        const std::string& bundler,
                                                        const std::string& target)
{
    std::vector<std::string> results;
    size_t pos = 0;

    while (pos < bits.size() && isCCOB(bits.data() + pos, bits.size() - pos))
    {
        size_t block_size = getCCOBBlockSize(bits.data() + pos, bits.size() - pos);
        if (block_size == 0 || pos + block_size > bits.size())
            break;

        // Write this CCOB block to a temp file
        std::string ccob_temp;
        try {
            ccob_temp = createTempFileFromBuffer(bits.data() + pos, block_size);
        } catch (const std::runtime_error&) {
            break;
        }

        // Unbundle it
        std::string hsaco = unbundleCCOB(bundler, ccob_temp, target);
        remove(ccob_temp.c_str());

        if (!hsaco.empty())
            results.push_back(hsaco);

        // Advance to the next 4096-aligned position after this block
        size_t next = pos + block_size;
        next = ((next + 4095) / 4096) * 4096;
        pos = next;
    }

    return results;
}


std::vector<std::string> extractCodeObjects(hsa_agent_t agent, const std::string& fileName)
{
    std::vector<std::string> results;
    if (!fileName.ends_with(".hsaco"))
    {
        // Check if the file itself is CCOB-compressed (e.g., standalone Tensile .co files)
        if (isCCOBFile(fileName))
        {
            std::string bundler = findOffloadBundler();
            std::string target = buildTarget(agent);
            if (target.empty())
                return results;

            std::string hsaco = unbundleCCOB(bundler, fileName, target);
            if (!hsaco.empty())
                results.push_back(hsaco);
            return results;
        }

        size_t section_offset = 0;
        std::vector<uint8_t> bits;
        try
        {
            kernelDB::kernelDB::getElfSectionBits(fileName, std::string(".hip_fatbin"), section_offset, bits);
        }
        catch (const std::runtime_error& e)
        {
            // Not a fat binary
            return results;
        }

        // Check if the .hip_fatbin section contains CCOB-compressed data
        if (!bits.empty() && isCCOB(bits.data(), bits.size()))
        {
            std::string bundler = findOffloadBundler();
            std::string target = buildTarget(agent);
            if (target.empty())
                return results;

            results = extractFromCCOBSection(bits, bundler, target);
            return results;
        }

        // Standard uncompressed path
        std::vector<amd_comgr_code_object_info_t> code_objects = kernelDB::kernelDB::getCodeObjectInfo(agent, bits);

        // Extract ALL code objects
        for (const auto& info : code_objects)
        {
            if (info.size)
            {
                std::string temp_file = create_temp_file_segment(fileName, section_offset + info.offset, info.size);
                results.push_back(temp_file);
            }
        }
    }
    else
    {
        results.push_back(fileName);
    }
    return results;
}
