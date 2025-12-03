
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


std::vector<std::string> extractCodeObjects(hsa_agent_t agent, const std::string& fileName)
{
    std::vector<std::string> results;
    if (!fileName.ends_with(".hsaco"))
    {
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
