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
#include <map>
#include <string>
#include <stdexcept>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include "include/kernelDB.h"
#include <cstdint>
#include <optional>
#include <gelf.h>
#include <libelf.h>
#include <cstring>

template <typename T>
std::optional<Dwarf_Addr> find_floor_key(const std::map<Dwarf_Addr, T>& map, Dwarf_Addr target) {
    if (map.empty()) {
        return std::nullopt; // No keys in the map
    }

    auto it = map.find(target);
    // If there's an exact match, return that
    if (it != map.end())
        return it->first;

    // If there's no exact match, use next greater addr
    it = map.upper_bound(target); // First key > target
    if (it == map.begin()) {
        return std::nullopt; // No key <= target (all keys are > target)
    }

    // Move to the key <= target (previous iterator)
    //if (it == map.end())
        --it;
    return it->first;
}

bool is_subprogram(Dwarf_Die die, Dwarf_Error *error) {
    Dwarf_Half tag;
    int result = dwarf_tag(die, &tag, error);
    if (result != DW_DLV_OK)
    {
        printf("Error getting subprogram tag\n");
        return 0; // Error occurred
    }
    if (tag == DW_TAG_subprogram)
        printf("Found a subprogram\n");
    else
        printf("Don't know what this tag is: %u\n", tag);
    return tag == DW_TAG_subprogram;
}

bool process_function_die(Dwarf_Debug dbg, Dwarf_Die die, uint32_t& decl_line) {
    Dwarf_Attribute *attrs;
    Dwarf_Signed attr_count;
    Dwarf_Error err;
    bool bResult = false;

    // Get all attributes of the DIE
    if (dwarf_attrlist(die, &attrs, &attr_count, &err) == DW_DLV_OK) {
        for (int i = 0; i < attr_count; i++) {
            Dwarf_Half attr;
            if (dwarf_whatattr(attrs[i], &attr, &err) == DW_DLV_OK) {
                if (attr == DW_AT_decl_line) {
                    Dwarf_Unsigned line_num;
                    if (dwarf_formudata(attrs[i], &line_num, &err) == DW_DLV_OK) {
                        decl_line = line_num;
                        printf("Function declared at line: %llu\n", line_num);
                        bResult = true;
                        break;
                    }
                }
                // Optionally, also check DW_AT_decl_file for the file name
            }
        }
        // Clean up attributes
        for (int i = 0; i < attr_count; i++) {
            dwarf_dealloc(dbg, attrs[i], DW_DLA_ATTR);
        }
        dwarf_dealloc(dbg, attrs, DW_DLA_LIST);
    }
    return bResult;
}


SourceLocation getSourceLocation(std::map<Dwarf_Addr, SourceLocation>& addrMap, Dwarf_Addr addr)
{
    auto key = find_floor_key<SourceLocation>(addrMap, addr);
    if (key.has_value())
        return addrMap[*key];
    else
        throw std::runtime_error("No matching key for this address.\n");
}

std::string create_temp_file_segment(const std::string& filename, std::streamoff offset, std::streamsize length) {
    // Open the source file in binary mode
    std::ifstream source_file(filename, std::ios::binary);
    if (!source_file.is_open()) {
        throw std::runtime_error("Failed to open source file: " + filename);
    }

    // Get the file size to validate offset and length
    source_file.seekg(0, std::ios::end);
    std::streamoff file_size = source_file.tellg();
    if (offset < 0 || offset >= file_size || length < 0 || offset + length > file_size) {
        source_file.close();
        throw std::runtime_error("Invalid offset or length for file: " + filename);
    }

    // Seek to the specified offset
    source_file.seekg(offset, std::ios::beg);

    // Create a temporary file using tmpnam (note: tmpnam is not the most secure option)
    char temp_filename[L_tmpnam];
    if (tmpnam(temp_filename) == nullptr) {
        source_file.close();
        throw std::runtime_error("Failed to generate temporary filename");
    }

    // Open the temporary file in binary mode
    std::ofstream temp_file(temp_filename, std::ios::binary);
    if (!temp_file.is_open()) {
        source_file.close();
        throw std::runtime_error("Failed to create temporary file: " + std::string(temp_filename));
    }

    // Copy the specified segment
    char buffer[4096]; // 4KB buffer for efficiency
    std::streamsize bytes_remaining = length;
    while (bytes_remaining > 0) {
        std::streamsize chunk_size = std::min(static_cast<std::streamsize>(sizeof(buffer)), bytes_remaining);
        source_file.read(buffer, chunk_size);
        if (source_file.fail() && !source_file.eof()) {
            temp_file.close();
            source_file.close();
            remove(temp_filename); // Clean up on error
            throw std::runtime_error("Error reading from source file");
        }
        temp_file.write(buffer, chunk_size);
        if (temp_file.fail()) {
            temp_file.close();
            source_file.close();
            remove(temp_filename);
            throw std::runtime_error("Error writing to temporary file");
        }
        bytes_remaining -= chunk_size;
    }

    // Close both files
    temp_file.close();
    source_file.close();

    return std::string(temp_filename);
}

void enumerate_subprograms(Dwarf_Debug dbg, Dwarf_Die cu_die) {
    Dwarf_Die child_die = NULL;
    Dwarf_Die sibling_die = NULL;
    Dwarf_Error err;
    int result;
    char **srcfiles;
    Dwarf_Signed file_count = 0;
    if(dwarf_srcfiles(cu_die, &srcfiles, &file_count, &err) != DW_DLV_OK) {
        printf("Source file information for declarations is not available.\n");
    }
    printf("File Count: %lld\n", file_count);

    // Get the first child of the compile unit DIE
    result = dwarf_child(cu_die, &child_die, &err);
    if (result == DW_DLV_OK){

        do {
            Dwarf_Half tag;
            if (dwarf_tag(child_die, &tag, &err) == DW_DLV_OK) {
                if (tag == DW_TAG_subprogram) {
                    // Found a subprogram DIE
                    char *name = NULL, *file_name = NULL;
                    Dwarf_Unsigned decl_line = 0;

                    // Extract DW_AT_name
                    Dwarf_Attribute name_attr;
                    if (dwarf_attr(child_die, DW_AT_name, &name_attr, &err) == DW_DLV_OK) {
                        dwarf_formstring(name_attr, &name, &err);
                        printf("Subprogram: %s\n", name);
                        dwarf_dealloc_attribute(name_attr);
                    }

                    Dwarf_Attribute file_attr;
                    Dwarf_Unsigned file_index = 0;
                    if (dwarf_attr(child_die, DW_AT_decl_file, &file_attr, &err) == DW_DLV_OK) {
                        dwarf_formudata(file_attr, &file_index, &err);
                        printf("File Index: %llu\n", file_index);
                        if (file_index > 0 && srcfiles)
                            printf("File name: %s\n", srcfiles[file_index - 1]);
                        dwarf_dealloc_attribute(file_attr);
                    }

                    // Extract DW_AT_decl_line
                    Dwarf_Attribute line_attr;
                    if (dwarf_attr(child_die, DW_AT_decl_line, &line_attr, &err) == DW_DLV_OK) {
                        dwarf_formudata(line_attr, &decl_line, &err);
                        printf("Decl Line Number: %llu\n", decl_line);
                        dwarf_dealloc_attribute(line_attr);
                    }

                    // Print subprogram details
                    printf("Subprogram: %s, Declared at line: %llu\n",
                           name ? name : "<no name>", decl_line);
                }
            }
            // Move to the next sibling
            sibling_die = NULL;
            if (dwarf_siblingof_b(dbg, child_die, true, &sibling_die, &err) != DW_DLV_OK) {
                break;
            }
            dwarf_dealloc(dbg, child_die, DW_DLA_DIE);
            child_die = sibling_die;
        } while (child_die != NULL);
        if (child_die) {
            dwarf_dealloc(dbg, child_die, DW_DLA_DIE);
        }
    }
    else
    {
        // Extract DW_AT_decl_line
        Dwarf_Attribute line_attr;
        Dwarf_Unsigned decl_line = 0;
        if (dwarf_attr(cu_die, DW_AT_decl_line, &line_attr, &err) == DW_DLV_OK) {
            dwarf_formudata(line_attr, &decl_line, &err);
            printf("CU Decl Line Number: %llu\n", decl_line);
            dwarf_dealloc_attribute(line_attr);
        }
    }
    if (srcfiles)
        dwarf_dealloc(dbg, srcfiles, DW_DLA_LIST);
}


// Function to build address-to-source-location map
bool buildDwarfAddressMap(const char* filename, size_t offset, size_t hsaco_length, std::map<Dwarf_Addr, SourceLocation>& addressMap) {

    std::string inFileName = filename;

    if (offset != 0)
    {
        inFileName = create_temp_file_segment(inFileName, offset, hsaco_length);
    }

    // Open the file
    int fd = open(inFileName.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("Failed to open file");
    }


    // Initialize DWARF
    Dwarf_Debug dbg;
    Dwarf_Error err;
    int status = 0;

    // Define a macro to handle the different signatures of dwarf_init_b
    #if defined(UBUNTU_LIBDWARF) // Define this macro for Ubuntu builds
    status = dwarf_init_b(fd, DW_DLC_READ, DW_GROUPNUMBER_ANY, NULL, NULL, &dbg, &err);
    #else // Default to RHEL9 or other versions
    status = dwarf_init_b(fd, DW_GROUPNUMBER_ANY, NULL, NULL, &dbg, &err);
    #endif
    if (status == DW_DLV_ERROR){
        close(fd);
        throw std::runtime_error(std::string("DWARF init failed: ") + dwarf_errmsg(err));
    }
    else if (status == DW_DLV_NO_ENTRY)
    {
        // No DWARF data in this file
        close(fd);
        return false;
    }

    // Iterate through all compilation units
    uint32_t decl_line = UINT32_MAX;;
    Dwarf_Unsigned cu_header_length;
    Dwarf_Half version_stamp;
    Dwarf_Unsigned abbrev_offset;
    Dwarf_Half address_size, length_size, extension_size, offset_size, header_cu_type;
    Dwarf_Sig8 type_sig;
    Dwarf_Unsigned next_cu_header, type_offset;
    while (dwarf_next_cu_header_d(dbg, true, &cu_header_length, &version_stamp, &abbrev_offset,
                                &address_size, &length_size, &extension_size, &type_sig, &type_offset,
                                &next_cu_header, &header_cu_type, &err) == DW_DLV_OK) {
        Dwarf_Die cu_die = 0;
        if (dwarf_siblingof_b(dbg, NULL, true, &cu_die, &err) != DW_DLV_OK) {
            continue;
        }

        //enumerate_subprograms(dbg, cu_die);

        // Get line table for this CU
        Dwarf_Line* linebuf;
        Dwarf_Signed linecount;
        Dwarf_Unsigned     dw_version_out;
        Dwarf_Small        dw_table_count;
        Dwarf_Line_Context dw_linecontext;
        //if (dwarf_srclines_b(cu_die, &linebuf, &linecount, &err) != DW_DLV_OK) {
        if (dwarf_srclines_b(cu_die, &dw_version_out, &dw_table_count, &dw_linecontext, &err) != DW_DLV_OK) {
            dwarf_dealloc(dbg, cu_die, DW_DLA_DIE);
            continue;
        }

        // Get source file list
        char** srcfiles;
        Dwarf_Signed filecount;
        if (dwarf_srcfiles(cu_die, &srcfiles, &filecount, &err) != DW_DLV_OK) {
            dwarf_srclines_dealloc_b(dw_linecontext);
            dwarf_dealloc(dbg, cu_die, DW_DLA_DIE);
            continue;
        }

        // Iterate through line table entries
        Dwarf_Addr prev_addr = 0;
        if (dwarf_srclines_from_linecontext(dw_linecontext, &linebuf, &linecount, &err) == DW_DLV_OK){
            for (Dwarf_Signed i = 0; i < linecount; i++) {
                Dwarf_Addr addr;
                if (dwarf_lineaddr(linebuf[i], &addr, &err) != DW_DLV_OK) {
                    continue;
                }

                // Skip if address is zero or same as previous (avoid duplicates)
                if (addr == 0 || addr == prev_addr) {
                    prev_addr = addr;
                    continue;
                }

                Dwarf_Unsigned lineno;
                Dwarf_Unsigned colno = 0;  // Default to 0 if not available
                Dwarf_Signed file_index;
                char *file_name;

                if (dwarf_lineno(linebuf[i], &lineno, &err) == DW_DLV_OK &&
                    dwarf_linesrc(linebuf[i], &file_name, &err) == DW_DLV_OK) {
                    std::string fileName = file_name ? file_name : "unknown";
                    if (file_name)
                        dwarf_dealloc(dbg, file_name, DW_DLA_STRING);

                    // Get column number if available
                    dwarf_lineoff_b(linebuf[i], &colno, &err);  // Ignore error if column not present

                    // Add to map (overwrite if address already exists)
                    addressMap[addr] = SourceLocation(fileName, lineno, colno);
                    //printf("%s,%llu,%llu,%p\n", fileName.c_str(), lineno, colno, (void *)addr);
                }

                prev_addr = addr;
            }
        }

        // Clean up line table and source files
        for (Dwarf_Signed i = 0; i < filecount; i++) {
            dwarf_dealloc(dbg, srcfiles[i], DW_DLA_STRING);
        }
        dwarf_dealloc(dbg, srcfiles, DW_DLA_LIST);
        dwarf_srclines_dealloc_b(dw_linecontext);
        dwarf_dealloc(dbg, cu_die, DW_DLA_DIE);
    }

    // Final cleanup
    // Define a macro to handle the different signatures of dwarf_finish
    #if defined(UBUNTU_LIBDWARF) // Define this macro for Ubuntu builds
    dwarf_finish(dbg, &err);
    #else // Default to RHEL9 or other versions
    dwarf_finish(dbg);
    #endif
    close(fd);

    if (offset != 0)
        unlink(inFileName.c_str());

    return true;
}


// Helper function to resolve type information from a type DIE
std::string resolveTypeName(Dwarf_Debug dbg, Dwarf_Die type_die, Dwarf_Error *err) {
    if (!type_die) {
        return "void";
    }

    Dwarf_Half tag;
    if (dwarf_tag(type_die, &tag, err) != DW_DLV_OK) {
        return "unknown";
    }

    // Handle pointer types
    if (tag == DW_TAG_pointer_type) {
        Dwarf_Attribute type_attr;
        if (dwarf_attr(type_die, DW_AT_type, &type_attr, err) == DW_DLV_OK) {
            Dwarf_Off type_offset;
            if (dwarf_global_formref(type_attr, &type_offset, err) == DW_DLV_OK) {
                Dwarf_Die pointee_die;
                if (dwarf_offdie_b(dbg, type_offset, true, &pointee_die, err) == DW_DLV_OK) {
                    std::string pointee_type = resolveTypeName(dbg, pointee_die, err);
                    dwarf_dealloc(dbg, pointee_die, DW_DLA_DIE);
                    dwarf_dealloc_attribute(type_attr);
                    return pointee_type + "*";
                }
            }
            dwarf_dealloc_attribute(type_attr);
        }
        return "void*";
    }

    // Handle const/volatile qualifiers
    if (tag == DW_TAG_const_type || tag == DW_TAG_volatile_type || tag == DW_TAG_restrict_type) {
        std::string qualifier = (tag == DW_TAG_const_type) ? "const " :
                               (tag == DW_TAG_volatile_type) ? "volatile " : "restrict ";

        Dwarf_Attribute type_attr;
        if (dwarf_attr(type_die, DW_AT_type, &type_attr, err) == DW_DLV_OK) {
            Dwarf_Off type_offset;
            if (dwarf_global_formref(type_attr, &type_offset, err) == DW_DLV_OK) {
                Dwarf_Die qualified_die;
                if (dwarf_offdie_b(dbg, type_offset, true, &qualified_die, err) == DW_DLV_OK) {
                    std::string qualified_type = resolveTypeName(dbg, qualified_die, err);
                    dwarf_dealloc(dbg, qualified_die, DW_DLA_DIE);
                    dwarf_dealloc_attribute(type_attr);
                    return qualifier + qualified_type;
                }
            }
            dwarf_dealloc_attribute(type_attr);
        }
        return qualifier + "void";
    }

    // Handle typedef
    if (tag == DW_TAG_typedef) {
        char *typedef_name = NULL;
        Dwarf_Attribute name_attr;
        if (dwarf_attr(type_die, DW_AT_name, &name_attr, err) == DW_DLV_OK) {
            if (dwarf_formstring(name_attr, &typedef_name, err) == DW_DLV_OK && typedef_name) {
                std::string result(typedef_name);
                dwarf_dealloc_attribute(name_attr);
                return result;
            }
            dwarf_dealloc_attribute(name_attr);
        }
    }

    // Handle base types
    if (tag == DW_TAG_base_type) {
        char *type_name = NULL;
        Dwarf_Attribute name_attr;
        if (dwarf_attr(type_die, DW_AT_name, &name_attr, err) == DW_DLV_OK) {
            if (dwarf_formstring(name_attr, &type_name, err) == DW_DLV_OK && type_name) {
                std::string result(type_name);
                dwarf_dealloc_attribute(name_attr);
                return result;
            }
            dwarf_dealloc_attribute(name_attr);
        }
        return "int";  // default fallback
    }

    // Handle structure/class types
    if (tag == DW_TAG_structure_type || tag == DW_TAG_class_type) {
        char *struct_name = NULL;
        Dwarf_Attribute name_attr;
        if (dwarf_attr(type_die, DW_AT_name, &name_attr, err) == DW_DLV_OK) {
            if (dwarf_formstring(name_attr, &struct_name, err) == DW_DLV_OK && struct_name) {
                std::string prefix = (tag == DW_TAG_structure_type) ? "struct " : "class ";
                std::string result = prefix + std::string(struct_name);
                dwarf_dealloc_attribute(name_attr);
                return result;
            }
            dwarf_dealloc_attribute(name_attr);
        }
        return (tag == DW_TAG_structure_type) ? "struct <anonymous>" : "class <anonymous>";
    }

    // Handle array types
    if (tag == DW_TAG_array_type) {
        std::string element_type = "unknown";
        std::string dimensions;

        // Get the element type
        Dwarf_Attribute type_attr;
        if (dwarf_attr(type_die, DW_AT_type, &type_attr, err) == DW_DLV_OK) {
            Dwarf_Off type_offset;
            if (dwarf_global_formref(type_attr, &type_offset, err) == DW_DLV_OK) {
                Dwarf_Die elem_die;
                if (dwarf_offdie_b(dbg, type_offset, true, &elem_die, err) == DW_DLV_OK) {
                    element_type = resolveTypeName(dbg, elem_die, err);
                    dwarf_dealloc(dbg, elem_die, DW_DLA_DIE);
                }
            }
            dwarf_dealloc_attribute(type_attr);
        }

        // Get array dimensions (subrange)
        Dwarf_Die child_die = NULL;
        if (dwarf_child(type_die, &child_die, err) == DW_DLV_OK) {
            do {
                Dwarf_Half child_tag;
                if (dwarf_tag(child_die, &child_tag, err) == DW_DLV_OK) {
                    if (child_tag == DW_TAG_subrange_type) {
                        Dwarf_Attribute count_attr;
                        // Try DW_AT_count first
                        if (dwarf_attr(child_die, DW_AT_count, &count_attr, err) == DW_DLV_OK) {
                            Dwarf_Unsigned count;
                            if (dwarf_formudata(count_attr, &count, err) == DW_DLV_OK) {
                                dimensions = "[" + std::to_string(count) + "]";
                            }
                            dwarf_dealloc_attribute(count_attr);
                        }
                        // Try DW_AT_upper_bound if count not present
                        else if (dwarf_attr(child_die, DW_AT_upper_bound, &count_attr, err) == DW_DLV_OK) {
                            Dwarf_Unsigned upper;
                            if (dwarf_formudata(count_attr, &upper, err) == DW_DLV_OK) {
                                dimensions = "[" + std::to_string(upper + 1) + "]";
                            }
                            dwarf_dealloc_attribute(count_attr);
                        }
                        else {
                            dimensions = "[]";
                        }
                    }
                }

                Dwarf_Die sibling_die = NULL;
                if (dwarf_siblingof_b(dbg, child_die, true, &sibling_die, err) != DW_DLV_OK) {
                    break;
                }
                dwarf_dealloc(dbg, child_die, DW_DLA_DIE);
                child_die = sibling_die;
            } while (child_die != NULL);

            if (child_die) {
                dwarf_dealloc(dbg, child_die, DW_DLA_DIE);
            }
        }

        if (dimensions.empty()) {
            dimensions = "[]";
        }

        return element_type + dimensions;
    }

    return "unknown";
}

// Helper function to get type size
size_t getTypeSize(Dwarf_Debug dbg, Dwarf_Die type_die, Dwarf_Error *err) {
    if (!type_die) {
        return 0;
    }

    Dwarf_Attribute size_attr;
    if (dwarf_attr(type_die, DW_AT_byte_size, &size_attr, err) == DW_DLV_OK) {
        Dwarf_Unsigned size;
        if (dwarf_formudata(size_attr, &size, err) == DW_DLV_OK) {
            dwarf_dealloc_attribute(size_attr);
            return static_cast<size_t>(size);
        }
        dwarf_dealloc_attribute(size_attr);
    }

    // For pointer types, return pointer size (typically 8 bytes on 64-bit systems)
    Dwarf_Half tag;
    if (dwarf_tag(type_die, &tag, err) == DW_DLV_OK) {
        if (tag == DW_TAG_pointer_type) {
            return 8;  // Assume 64-bit pointers for GPU code
        }

        // For array types, calculate size as element_size * count
        if (tag == DW_TAG_array_type) {
            size_t element_size = 0;
            size_t array_count = 0;

            // Get element type size
            Dwarf_Attribute type_attr;
            if (dwarf_attr(type_die, DW_AT_type, &type_attr, err) == DW_DLV_OK) {
                Dwarf_Off type_offset;
                if (dwarf_global_formref(type_attr, &type_offset, err) == DW_DLV_OK) {
                    Dwarf_Die elem_die;
                    if (dwarf_offdie_b(dbg, type_offset, true, &elem_die, err) == DW_DLV_OK) {
                        element_size = getTypeSize(dbg, elem_die, err);
                        dwarf_dealloc(dbg, elem_die, DW_DLA_DIE);
                    }
                }
                dwarf_dealloc_attribute(type_attr);
            }

            // Get array count from subrange
            Dwarf_Die child_die = NULL;
            if (dwarf_child(type_die, &child_die, err) == DW_DLV_OK) {
                Dwarf_Half child_tag;
                if (dwarf_tag(child_die, &child_tag, err) == DW_DLV_OK) {
                    if (child_tag == DW_TAG_subrange_type) {
                        Dwarf_Attribute count_attr;
                        if (dwarf_attr(child_die, DW_AT_count, &count_attr, err) == DW_DLV_OK) {
                            Dwarf_Unsigned count;
                            if (dwarf_formudata(count_attr, &count, err) == DW_DLV_OK) {
                                array_count = static_cast<size_t>(count);
                            }
                            dwarf_dealloc_attribute(count_attr);
                        }
                        else if (dwarf_attr(child_die, DW_AT_upper_bound, &count_attr, err) == DW_DLV_OK) {
                            Dwarf_Unsigned upper;
                            if (dwarf_formudata(count_attr, &upper, err) == DW_DLV_OK) {
                                array_count = static_cast<size_t>(upper + 1);
                            }
                            dwarf_dealloc_attribute(count_attr);
                        }
                    }
                }
                dwarf_dealloc(dbg, child_die, DW_DLA_DIE);
            }

            if (element_size > 0 && array_count > 0) {
                return element_size * array_count;
            }
        }
    }

    return 0;
}

// Forward declaration for recursive struct member extraction
void extractStructMembers(Dwarf_Debug dbg, Dwarf_Die struct_die, std::vector<KernelArgument>& members, Dwarf_Error *err);

// Helper function to extract struct members recursively
void extractStructMembers(Dwarf_Debug dbg, Dwarf_Die struct_die, std::vector<KernelArgument>& members, Dwarf_Error *err) {
    if (!struct_die) return;

    // Get the first child (member)
    Dwarf_Die member_die = NULL;
    if (dwarf_child(struct_die, &member_die, err) != DW_DLV_OK) {
        return;
    }

    do {
        Dwarf_Half tag;
        if (dwarf_tag(member_die, &tag, err) == DW_DLV_OK) {
            if (tag == DW_TAG_member) {
                // Extract member name
                char *member_name = NULL;
                std::string memberNameStr = "<anonymous>";
                Dwarf_Attribute name_attr;
                if (dwarf_attr(member_die, DW_AT_name, &name_attr, err) == DW_DLV_OK) {
                    if (dwarf_formstring(name_attr, &member_name, err) == DW_DLV_OK && member_name) {
                        memberNameStr = member_name;
                    }
                    dwarf_dealloc_attribute(name_attr);
                }

                // Extract member offset
                size_t offset = 0;
                Dwarf_Attribute offset_attr;
                if (dwarf_attr(member_die, DW_AT_data_member_location, &offset_attr, err) == DW_DLV_OK) {
                    Dwarf_Unsigned off;
                    if (dwarf_formudata(offset_attr, &off, err) == DW_DLV_OK) {
                        offset = static_cast<size_t>(off);
                    }
                    dwarf_dealloc_attribute(offset_attr);
                }

                // Extract member type
                std::string memberType = "unknown";
                size_t memberSize = 0;
                std::vector<KernelArgument> nestedMembers;

                Dwarf_Attribute type_attr;
                if (dwarf_attr(member_die, DW_AT_type, &type_attr, err) == DW_DLV_OK) {
                    Dwarf_Off type_offset;
                    if (dwarf_global_formref(type_attr, &type_offset, err) == DW_DLV_OK) {
                        Dwarf_Die type_die;
                        if (dwarf_offdie_b(dbg, type_offset, true, &type_die, err) == DW_DLV_OK) {
                            memberType = resolveTypeName(dbg, type_die, err);
                            memberSize = getTypeSize(dbg, type_die, err);

                            // Check if this is a struct/class type - if so, recursively extract its members
                            Dwarf_Half type_tag;
                            if (dwarf_tag(type_die, &type_tag, err) == DW_DLV_OK) {
                                if (type_tag == DW_TAG_structure_type || type_tag == DW_TAG_class_type) {
                                    extractStructMembers(dbg, type_die, nestedMembers, err);
                                }
                            }

                            dwarf_dealloc(dbg, type_die, DW_DLA_DIE);
                        }
                    }
                    dwarf_dealloc_attribute(type_attr);
                }

                // Create and add the member using KernelArgument
                // For nested members: offset is meaningful, position/alignment are 0
                KernelArgument member(memberNameStr, memberType, memberSize, offset, 0, 0);
                member.members = nestedMembers;
                members.push_back(member);
            }
        }

        // Move to next sibling (next member)
        Dwarf_Die member_sibling = NULL;
        if (dwarf_siblingof_b(dbg, member_die, true, &member_sibling, err) != DW_DLV_OK) {
            break;
        }
        dwarf_dealloc(dbg, member_die, DW_DLA_DIE);
        member_die = member_sibling;
    } while (member_die != NULL);

    if (member_die) {
        dwarf_dealloc(dbg, member_die, DW_DLA_DIE);
    }
}

bool extractKernelArguments(const char* filename, size_t offset, size_t hsaco_length,
                            std::map<std::string, std::vector<KernelArgument>>& kernelArgsMap) {

    std::string inFileName = filename;

    if (offset != 0) {
        inFileName = create_temp_file_segment(inFileName, offset, hsaco_length);
    }

    // Open the file
    int fd = open(inFileName.c_str(), O_RDONLY);
    if (fd < 0) {
        if (offset != 0) {
            unlink(inFileName.c_str());
        }
        throw std::runtime_error("Failed to open file for DWARF extraction");
    }

    // Initialize DWARF
    Dwarf_Debug dbg;
    Dwarf_Error err;
    int status = 0;

    #if defined(UBUNTU_LIBDWARF)
    status = dwarf_init_b(fd, DW_DLC_READ, DW_GROUPNUMBER_ANY, NULL, NULL, &dbg, &err);
    #else
    status = dwarf_init_b(fd, DW_GROUPNUMBER_ANY, NULL, NULL, &dbg, &err);
    #endif

    if (status == DW_DLV_ERROR) {
        close(fd);
        if (offset != 0) {
            unlink(inFileName.c_str());
        }
        throw std::runtime_error(std::string("DWARF init failed: ") + dwarf_errmsg(err));
    } else if (status == DW_DLV_NO_ENTRY) {
        close(fd);
        if (offset != 0) {
            unlink(inFileName.c_str());
        }
        return false;
    }

    // Iterate through all compilation units
    Dwarf_Unsigned cu_header_length;
    Dwarf_Half version_stamp;
    Dwarf_Unsigned abbrev_offset;
    Dwarf_Half address_size, length_size, extension_size, offset_size, header_cu_type;
    Dwarf_Sig8 type_sig;
    Dwarf_Unsigned next_cu_header, type_offset;

    while (dwarf_next_cu_header_d(dbg, true, &cu_header_length, &version_stamp, &abbrev_offset,
                                &address_size, &length_size, &extension_size, &type_sig, &type_offset,
                                &next_cu_header, &header_cu_type, &err) == DW_DLV_OK) {

        Dwarf_Die cu_die = 0;
        if (dwarf_siblingof_b(dbg, NULL, true, &cu_die, &err) != DW_DLV_OK) {
            continue;
        }

        // Traverse all DIEs in this CU to find subprograms (kernels)
        Dwarf_Die child_die = NULL;
        int result = dwarf_child(cu_die, &child_die, &err);

        if (result == DW_DLV_OK) {
            do {
                Dwarf_Half tag;
                if (dwarf_tag(child_die, &tag, &err) == DW_DLV_OK) {
                    if (tag == DW_TAG_subprogram) {
                        // Found a subprogram (kernel function)
                        char *kernel_name = NULL;
                        Dwarf_Attribute name_attr;

                        // Try DW_AT_linkage_name first (contains mangled name for templates/C++)
                        bool found_name = false;
                        if (dwarf_attr(child_die, DW_AT_linkage_name, &name_attr, &err) == DW_DLV_OK) {
                            if (dwarf_formstring(name_attr, &kernel_name, &err) == DW_DLV_OK && kernel_name) {
                                found_name = true;
                            }
                            dwarf_dealloc_attribute(name_attr);
                        }

                        // Fall back to DW_AT_MIPS_linkage_name (older DWARF versions)
                        if (!found_name && dwarf_attr(child_die, DW_AT_MIPS_linkage_name, &name_attr, &err) == DW_DLV_OK) {
                            if (dwarf_formstring(name_attr, &kernel_name, &err) == DW_DLV_OK && kernel_name) {
                                found_name = true;
                            }
                            dwarf_dealloc_attribute(name_attr);
                        }

                        // Fall back to DW_AT_name if no linkage name
                        if (!found_name && dwarf_attr(child_die, DW_AT_name, &name_attr, &err) == DW_DLV_OK) {
                            if (dwarf_formstring(name_attr, &kernel_name, &err) == DW_DLV_OK && kernel_name) {
                                found_name = true;
                            }
                            dwarf_dealloc_attribute(name_attr);
                        }

                        if (found_name && kernel_name) {
                                std::string kernelNameStr(kernel_name);
                                std::vector<KernelArgument> args;

                                // Now find formal parameters (arguments) of this kernel
                                Dwarf_Die param_die = NULL;
                                int param_result = dwarf_child(child_die, &param_die, &err);
                                uint32_t arg_position = 0;

                                if (param_result == DW_DLV_OK) {
                                    do {
                                        Dwarf_Half param_tag;
                                        if (dwarf_tag(param_die, &param_tag, &err) == DW_DLV_OK) {
                                            if (param_tag == DW_TAG_formal_parameter) {
                                                // Extract parameter information
                                                char *param_name = NULL;
                                                Dwarf_Attribute param_name_attr;

                                                std::string paramNameStr = "";
                                                if (dwarf_attr(param_die, DW_AT_name, &param_name_attr, &err) == DW_DLV_OK) {
                                                    if (dwarf_formstring(param_name_attr, &param_name, &err) == DW_DLV_OK && param_name) {
                                                        paramNameStr = param_name;
                                                    }
                                                    dwarf_dealloc_attribute(param_name_attr);
                                                }

                                                // Get type information
                                                std::string typeStr = "unknown";
                                                size_t typeSize = 0;
                                                size_t alignment = 0;

                                                Dwarf_Attribute type_attr;
                                                if (dwarf_attr(param_die, DW_AT_type, &type_attr, &err) == DW_DLV_OK) {
                                                    Dwarf_Off type_offset;
                                                    if (dwarf_global_formref(type_attr, &type_offset, &err) == DW_DLV_OK) {
                                                        Dwarf_Die type_die;
                                                        if (dwarf_offdie_b(dbg, type_offset, true, &type_die, &err) == DW_DLV_OK) {
                                                            typeStr = resolveTypeName(dbg, type_die, &err);
                                                            typeSize = getTypeSize(dbg, type_die, &err);

                                                            // Get alignment if available
                                                            Dwarf_Attribute align_attr;
                                                            if (dwarf_attr(type_die, DW_AT_alignment, &align_attr, &err) == DW_DLV_OK) {
                                                                Dwarf_Unsigned align;
                                                                if (dwarf_formudata(align_attr, &align, &err) == DW_DLV_OK) {
                                                                    alignment = static_cast<size_t>(align);
                                                                }
                                                                dwarf_dealloc_attribute(align_attr);
                                                            }

                                                            // Create kernel argument (offset is 0 for top-level args)
                                                            KernelArgument arg(paramNameStr, typeStr, typeSize, 0, alignment, arg_position);

                                                            // If this is a struct/class type (not a pointer), extract its members
                                                            Dwarf_Half type_tag;
                                                            if (dwarf_tag(type_die, &type_tag, &err) == DW_DLV_OK) {
                                                                if (type_tag == DW_TAG_structure_type || type_tag == DW_TAG_class_type) {
                                                                    extractStructMembers(dbg, type_die, arg.members, &err);
                                                                }
                                                            }

                                                            args.push_back(arg);

                                                            dwarf_dealloc(dbg, type_die, DW_DLA_DIE);
                                                        }
                                                    }
                                                    dwarf_dealloc_attribute(type_attr);
                                                }

                                                arg_position++;
                                            }
                                        }

                                        // Move to next sibling (next parameter)
                                        Dwarf_Die param_sibling = NULL;
                                        if (dwarf_siblingof_b(dbg, param_die, true, &param_sibling, &err) != DW_DLV_OK) {
                                            break;
                                        }
                                        dwarf_dealloc(dbg, param_die, DW_DLA_DIE);
                                        param_die = param_sibling;
                                    } while (param_die != NULL);

                                    if (param_die) {
                                        dwarf_dealloc(dbg, param_die, DW_DLA_DIE);
                                    }
                                }

                                // Store kernel arguments
                                kernelArgsMap[kernelNameStr] = args;
                        }
                    }
                }

                // Move to next sibling
                Dwarf_Die sibling_die = NULL;
                if (dwarf_siblingof_b(dbg, child_die, true, &sibling_die, &err) != DW_DLV_OK) {
                    break;
                }
                dwarf_dealloc(dbg, child_die, DW_DLA_DIE);
                child_die = sibling_die;
            } while (child_die != NULL);

            if (child_die) {
                dwarf_dealloc(dbg, child_die, DW_DLA_DIE);
            }
        }

        dwarf_dealloc(dbg, cu_die, DW_DLA_DIE);
    }

    // Cleanup
    #if defined(UBUNTU_LIBDWARF)
    dwarf_finish(dbg, &err);
    #else
    dwarf_finish(dbg);
    #endif
    close(fd);

    if (offset != 0) {
        unlink(inFileName.c_str());
    }

    return !kernelArgsMap.empty();
}

