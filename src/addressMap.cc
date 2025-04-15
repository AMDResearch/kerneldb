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

template <typename T>
std::optional<Dwarf_Addr> find_floor_key(const std::map<Dwarf_Addr, T>& map, Dwarf_Addr target) {
    if (map.empty()) {
        return std::nullopt; // No keys in the map
    }

    auto it = map.upper_bound(target); // First key > target
    if (it == map.begin()) {
        return std::nullopt; // No key <= target (all keys are > target)
    }

    // Move to the key <= target (previous iterator)
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
    if (dwarf_init_b(fd, DW_GROUPNUMBER_ANY, NULL, NULL, &dbg, &err) != DW_DLV_OK) {
        close(fd);
        throw std::runtime_error(std::string("DWARF init failed: ") + dwarf_errmsg(err));
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
    dwarf_finish(dbg);
    close(fd);

    if (offset != 0)
        unlink(inFileName.c_str());

    return true;
}

