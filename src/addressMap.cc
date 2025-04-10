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
    
    /*off_t new_offset = lseek(fd, offset, SEEK_SET);
    if (new_offset == (off_t)-1) {
        perror("lseek failed");
        close(fd);
        throw std::runtime_error("Unable to read " + std::string(filename));
    }*/

    // Initialize DWARF
    Dwarf_Debug dbg;
    Dwarf_Error err;
    if (dwarf_init_b(fd, DW_GROUPNUMBER_ANY, NULL, NULL, &dbg, &err) != DW_DLV_OK) {
        close(fd);
        throw std::runtime_error(std::string("DWARF init failed: ") + dwarf_errmsg(err));
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
    /*while (dwarf_next_cu_header_d(dbg, NULL, &cu_header_length, &version_stamp,
                                 &abbrev_offset, &address_size, NULL, NULL,
                                 &next_cu_header, NULL, &err) == DW_DLV_OK) {*/
        Dwarf_Die cu_die = 0;
        if (dwarf_siblingof_b(dbg, NULL, true, &cu_die, &err) != DW_DLV_OK) {
            continue;
        }

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

