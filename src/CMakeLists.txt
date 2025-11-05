################################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

set ( TARGET_LIB "${TARGET_NAME}" )
set ( LIB_SRC
#  ${LIB_DIR}/utils.cc
  ${LIB_DIR}/kernelDB.cc
  ${LIB_DIR}/addressMap.cc
  ${LIB_DIR}/disassemble.cc
  ${LIB_DIR}/co_extract.cc
)


include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Default ROCm path can be infered
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH})
else()
  set(ROCM_PATH "/opt/rocm")
endif()
message("ROCM_PATH: ${ROCM_PATH}")

link_directories(${ROCM_ROOT_DIR}/lib $ENV{HOME}/.local/lib64)
add_library ( ${TARGET_LIB} SHARED ${LIB_SRC})

target_include_directories (
    ${TARGET_LIB}
  PUBLIC
    ${LIB_DIR}
    ${ROCM_ROOT_DIR}/include
    ${ROOT_DIR}
    ${HSA_RUNTIME_INC_PATH}
    ${HSA_KMT_LIB_PATH}/..
    ${LIBDWARF_INCLUDE_DIR}
)


target_compile_options(
    ${TARGET_LIB}
  PRIVATE
    -O0
)

target_link_libraries(
    ${TARGET_LIB}
  PRIVATE
    ${HSA_RUNTIME_LIB}
    c
    stdc++
    stdc++fs
    amd_comgr
    z
    zstd
    dwarf
)
# Install the kernelDB library and public headers
file(GLOB_RECURSE KERNELDB_INCLUDE_FILES "include/*.h")
set_target_properties(${TARGET_LIB} PROPERTIES PUBLIC_HEADER "${KERNELDB_INCLUDE_FILES}")
include(GNUInstallDirs)
install(TARGETS ${TARGET_LIB}
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/kerneldb)

# Install all libdwarf headers
if(LIBDWARF_INCLUDE_DIR)
    install(DIRECTORY ${LIBDWARF_INCLUDE_DIR}/
            DESTINATION ${CMAKE_INSTALL_PREFIX}/include/kerneldb
            FILES_MATCHING PATTERN "*.h")
    message(STATUS "libdwarf headers installed to ${CMAKE_INSTALL_PREFIX}/include/libdwarf")
else()
    message(WARNING "libdwarf headers not found, not installing.")
endif()

message(STATUS "==> {Library,include} install dirs: ${CMAKE_INSTALL_PREFIX}/{${CMAKE_INSTALL_LIBDIR},${CMAKE_INSTALL_INCLUDEDIR}/kerneldb}")
message(STATUS "==> To change install dirs, rerun cmake with argument \"-DCMAKE_INSTALL_PREFIX=/path/to/install/dir\"")

# Build Python bindings if requested
option(BUILD_PYTHON_BINDINGS "Build Python bindings with pybind11" OFF)
if(BUILD_PYTHON_BINDINGS)
    # Find Python - make development optional
    find_package(Python3 COMPONENTS Interpreter QUIET)
    find_package(Python3 COMPONENTS Development.Module QUIET)

    # Try to find pybind11
    find_package(pybind11 CONFIG QUIET)

    if(pybind11_FOUND AND Python3_Development.Module_FOUND)
        message(STATUS "Found pybind11: ${pybind11_DIR}")
        message(STATUS "Found Python3 Development: ${Python3_VERSION}")

        # Create pybind11 module
        pybind11_add_module(_kerneldb ${LIB_DIR}/pybind11_wrapper.cc)

        # Link against kernelDB library
        target_link_libraries(_kerneldb PRIVATE ${TARGET_LIB})

        # Set RPATH so the module can find the C++ library in the same directory
        # Use CMAKE_LIBRARY_OUTPUT_DIRECTORY so it respects setup.py's choice
        set_target_properties(_kerneldb PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            INSTALL_RPATH "$ORIGIN"
            BUILD_WITH_INSTALL_RPATH TRUE
        )

        # Install the module
        install(TARGETS _kerneldb
            LIBRARY DESTINATION kerneldb)

        message(STATUS "Python bindings will be built")
    else()
        if(NOT pybind11_FOUND)
            message(WARNING "pybind11 not found. Python bindings will not be built.")
            message(WARNING "Install with: pip install pybind11")
        endif()
        if(NOT Python3_Development.Module_FOUND)
            message(WARNING "Python development headers not found. Python bindings will not be built.")
            message(WARNING "Install with: dnf install python3-devel (RHEL) or apt install python3-dev (Ubuntu)")
        endif()
    endif()
endif()
