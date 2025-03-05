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
  ${LIB_DIR}/kernelDB.cc
)


include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Default ROCm path can be infered
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH})
else()
  set(ROCM_PATH "/opt/rocm")
endif()
message("ROCM_PATH: ${ROCM_PATH}")
set(ROCM_LLVM "${ROCM_PATH}/llvm" CACHE PATH "Path to the ROCm LLVM directory")

link_directories(${ROCM_ROOT_DIR}/lib $ENV{HOME}/.local/lib64 ${ROCM_LLVM}/lib)
add_library ( ${TARGET_LIB} SHARED ${LIB_SRC})

target_include_directories (
    ${TARGET_LIB}
  PRIVATE
    ${LIB_DIR}
    ${ROCM_ROOT_DIR}/include
    ${ROOT_DIR}
    ${HSA_RUNTIME_INC_PATH}
    ${HSA_KMT_LIB_PATH}/..
    ${ROCM_LLVM}/include
)

execute_process(
    COMMAND /bin/sh -c "${ROCM_LLVM}/bin/llvm-config --libs all | awk '{for (i=1; i<=NF; i++) printf(\"%s;\",substr($i, 3))}'"
    OUTPUT_VARIABLE LLVM_LIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "LLVM Link Libraries: ${LLVM_LIBS}")

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
    ${LLVM_LIBS}
)

file(GLOB_RECURSE KERNELDB_INCLUDE_FILES "include/*.h")
set_target_properties(${TARGET_LIB} PROPERTIES PUBLIC_HEADER "${KERNELDB_INCLUDE_FILES}")
include(GNUInstallDirs)
install(TARGETS ${TARGET_LIB}
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/kerneldb)

message(STATUS "==> {Library,include} install dirs: ${CMAKE_INSTALL_PREFIX}/{${CMAKE_INSTALL_LIBDIR},${CMAKE_INSTALL_INCLUDEDIR}/kerneldb}")
message(STATUS "==> To change install dirs, rerun cmake with argument \"-DCMAKE_INSTALL_PREFIX=/path/to/install/dir\"")
