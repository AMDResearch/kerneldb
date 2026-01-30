################################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

# Install prefix
set(CMAKE_INSTALL_PREFIX "/" CACHE PATH "Install path prefix default")

# Verbose output.
set(CMAKE_VERBOSE_MAKEFILE TRUE CACHE BOOL "Verbose Output" FORCE)

# Adding default path cmake modules
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")

set(CMAKE_CXX_FLAGS_DEBUG_INIT "-ggdb")

include(utils)

get_version("1.0.0")

if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH})
else()
    # Set a default value if ROCM_PATH is not set
    set(ROCM_PATH "/opt/rocm")
endif()

message("ROCM_PATH: ${ROCM_PATH}")

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

find_program(
    HIPCC_EXECUTABLE
    #NAMES g++
    NAMES hipcc
    HINTS ${CMAKE_INSTALL_PREFIX}
    PATHS ${CMAKE_INSTALL_PREFIX}
    PATH_SUFFIXES bin)

set(CMAKE_CXX_COMPILER
    ${HIPCC_EXECUTABLE}
    CACHE FILEPATH "C++ compiler" FORCE)

# Set module name and project name.
set(KERNELDB_NAME "kernelDB")
set(KERNELDB_TARGET "kernelDB64")
set(INTERCEPTOR_LIBRARY "lib${KERNELDB_TARGET}")

project(
    ${KERNELDB_NAME}
    LANGUAGES CXX HIP
    VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}
    DESCRIPTION "Library for exposing CDNA kernels as a queryable databqase"
    HOMEPAGE_URL "https://github.com/AARInternal/kerneldb")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    set(KERNELDB_TOPLEVEL_PROJECT ON)
else()
    set(KERNELDB_TOPLEVEL_PROJECT OFF)
endif()

find_package(hip REQUIRED)

if(INTERCEPTOR_TOPLEVEL_PROJECT)
    set(CMAKE_CXX_STANDARD
        20
        CACHE STRING "Description")
    option(CMAKE_CXX_STANDARD_REQUIRED "C++ required standard" ON)
    option(CMAKE_CXX_EXTENSIONS "C++ extensions" ON)
else()
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# Set build environment
include ( env )

set(BUILD_VERSION_MAJOR ${VERSION_MAJOR})
set(BUILD_VERSION_MINOR ${VERSION_MINOR})
set(BUILD_VERSION_PATCH ${VERSION_PATCH})
if(DEFINED VERSION_BUILD AND NOT ${VERSION_BUILD} STREQUAL "")
  message("VERSION BUILD DEFINED ${VERSION_BUILD}")
  set(BUILD_VERSION_PATCH "${BUILD_VERSION_PATCH}-${VERSION_BUILD}")
endif()
set(BUILD_VERSION_STRING "${BUILD_VERSION_MAJOR}.${BUILD_VERSION_MINOR}.${BUILD_VERSION_PATCH}")

set(LIB_VERSION_MAJOR ${VERSION_MAJOR})
set(LIB_VERSION_MINOR ${VERSION_MINOR})
if(${ROCM_PATCH_VERSION})
    set(LIB_VERSION_PATCH ${ROCM_PATCH_VERSION})
else()
    set(LIB_VERSION_PATCH ${VERSION_PATCH})
endif()
set(LIB_VERSION_STRING "${LIB_VERSION_MAJOR}.${LIB_VERSION_MINOR}.${LIB_VERSION_PATCH}")
message("-- LIB-VERSION STRING: ${LIB_VERSION_STRING}")

# Set target and root/lib/test directory
set(TARGET_NAME "${KERNELDB_TARGET}")
set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LIB_DIR "${ROOT_DIR}/src")
set(TEST_DIR "${ROOT_DIR}/test")

# Enable tracing API
if (NOT USE_PROF_API)
  set(USE_PROF_API 1)
endif()

# Find the libdwarf headers
find_path(LIBDWARF_INCLUDE_DIR
  NAMES libdwarf.h dwarf.h
  HINTS ${CMAKE_INCLUDE_PATH}
  PATHS
    ${CMAKE_INCLUDE_PATH}
    /usr/include/libdwarf-0
    /usr/include/libdwarf
    /usr/local/include/libdwarf-0
    /usr/local/include/libdwarf
  DOC "Path to libdwarf header files")

if(NOT LIBDWARF_INCLUDE_DIR)
  message(FATAL_ERROR "libdwarf headers not found. Please install libdwarf or specify the include path.")
else()
  message(STATUS "Found libdwarf headers in: ${LIBDWARF_INCLUDE_DIR}")
  include_directories(${LIBDWARF_INCLUDE_DIR})
endif()

# Check the Linux distribution by examining /etc/os-release
# Certain macros are defined for Ubuntu
if(EXISTS "/etc/os-release")
    file(READ "/etc/os-release" OS_RELEASE_CONTENT)
    string(FIND "${OS_RELEASE_CONTENT}" "ID=ubuntu" FOUND_UBUNTU)
    if(NOT FOUND_UBUNTU EQUAL -1)
        message(STATUS "Detected Ubuntu Linux distribution")
        add_definitions(-DUBUNTU_LIBDWARF)
    else()
        message(STATUS "Detected non-Ubuntu Linux distribution (e.g., RHEL9 or other)")
    endif()
else()
    message(WARNING "/etc/os-release not found. Unable to determine Linux distribution.")
endif()

# Fallback option to allow the user to manually specify libdwarf version
option(USE_UBUNTU_LIBDWARF "Use Ubuntu-specific libdwarf version" OFF)
if(USE_UBUNTU_LIBDWARF)
    add_definitions(-DUBUNTU_LIBDWARF)
endif()

# Build libraries
include(${LIB_DIR}/CMakeLists.txt)
include(test/CMakeLists.txt)

# Set the VERSION and SOVERSION values
set_property(TARGET ${TARGET_NAME} PROPERTY VERSION "${LIB_VERSION_STRING}")
set_property(TARGET ${TARGET_NAME} PROPERTY SOVERSION "${LIB_VERSION_MAJOR}")
# If the library is a release, strip the target library
if("${CMAKE_BUILD_TYPE}" STREQUAL release)
  add_custom_command(TARGET ${KERNELDB_TARGET} POST_BUILD
    COMMAND ${CMAKE_STRIP} $<TARGET_FILE:${KERNELDB_TARGET}>
    COMMENT "Stripping ${KERNELDB_TARGET}")
endif()

# Build tests
option(KERNELDB_BUILD_TESTING "Build test examples" OFF)
if(KERNELDB_BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(examples)
    add_subdirectory(test)
endif()

# Installation and packaging
set(DEST_NAME ${KERNELDB_NAME})
if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
  get_filename_component(pkg_name ${CPACK_PACKAGING_INSTALL_PREFIX} NAME)
  get_filename_component(pkg_dir ${CPACK_PACKAGING_INSTALL_PREFIX} DIRECTORY)
  if(pkg_name STREQUAL ${DEST_NAME})
    set(CPACK_PACKAGING_INSTALL_PREFIX ${pkg_dir})
  endif()
else()
  set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()
message("CMake-install-prefix: ${CMAKE_INSTALL_PREFIX}")
message("CPack-install-prefix: ${CPACK_PACKAGING_INSTALL_PREFIX}")
message("-----------Dest-name: ${DEST_NAME}")

# Install header and library
#install(TARGETS ${KERNELDB_TARGET} LIBRARY DESTINATION ${DEST_NAME}/lib)
#install(FILES ${LIB_DIR}/kerneldb DESTINATION ${DEST_NAME}/bin)

# Packaging directives
set(CPACK_GENERATOR "DEB" "RPM" "TGZ")
set(CPACK_PACKAGE_NAME "${KERNELDB_NAME}-dev")
set(CPACK_PACKAGE_VENDOR "AMD" )
set(CPACK_PACKAGE_VERSION_MAJOR ${BUILD_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${BUILD_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${BUILD_VERSION_PATCH})
set(CPACK_PACKAGE_CONTACT "Advanced Micro Devices Inc.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CDNA Kernels as Queryable Databases")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

# Debian package specific variables
set(CPACK_DEBIAN_PACKAGE_DEPENDS "hsa-rocr-dev, rocminfo")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/postinst;${CMAKE_CURRENT_SOURCE_DIR}/DEBIAN/prerm")

# RPM package specific variables
set(CPACK_RPM_PACKAGE_DEPENDS "hsa-rocr-dev, rocminfo")
set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_post")
set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_postun")

include(CPack)

