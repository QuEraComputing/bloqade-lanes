cmake_minimum_required(VERSION 3.14)
project(bloqade_c_smoke_test C)

# --- Paths to the Rust-built library and header ---
# These are passed in by the caller (justfile / CI):
#   -DBLOQADE_LIB_DIR=<path to directory containing the shared/static lib>
#   -DBLOQADE_INCLUDE_DIR=<path to directory containing bloqade_lanes_bytecode.h>

if(NOT DEFINED BLOQADE_LIB_DIR)
    message(FATAL_ERROR "BLOQADE_LIB_DIR must be set (path to Rust-built library)")
endif()

if(NOT DEFINED BLOQADE_INCLUDE_DIR)
    message(FATAL_ERROR "BLOQADE_INCLUDE_DIR must be set (path to bloqade_lanes_bytecode.h)")
endif()

# Find the static library explicitly (avoids resolving to the shared library
# when both .so/.dylib and .a are present, which would reintroduce rpath issues).
set(_ORIG_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
if(WIN32)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif()

find_library(BLOQADE_LIB
    NAMES bloqade_lanes_bytecode
    PATHS "${BLOQADE_LIB_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
)

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_ORIG_SUFFIXES})

message(STATUS "Found bloqade library: ${BLOQADE_LIB}")
message(STATUS "Include directory: ${BLOQADE_INCLUDE_DIR}")

# Build the smoke test executable
add_executable(smoke_test smoke.c)
target_include_directories(smoke_test PRIVATE "${BLOQADE_INCLUDE_DIR}")
target_link_libraries(smoke_test PRIVATE "${BLOQADE_LIB}")

# Platform-specific system libraries required by the Rust static lib
if(UNIX AND NOT APPLE)
    target_link_libraries(smoke_test PRIVATE pthread dl m)
elseif(APPLE)
    target_link_libraries(smoke_test PRIVATE "-framework Security" "-framework CoreFoundation")
endif()

# Register as a CTest
enable_testing()
add_test(NAME c_ffi_smoke COMMAND smoke_test)
