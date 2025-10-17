cmake_minimum_required(VERSION 3.2...3.12 FATAL_ERROR)
project(open_camera_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message (STATUS "[${PROJECT_NAME}] Processing ${CMAKE_CURRENT_LIST_FILE}")

# Setup target executable with the same name as our project
add_executable (${PROJECT_NAME}
    src/camera.cpp
    src/screen.cpp
    src/main.cpp
    src/deflectometry.cpp
)

# Find packages
if (NOT TARGET ids_peak)
    find_package (ids_peak REQUIRED
        HINTS 
            ../../../../../../../lib/
    )
endif ()
find_package (Threads REQUIRED)
# For vcpkg manifest mode (local install)
set(OpenCV_ROOT "${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/share/opencv4")
find_package(OpenCV REQUIRED)

# Link against libraries
target_link_libraries (${PROJECT_NAME}
    ids_peak
    ${CMAKE_THREAD_LIBS_INIT}
    ${OpenCV_LIBS}
)

# Call deploy function
# This function will add a post-build steps to your target in order to copy all needed files (e.g. DLL's) to the output directory.
ids_peak_deploy(${PROJECT_NAME})

# Set C++ standard to 14 (required for ids_peak)
set_target_properties(${PROJECT_NAME} PROPERTIES
    #CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS NO
)

# Add include path
include_directories(${CMAKE_SOURCE_DIR}/include)

# Enable multiprocessing for MSVC
if (MSVC)
    target_compile_options (${PROJECT_NAME}
        PRIVATE "/MP"
    )
endif ()

# For unix Build we need the environment variable GENICAM_GENTL32_PATH respectivily GENICAM_GENTL64_PATH to find the GenTL producer libraries.
# To set these environment variables a shell script is used. This script can be automatically generated via ids_peak_generate_starter_script.
# The shell script will be saved at ${CMAKE_CURRENT_BINARY_DIR}/${targetName}.sh and automatically copied to the output directory during post-build.

# To run the samples run this script, not the binary.
if(UNIX)
    ids_peak_generate_starter_script(${PROJECT_NAME})
endif()
