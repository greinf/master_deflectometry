cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(open_camera_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message (STATUS "[${PROJECT_NAME}] Processing ${CMAKE_CURRENT_LIST_FILE}")

# ---------------- Vimba X ----------------
# Cache variable (so VS / presets can set it)
set(VIMBAX_ROOT "${VIMBAX_ROOT}" CACHE PATH "Root of Vimba X installation")

# Allow setting via environment variable VIMBAX_ROOT
if(NOT DEFINED VIMBAX_ROOT AND DEFINED ENV{VIMBAX_ROOT})
    set(VIMBAX_ROOT "$ENV{VIMBAX_ROOT}")
endif()

message(STATUS "DEBUG: VIMBAX_ROOT='${VIMBAX_ROOT}'")
message(STATUS "DEBUG: CMAKE_CACHEFILE_DIR='${CMAKE_CACHEFILE_DIR}'")

if(NOT EXISTS "${VIMBAX_ROOT}/api/include")
    message(FATAL_ERROR "Invalid VIMBAX_ROOT: '${VIMBAX_ROOT}'. Set it via CMake preset, VS cache, or env var.")
endif()

# Define dirs FIRST
set(VIMBAX_INCLUDE_DIR "${VIMBAX_ROOT}/api/include")
set(VIMBAX_LIB_DIR     "${VIMBAX_ROOT}/api/lib")
set(VIMBAX_BIN_DIR     "${VIMBAX_ROOT}/bin")

message(STATUS "DEBUG: VIMBAX_LIB_DIR='${VIMBAX_LIB_DIR}'")
message(STATUS "DEBUG: VIMBAX_BIN_DIR='${VIMBAX_BIN_DIR}'")

# Find libs (robust to minor naming differences)
find_library(VIMBAX_VMBCPP_LIB NAMES VmbCPP PATHS "${VIMBAX_LIB_DIR}" NO_DEFAULT_PATH)
if(NOT VIMBAX_VMBCPP_LIB)
    message(FATAL_ERROR "VimbaX: VmbCPP.lib not found in '${VIMBAX_LIB_DIR}'")
endif()

find_library(VIMBAX_VMBC_LIB NAMES VmbC PATHS "${VIMBAX_LIB_DIR}" NO_DEFAULT_PATH)

# DLL paths
set(VIMBAX_VMBCPP_DLL "${VIMBAX_BIN_DIR}/VmbCPP.dll")
set(VIMBAX_VMBC_DLL   "${VIMBAX_BIN_DIR}/VmbC.dll")

# Imported target
add_library(VimbaX::VmbCPP UNKNOWN IMPORTED)
set_target_properties(VimbaX::VmbCPP PROPERTIES
    IMPORTED_LOCATION "${VIMBAX_VMBCPP_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${VIMBAX_INCLUDE_DIR}"
)

# Optionally link VmbC if present
if(VIMBAX_VMBC_LIB)
    set_property(TARGET VimbaX::VmbCPP APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES "${VIMBAX_VMBC_LIB}"
    )
endif()


# Setup target executable with the same name as our project
add_executable (${PROJECT_NAME}
    #src/camera.cpp
    src/screen.cpp
    src/main.cpp
    src/deflectometry.cpp
    src/imgProcessing.cpp
    src/imageStore.cpp
    "src/cameraN.cpp" 
    "src/VimbaBackend.cpp" 
    "src/GrayCalibration.cpp" 
    "src/GrayCodeDecoder.cpp" 
    "src/CameraSimulation.cpp" 
    "src/CameraCalibration.cpp" 
    "src/GeometricCalibration.cpp" 
    "src/PSF.cpp"  "MainScriptsReady/Raycaster.cpp")


#Debug 
message(STATUS "DEBUG: VCPKG_INSTALLED_DIR='${VCPKG_INSTALLED_DIR}'")
message(STATUS "DEBUG: VCPKG_TARGET_TRIPLET='${VCPKG_TARGET_TRIPLET}'")
message(STATUS "DEBUG: CMAKE_PREFIX_PATH='${CMAKE_PREFIX_PATH}'")


# --- Find packages ---
#if (NOT TARGET ids_peak)
#    find_package (ids_peak REQUIRED
#        HINTS 
#            ../../../../../../../lib/
#    )

#endif ()

find_package (Threads REQUIRED)
find_package(Ceres CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
set(Open3D_DIR "C:/Users/grein/Documents/libraries/open3d-devel-windows-amd64-0.19.0/CMake")
find_package(Open3D REQUIRED)

message(STATUS "Open3D_FOUND='${Open3D_FOUND}'")
message(STATUS "Open3D_VERSION='${Open3D_VERSION}'")
message(STATUS "Open3D_DIR='${Open3D_DIR}'")

get_target_property(Open3D_IMPORTED_CONFIGS Open3D::Open3D IMPORTED_CONFIGURATIONS)
message(STATUS "Open3D IMPORTED_CONFIGURATIONS='${Open3D_IMPORTED_CONFIGS}'")

get_target_property(Open3D_LOCATION_RELEASE Open3D::Open3D IMPORTED_LOCATION_RELEASE)
get_target_property(Open3D_LOCATION_DEBUG   Open3D::Open3D IMPORTED_LOCATION_DEBUG)
get_target_property(Open3D_LOCATION         Open3D::Open3D IMPORTED_LOCATION)

message(STATUS "Open3D IMPORTED_LOCATION_RELEASE='${Open3D_LOCATION_RELEASE}'")
message(STATUS "Open3D IMPORTED_LOCATION_DEBUG='${Open3D_LOCATION_DEBUG}'")
message(STATUS "Open3D IMPORTED_LOCATION='${Open3D_LOCATION}'")

get_target_property(Open3D_LINK_LIBS Open3D::Open3D INTERFACE_LINK_LIBRARIES)
message(STATUS "Open3D INTERFACE_LINK_LIBRARIES='${Open3D_LINK_LIBS}'")

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/Users/grein/Documents/libraries/open3d-devel-windows-amd64-0.19.0/bin/Open3D.dll"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/Users/grein/Documents/libraries/open3d-devel-windows-amd64-0.19.0/bin/tbb12.dll"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
)


find_package(Boost CONFIG REQUIRED)

# Link against libraries
target_link_libraries (${PROJECT_NAME}
    #ids_peak
    ${CMAKE_THREAD_LIBS_INIT}
    ${OpenCV_LIBS}
    VimbaX::VmbCPP 
    Ceres::ceres
    Boost::boost
    Open3D::Open3D
)


# Add include directory (root)
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/config
	${CMAKE_CURRENT_SOURCE_DIR}/include/RayTracer
        ${CMAKE_CURRENT_SOURCE_DIR}/include/runtime
	${VIMBAX_INCLUDE_DIR} 
)


# Call deploy function
# This function will add a post-build steps to your target in order to copy all needed files (e.g. DLL's) to the output directory.
#ids_peak_deploy(${PROJECT_NAME})

# ---------------- Vimba X runtime deploy (ADD) ----------------
if (WIN32 AND (NOT VIMBAX_ROOT OR VIMBAX_ROOT STREQUAL ""))
    if (EXISTS "C:/Program Files/Allied Vision/Vimba X/api/include")
        set(VIMBAX_ROOT "C:/Program Files/Allied Vision/Vimba X" CACHE PATH "" FORCE)
    elseif (EXISTS "C:/Program Files/Allied Vision/VimbaX/api/include")
        set(VIMBAX_ROOT "C:/Program Files/Allied Vision/VimbaX" CACHE PATH "" FORCE)
    endif()
endif()

# Copy VimbaX DLLs next to the executable so the app can start.
if(EXISTS "${VIMBAX_VMBCPP_DLL}")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${VIMBAX_VMBCPP_DLL}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    )
else()
    message(WARNING "VimbaX: ${VIMBAX_VMBCPP_DLL} not found. App may fail at runtime unless PATH contains VimbaX/bin.")
endif()

if(EXISTS "${VIMBAX_VMBC_DLL}")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${VIMBAX_VMBC_DLL}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    )
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS NO
)

# Enable multiprocessing for MSVC
if (MSVC)
    target_compile_options (${PROJECT_NAME}
        PRIVATE "/MP"
    )
endif ()

if (MSVC)
    add_compile_definitions(NOMINMAX)
endif()