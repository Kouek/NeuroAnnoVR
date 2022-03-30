cmake_minimum_required(VERSION 3.10)

project(NeuroAnnoVR LANGUAGES CXX C CUDA)

option(${PROJECT_NAME}_BUILD_DEMO "Build all demos" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(THIRDPARTY_DIR "${CMAKE_CURRENT_LIST_DIR}/3rdparty")
set(INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include")

# cfg injection
configure_file("${INCLUDE_DIR}/CMakeIn.h.in" "${INCLUDE_DIR}/CMakeIn.h")

# Qt dep
## Note 1:
## Append CMAKE_PREFIX_PATH with your Qt 5 installation prefix
## eg. In Windows, -DCMAKE_PREFIX_PATH="D:\Qt\Qt5.9.8\5.9.8\msvc2017_64\lib\cmake"
## Note 2:
## Make sure Qt5 shared libs path are in ENV_PATH
## eg. In Windows, append system/user PATH with "D:\Qt\Qt5.9.8\5.9.8\msvc2017_64\bin"
foreach(mod "Core" "Gui" "Widgets")
	find_package("Qt5" COMPONENTS ${mod} REQUIRED)
	list(APPEND Qt5_LIBS "Qt5::${mod}")
endforeach()
message(STATUS "Qt5_LIBS: ${Qt5_LIBS}")
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# CUDA dep
## No need to find_package("CUDA" REQUIRED) any more since CMake 3.9
message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

# OpenVR dep
set(OPENVR_INC_DIR "${THIRDPARTY_DIR}/openvr/headers")
set(OPENVR_BIN_DIR "${THIRDPARTY_DIR}/openvr/bin")
set(OPENVR_LIB_DIR "${THIRDPARTY_DIR}/openvr/lib")

# GLM dep
include("${THIRDPARTY_DIR}/glm.cmake")

# VolumeCompression dep
include("${THIRDPARTY_DIR}/VolumeCompression.cmake")

# Utils
include("${THIRDPARTY_DIR}/spdlog.cmake")
include_directories("${THIRDPARTY_DIR}/glad/include")
include_directories(${THIRDPARTY_DIR})

add_subdirectory("src")
