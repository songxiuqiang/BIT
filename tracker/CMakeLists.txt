cmake_minimum_required(VERSION 3.5)

project(AR3DVEngine)
message(STATUS "BIT")

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_INCLUDE_CURRENT_DIR ON)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

option(ENABLE_UTILS "enable ar3dv utils" ON)
option(ENABLE_TRACKER "enable ar3dv tracker" ON)
option(ENABLE_DETECTOR "enable ar3dv detector" OFF)
option(ENABLE_RECONSTRUCTOR "enable ar3dv reconstructor" OFF)
option(ENABLE_INTERFACE "enable ar3dv interface" ON)
option(ENABLE_UI "enable ar3dv UI" OFF)

FIND_PACKAGE(Glog REQUIRED)
FIND_PACKAGE(Gflags REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(Qt5Widgets REQUIRED)
FIND_PACKAGE(Qt5OpenGL REQUIRED)

if (OPENMP_FOUND)
    if (WIN32)
        set(CMAKE_CXX_FLAGS_RELEASE
                "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS_RELEASE
                "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS
                "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS
                "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endIf ()
endif ()

if(ENABLE_UTILS)
    add_subdirectory(ar_utils)
endif(ENABLE_UTILS)

if(ENABLE_TRACKER)
    option(ENABLE_SUMMER "enable summer" ON)
	if(ENABLE_SUMMER)
		add_definitions(-DACTIVE_SUMMER)
	endif(ENABLE_SUMMER)

    add_subdirectory(tracker)
endif(ENABLE_TRACKER)

if(ENABLE_DETECTOR)
    add_subdirectory(detector)
endif(ENABLE_DETECTOR)

if(ENABLE_RECONSTRUCTOR)
    add_subdirectory(reconstructor)
endif(ENABLE_RECONSTRUCTOR)

if(ENABLE_UI)
    add_subdirectory(ar3dv_ui)
endif(ENABLE_UI)

if(ENABLE_INTERFACE)
    add_subdirectory(interface)
endif(ENABLE_INTERFACE)

include_directories(./)

add_subdirectory(example)
