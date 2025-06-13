cmake_minimum_required(VERSION 3.5)

project(Utils)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "add a postfix, usually d on windows")
set(CMAKE_RELEASE_POSTFIX "" CACHE STRING "add a postfix, usually empty on windows")
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS ${COMPILE_FLAG})
set(CMAKE_C_FLAGS ${COMPILE_FLAG})

FIND_PACKAGE(OpenCV 4.0 REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED NO_MODULE)
FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(Qt5Widgets REQUIRED)
FIND_PACKAGE(Qt5OpenGL REQUIRED)
FIND_PACKAGE(ASSIMP REQUIRED)
FIND_PACKAGE(Glog REQUIRED)
FIND_PACKAGE(Gflags REQUIRED)
FIND_PACKAGE(Sophus REQUIRED)
find_package(nlohmann_json REQUIRED)

INCLUDE_DIRECTORIES(
		../
        ./
        ${EIGEN3_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR}
        ${ASSIMP_INCLUDE_DIR}
        ${Sophus_INCLUDE_DIRS}
)

set(UTILS_DEP_LIB
        glog::glog
        Qt5::Widgets
        Qt5::OpenGL
		Sophus::Sophus
        ${OpenCV_LIBS}
        ${OPENGL_LIBRARIES}
        ${ASSIMP_LIBRARIES}
		nlohmann_json::nlohmann_json
        )

FILE(GLOB UTILS_SOURCES
        data_io/data_loader.cpp data_io/data_loader.h
        model/*.cpp model/*.h
        renderer/*.cpp renderer/*.h renderer/glad/glad.c
		pose_converter/*.cpp pose_converter/*.h
		timer/*.cpp timer/*.h
		video_img_processor/*.cpp video_img_processor/*.h
		visualizer/*.cpp visualizer/*.h
        )
		
add_library(UTILS_LIB ${UTILS_SOURCES})
target_link_libraries(UTILS_LIB ${UTILS_DEP_LIB})