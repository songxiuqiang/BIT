cmake_minimum_required(VERSION 2.8.11)

set(PROJECTNAME Summer)
project(${PROJECTNAME})

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

find_package(Qt5Widgets REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(assimp REQUIRED)
find_package(glog REQUIRED)
find_package(OpenGL REQUIRED)
find_package(TBB REQUIRED tbb)

set(DEP_LIBS
    glog::glog 
    Qt5::Widgets 
    Qt5::OpenGL
	TBB::tbb
    ${OpenCV_LIBS}
    ${OPENGL_LIBRARIES}
    ${ASSIMP_LIBRARIES}
)

include_directories(
    src
    ${OpenCV_INCLUDE_DIRS}
	${TBB_INCLUDE_DIRS} 
    ${OPENGL_INCLUDE_DIR}
    ${ASSIMP_INCLUDE_DIR}
)

link_directories(${PROJECT_ROOT}/lib)

add_subdirectory(src/base)
add_subdirectory(src/mbt)

file(GLOB SOURCES 
    src/base/*.cc src/base/*.cpp src/base/*.h src/base/*.hh 
    src/mbt/*.cc src/mbt/*.cpp src/mbt/*.h src/mbt/*.hh 
    src/mbt/*.cc src/mbt/*.cpp src/mbt/*.h src/mbt/*.hh 
    src/mbt/shader/*.glsl
)

add_library(SUMMER_LIB ${SOURCES})
target_link_libraries(SUMMER_LIB ${DEP_LIBS} -lglfw3 -lX11 -lXxf86vm -ldl -lpthread  assimp)

