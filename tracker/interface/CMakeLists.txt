include_directories(
		./
		../
		../detector
		../ar_utils
)

FIND_PACKAGE(ASSIMP REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(Glog REQUIRED)
FIND_PACKAGE(Gflags REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5OpenGL REQUIRED)


include_directories(
	Qt5::Widgets 
	Qt5::OpenGL
	${OPENGL_INCLUDE_DIR}
	${ASSIMP_INCLUDE_DIR}
	${EIGEN3_INCLUDE_DIR}
)

FILE(GLOB INTERFACE_FILES
		./*.cc
		./*.cpp
		./tracker_rbot/*.cc
		./tracker_slot/*.cc
		./tracker_summer/*.cc
		./tracker_awlb/*.cc
		./tracker_icg/*.cpp
		./tracker_srt3d/*.cpp
		./tracker_rbgt/*.cpp
		./tracker_rmt/*.cpp
		./tracker_null/*.cc
		./detector_null/*.cpp
		./detector_linemod/*.cpp
		)

add_library(INTERFACE_LIB
  ${INTERFACE_FILES}
)

target_link_libraries(INTERFACE_LIB
	Qt5::Widgets
	Qt5::OpenGL
	UTILS_LIB
)

if(ENABLE_RBOT)
	target_link_libraries(INTERFACE_LIB RBOT_LIB)
endif(ENABLE_RBOT)

if(ENABLE_SLOT)
	target_link_libraries(INTERFACE_LIB SLOT_LIB)
endif(ENABLE_SLOT)

if(ENABLE_SUMMER)
	target_link_libraries(INTERFACE_LIB SUMMER_LIB)
endif(ENABLE_SUMMER)


if(ENABLE_AWLB)
	target_link_libraries(INTERFACE_LIB AWLB_LIB)
endif(ENABLE_AWLB)

if(ENABLE_ICG)
	target_link_libraries(INTERFACE_LIB ICG_LIB)
endif(ENABLE_ICG)

if(ENABLE_SRT3D)
	target_link_libraries(INTERFACE_LIB SRT3D_LIB)
endif(ENABLE_SRT3D)

if(ENABLE_RBGT)
	target_link_libraries(INTERFACE_LIB RBGT_LIB)
endif(ENABLE_RBGT)

if(ENABLE_RMT)
	target_link_libraries(INTERFACE_LIB RMT_LIB)
endif(ENABLE_RMT)

if(ENABLE_DETECTOR)
	target_link_libraries(INTERFACE_LIB DETECTOR_LIB)
endif(ENABLE_DETECTOR)
