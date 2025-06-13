INCLUDE_DIRECTORIES(./)

FILE(GLOB summer_mbt_src *.hh *.cc *.cpp *.h *.hpp)
source_group("" FILES ${summer_mbt_src})

FILE(GLOB glsl_SOURCES shader/*.glsl)
source_group("shader" FILES ${glsl_SOURCES})

add_library(summer_mbt ${summer_mbt_src} ${glsl_SOURCES})
target_link_libraries(summer_mbt ${DEP_LIBS} UTILS_LIB)